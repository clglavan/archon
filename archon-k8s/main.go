package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"k8s-debugger/k8stools"
	"k8s-debugger/lmstudioproviders"
)

// ANSI terminal color codes
const (
	colorReset  = "\033[0m"
	colorRed    = "\033[31m"
	colorGreen  = "\033[32m"
	colorYellow = "\033[33m"
	colorBlue   = "\033[34m"
	colorPurple = "\033[35m"
	colorCyan   = "\033[36m"
	colorGray   = "\033[90m"
	colorBold   = "\033[1m"
)

func main() {
	ctx := context.Background()

	// Load configuration from config.env file
	loadEnvFile("config.env")

	// 1. Parse configuration
	lmStudioModelName := os.Getenv("LM_STUDIO_MODEL")
	if lmStudioModelName == "" {
		lmStudioModelName = "magos-k8s-0.6b"
	}

	lmStudioBaseURL := os.Getenv("LM_STUDIO_BASE_URL")
	if lmStudioBaseURL == "" {
		lmStudioBaseURL = "http://localhost:1234"
	}

	// 2. Initialize LM Studio model adapter
	modelAdapter := lmstudioproviders.NewLMStudioModel(lmStudioModelName, lmStudioBaseURL)

	// Print visual banner
	fmt.Print(colorBold + colorCyan)
	fmt.Println("==================================================================")
	fmt.Println("       ☸️   KUBERNETES TRIAGE DEBUGGER DAEMON   ☸️")
	fmt.Printf("   Powered by Agent Development Kit (ADK) Go & LM Studio (%s)\n", lmStudioModelName)
	fmt.Println("==================================================================")
	fmt.Print(colorReset)

	// 3. Initialize Kubernetes client and toolbox
	fmt.Printf("%s[System]%s Loading local kubeconfig...\n", colorGray, colorReset)
	toolbox, err := k8stools.NewK8sToolbox()
	var tools []tool.Tool
	if err != nil {
		log.Fatalf("Failed to initialize Kubernetes client: %v. A working cluster context is required to run the daemon.", err)
	}

	fmt.Printf("%s[System]%s Connected to cluster successfully! Registering diagnostic tools...\n", colorGray, colorReset)
	registeredTools, err := k8stools.RegisterAllTools(toolbox)
	if err != nil {
		log.Fatalf("Failed to register tools: %v", err)
	}
	tools = registeredTools

	// 4. Create the LLM Agent
	agentInstruction := `You are an expert Kubernetes triage debugger. Your goal is to help developers diagnose, inspect, and troubleshoot issues on their Kubernetes clusters.
You have access to real-time cluster query tools.

Systematic Triage Guidelines:
1. Scan for failed events (specifically Warning type events) using 'k8s_get_events' in the target namespace or across the cluster.
2. For any significant warning event, you MUST retrieve the raw configuration of the involved object (e.g. Pod, Service, Deployment, ReplicaSet) as YAML using 'k8s_get_object_yaml'.
3. Analyze the YAML configuration for configuration errors (e.g., mismatched port numbers, missing environment variables, invalid volume mounts, incorrect image tags).
4. Query other correlated events around that same timeframe using 'k8s_get_events' filtered by the objectName to establish a timeline of what went wrong (e.g. failed scheduling followed by pod failure).
5. If the object has container logs (i.e. if it is a Pod), pull the logs using 'k8s_get_pod_logs' to see stdout/stderr error traces.
6. Provide a detailed summary to the user:
   - The warning event details and timeline.
   - Any configuration errors found in the YAML config.
   - Any stack traces or logs.
   - Clear steps to fix the issue.
`
	k8sAgent, err := llmagent.New(llmagent.Config{
		Name:        "k8s-triage-debugger",
		Model:       modelAdapter,
		Instruction: agentInstruction,
		Tools:       tools,
		Description: "Kubernetes troubleshooting assistant capable of querying clusters and diagnosing pod failures.",
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	// 5. Initialize ADK Runner
	runnr, err := runner.New(runner.Config{
		AppName:           "k8s-triage-debugger",
		Agent:             k8sAgent,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		log.Fatalf("Failed to create runner: %v", err)
	}

	// 6. Polling loop setup
	alertCache := make(map[string]time.Time)

	pollInterval := 30 * time.Second
	intervalStr := os.Getenv("POLL_INTERVAL_SECONDS")
	if intervalStr != "" {
		if parsed, err := strconv.Atoi(intervalStr); err == nil {
			pollInterval = time.Duration(parsed) * time.Second
		}
	}

	ttlSeconds := 300
	ttlStr := os.Getenv("ALERT_TTL_SECONDS")
	if ttlStr != "" {
		if parsed, err := strconv.Atoi(ttlStr); err == nil {
			ttlSeconds = parsed
		}
	}
	alertTTL := time.Duration(ttlSeconds) * time.Second

	fmt.Printf("%s[System]%s Monitoring warning events every %s (Alert TTL: %s)...\n", colorGray, colorReset, pollInterval, alertTTL)

	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()

	// Initial scan or wait
	scanEvents(ctx, toolbox, runnr, alertCache, alertTTL)

	for {
		select {
		case <-ctx.Done():
			fmt.Printf("%s[System]%s Daemon shutting down.\n", colorGray, colorReset)
			return
		case <-ticker.C:
			scanEvents(ctx, toolbox, runnr, alertCache, alertTTL)
		}
	}
}

func scanEvents(ctx context.Context, toolbox *k8stools.K8sToolbox, runnr *runner.Runner, alertCache map[string]time.Time, alertTTL time.Duration) {
	// Clean up expired cache entries to prevent memory leak
	now := time.Now()
	for key, expireTime := range alertCache {
		if now.After(expireTime) {
			delete(alertCache, key)
		}
	}

	eventList, err := toolbox.Clientset.CoreV1().Events("").List(ctx, metav1.ListOptions{})
	if err != nil {
		fmt.Printf("%s[Error]%s Failed to list events: %v\n", colorRed, colorReset, err)
		return
	}

	for _, e := range eventList.Items {
		// Only triage Warning events
		if e.Type != "Warning" {
			continue
		}

		// Parse timestamps
		eventTime := e.LastTimestamp.Time
		if eventTime.IsZero() {
			eventTime = e.CreationTimestamp.Time
		}

		// Create a unique deduplication key defining "the same alert"
		alertKey := fmt.Sprintf("%s/%s/%s/%s", e.Namespace, e.InvolvedObject.Kind, e.InvolvedObject.Name, e.Reason)

		// Check if alert is currently within its TTL window
		if expireTime, exists := alertCache[alertKey]; exists && now.Before(expireTime) {
			continue
		}

		// Set/refresh the alert expiration time in the TTL cache
		alertCache[alertKey] = now.Add(alertTTL)

		// Run triage as a concurrent goroutine
		go triageEvent(ctx, runnr, e, eventTime)
	}
}

func triageEvent(ctx context.Context, runnr *runner.Runner, e corev1.Event, eventTime time.Time) {
	fmt.Printf("%s[System]%s Warning event detected on %s/%s. Starting triage session...\n", colorGray, colorReset, e.InvolvedObject.Kind, e.InvolvedObject.Name)

	prompt := fmt.Sprintf("A Warning event was detected on object %s/%s in namespace %q.\n"+
		"Reason: %s\n"+
		"Message: %s\n"+
		"Last Observed: %s\n"+
		"Count: %d\n\n"+
		"Please check the config of the object using k8s_get_object_yaml, pull its logs using k8s_get_pod_logs (if it's a pod), trace related events, and write a detailed analysis explaining the root cause and remedy.",
		e.InvolvedObject.Kind, e.InvolvedObject.Name, e.Namespace,
		e.Reason, e.Message, eventTime.Format("2006-01-02 15:04:05"), e.Count)

	userContent := &genai.Content{
		Role:  "user",
		Parts: []*genai.Part{{Text: prompt}},
	}

	sessionID := fmt.Sprintf("triage-%s", string(e.UID))
	userID := "event-monitor"

	var reportBuilder strings.Builder
	reportBuilder.WriteString("=========================================\n")
	reportBuilder.WriteString("⚠️ KUBERNETES WARNING EVENT DETECTED\n")
	reportBuilder.WriteString("=========================================\n")
	reportBuilder.WriteString(fmt.Sprintf("Namespace: %s\n", e.Namespace))
	reportBuilder.WriteString(fmt.Sprintf("Object:    %s/%s\n", e.InvolvedObject.Kind, e.InvolvedObject.Name))
	reportBuilder.WriteString(fmt.Sprintf("Reason:    %s\n", e.Reason))
	reportBuilder.WriteString(fmt.Sprintf("Message:   %s\n", e.Message))
	reportBuilder.WriteString(fmt.Sprintf("Time:      %s\n", eventTime.Format("2006-01-02 15:04:05")))
	reportBuilder.WriteString(fmt.Sprintf("Count:     %d\n", e.Count))
	reportBuilder.WriteString("-----------------------------------------\n")
	reportBuilder.WriteString("🔍 AUTOMATED TRIAGE REPORT:\n")

	for event, err := range runnr.Run(ctx, userID, sessionID, userContent, agent.RunConfig{}) {
		if err != nil {
			reportBuilder.WriteString(fmt.Sprintf("\nTriage Error: %v\n", err))
			break
		}

		if event.LLMResponse.Content != nil {
			for _, part := range event.LLMResponse.Content.Parts {
				if part.Text != "" {
					reportBuilder.WriteString(part.Text)
				}
				if part.FunctionCall != nil {
					reportBuilder.WriteString(fmt.Sprintf("\n[⚡ Running tool: %s with args %v]\n", part.FunctionCall.Name, part.FunctionCall.Args))
				}
			}
		}
	}

	reportBuilder.WriteString("\n=========================================\n")
	dispatchAlerts(reportBuilder.String())
}

func sendGoogleChat(url, text string) error {
	payload := map[string]string{"text": text}
	body, _ := json.Marshal(payload)
	resp, err := http.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("google chat status %d: %s", resp.StatusCode, string(b))
	}
	return nil
}

func sendTelegram(token, chatID, text string) error {
	url := fmt.Sprintf("https://api.telegram.org/bot%s/sendMessage", token)
	payload := map[string]any{
		"chat_id": chatID,
		"text":    text,
	}
	body, _ := json.Marshal(payload)
	resp, err := http.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("telegram status %d: %s", resp.StatusCode, string(b))
	}
	return nil
}

func dispatchAlerts(report string) {
	// 1. Log to standard console stdout
	fmt.Println(report)

	// 2. Google Chat webhook alert
	gchatURL := os.Getenv("GOOGLE_CHAT_WEBHOOK_URL")
	if gchatURL != "" {
		if err := sendGoogleChat(gchatURL, report); err != nil {
			fmt.Printf("%s[Alert Error]%s Failed to send Google Chat: %v\n", colorRed, colorReset, err)
		} else {
			fmt.Printf("%s[System]%s Sent alert to Google Chat.\n", colorGray, colorReset)
		}
	}

	// 3. Telegram chat bot alert
	tgToken := os.Getenv("TELEGRAM_BOT_TOKEN")
	tgChatID := os.Getenv("TELEGRAM_CHAT_ID")
	if tgToken != "" && tgChatID != "" {
		if err := sendTelegram(tgToken, tgChatID, report); err != nil {
			fmt.Printf("%s[Alert Error]%s Failed to send Telegram: %v\n", colorRed, colorReset, err)
		} else {
			fmt.Printf("%s[System]%s Sent alert to Telegram.\n", colorGray, colorReset)
		}
	}
}

func loadEnvFile(path string) {
	file, err := os.Open(path)
	if err != nil {
		return // Ignore if config file doesn't exist
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		val := strings.TrimSpace(parts[1])
		val = strings.Trim(val, `"'`)
		os.Setenv(key, val)
	}
}
