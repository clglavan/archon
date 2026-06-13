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
	agentInstruction := `You are an expert Kubernetes triage debugger. Your goal is to help developers diagnose, inspect, and troubleshoot warning events on their Kubernetes clusters.
You MUST follow this triage checklist step-by-step:
1. First, retrieve the raw configuration of the involved object as YAML using 'k8s_get_object_yaml'. You MUST call this tool immediately before writing any text or recommendations.
2. If the object is a Pod, retrieve its container logs using 'k8s_get_pod_logs' to check for error traces or stack traces.
3. Query other correlated events using 'k8s_get_events' filtered by the object's name to establish a timeline of issues.
4. Compile your findings from the above tool responses and write a detailed analysis explaining the root cause and the exact remediation steps.

Strict Rules:
- NEVER guess the root cause or output a triage analysis without executing 'k8s_get_object_yaml' first.
- Calling tools to inspect the active resources is your primary task. Always call the tools before summarizing.
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

	maxToolCalls := 5
	if maxStr := os.Getenv("MAX_TOOL_CALLS_PER_SESSION"); maxStr != "" {
		if val, err := strconv.Atoi(maxStr); err == nil && val > 0 {
			maxToolCalls = val
		}
	}

	sessionCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	prompt := fmt.Sprintf("A Warning event was detected on object %s/%s in namespace %q.\n"+
		"Reason: %s\n"+
		"Message: %s\n"+
		"Last Observed: %s\n"+
		"Count: %d\n\n"+
		"IMPORTANT: You MUST call k8s_get_object_yaml with the exact name %q and namespace %q. Do not change or translate the name (e.g. do not change \"failing\" to \"failed\").\n"+
		"Please check the config of the object using k8s_get_object_yaml, pull its logs using k8s_get_pod_logs (if it's a pod), trace related events, and write a detailed analysis explaining the root cause and remedy.",
		e.InvolvedObject.Kind, e.InvolvedObject.Name, e.Namespace,
		e.Reason, e.Message, eventTime.Format("2006-01-02 15:04:05"), e.Count,
		e.InvolvedObject.Name, e.Namespace)

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

	toolCallCount := 0
	seenToolCalls := make(map[string]bool)

	for event, err := range runnr.Run(sessionCtx, userID, sessionID, userContent, agent.RunConfig{}) {
		if err != nil {
			reportBuilder.WriteString(fmt.Sprintf("\nTriage Error: %v\n", err))
			break
		}

		if event.LLMResponse.Content != nil {
			var isAborted bool
			for _, part := range event.LLMResponse.Content.Parts {
				if part.Text != "" {
					reportBuilder.WriteString(part.Text)
				}
				if part.FunctionCall != nil {
					toolCallCount++

					// Serialize args for unique signature
					argsBytes, _ := json.Marshal(part.FunctionCall.Args)
					sig := fmt.Sprintf("%s:%s", part.FunctionCall.Name, string(argsBytes))

					if seenToolCalls[sig] {
						fmt.Printf("%s[System] [%s/%s] Loop detected! Tool %s already called with args %s. Aborting session.%s\n",
							colorRed, e.InvolvedObject.Kind, e.InvolvedObject.Name, part.FunctionCall.Name, string(argsBytes), colorReset)
						reportBuilder.WriteString(fmt.Sprintf("\n[⚡ Loop detected! Tool %s already called with args %s. Terminating run.]\n", part.FunctionCall.Name, string(argsBytes)))
						cancel()
						isAborted = true
						break
					}
					seenToolCalls[sig] = true

					if toolCallCount > maxToolCalls {
						fmt.Printf("%s[System] [%s/%s] Exceeded maximum tool calls (%d). Aborting session.%s\n",
							colorRed, e.InvolvedObject.Kind, e.InvolvedObject.Name, maxToolCalls, colorReset)
						reportBuilder.WriteString(fmt.Sprintf("\n[⚡ Exceeded maximum tool calls limit (%d). Terminating run.]\n", maxToolCalls))
						cancel()
						isAborted = true
						break
					}

					// Real-time tool invocation log
					fmt.Printf("%s[System] [%s/%s] AI invoking tool: %s with args %v%s\n",
						colorCyan, e.InvolvedObject.Kind, e.InvolvedObject.Name, part.FunctionCall.Name, part.FunctionCall.Args, colorReset)
					reportBuilder.WriteString(fmt.Sprintf("\n[⚡ Running tool: %s with args %v]\n", part.FunctionCall.Name, part.FunctionCall.Args))
				}
				if part.FunctionResponse != nil {
					// Real-time tool response log
					fmt.Printf("%s[System] [%s/%s] Tool returned response: %s -> %v%s\n",
						colorGreen, e.InvolvedObject.Kind, e.InvolvedObject.Name, part.FunctionResponse.Name, part.FunctionResponse.Response, colorReset)
				}
			}
			if isAborted {
				break
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
