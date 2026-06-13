package k8stools

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	// google.golang.org/adk provides the Agent Development Kit framework.
	// We use its tool package to register Go functions as LLM-callable tools.
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"sigs.k8s.io/yaml"
	
	// k8s.io packages are the official Kubernetes client packages (client-go).
	// They allow us to programmatically interact with a Kubernetes cluster.
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
)

// K8sToolbox acts as the receiver/container for all Kubernetes diagnostic tools.
// We define methods on this struct which we will later register as tools for the AI agent.
type K8sToolbox struct {
	// Clientset provides the API interface to interact with all standard Kubernetes resources.
	Clientset *kubernetes.Clientset
}

// NewK8sToolbox sets up a connection to the Kubernetes cluster and returns an initialized toolbox.
func NewK8sToolbox() (*K8sToolbox, error) {
	clientset, err := getKubeClient()
	if err != nil {
		return nil, err
	}
	return &K8sToolbox{Clientset: clientset}, nil
}

// getKubeClient handles configuring the Kubernetes cluster connection.
// It is designed to work in two modes:
// 1. In-Cluster (if the debugger is running as a Pod inside the cluster).
// 2. Out-of-Cluster (local execution using the user's ~/.kube/config file).
func getKubeClient() (*kubernetes.Clientset, error) {
	// Mode 1: Attempt to load the in-cluster config. This succeeds if the environment
	// has the service account token mounted (default for Pods).
	if config, err := rest.InClusterConfig(); err == nil {
		return kubernetes.NewForConfig(config)
	}

	// Mode 2: Fall back to local kubeconfig configuration.
	// Check the KUBECONFIG env variable first, otherwise default to ~/.kube/config.
	kubeconfig := os.Getenv("KUBECONFIG")
	if kubeconfig == "" {
		if home := homedir.HomeDir(); home != "" {
			kubeconfig = filepath.Join(home, ".kube", "config")
		}
	}
	if kubeconfig == "" {
		return nil, fmt.Errorf("kubeconfig file path not found")
	}

	// Build the REST client config from the local config file.
	config, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
	if err != nil {
		return nil, fmt.Errorf("failed to build kubeconfig: %w", err)
	}
	
	// Instantiate the Clientset interface.
	return kubernetes.NewForConfig(config)
}

// =========================================================================
// 1. List Namespaces Tool
// =========================================================================

// ListNamespacesArgs represents the input schema for the namespace list tool.
// Since listing namespaces requires no inputs, this struct is empty.
type ListNamespacesArgs struct{}

// ListNamespacesResult represents the output structure that will be serialized
// to JSON and returned to the LLM agent after executing the tool.
type ListNamespacesResult struct {
	Namespaces []string `json:"namespaces"`
}

// ListNamespaces queries the Kubernetes cluster API to get a list of all active namespaces.
func (tb *K8sToolbox) ListNamespaces(ctx tool.Context, args ListNamespacesArgs) (ListNamespacesResult, error) {
	// Call CoreV1 API to retrieve namespaces list.
	nsList, err := tb.Clientset.CoreV1().Namespaces().List(ctx, metav1.ListOptions{})
	if err != nil {
		return ListNamespacesResult{}, fmt.Errorf("failed to list namespaces: %w", err)
	}

	// Extract the namespace names into a slice of strings.
	namespaces := make([]string, 0, len(nsList.Items))
	for _, ns := range nsList.Items {
		namespaces = append(namespaces, ns.Name)
	}

	return ListNamespacesResult{Namespaces: namespaces}, nil
}

// =========================================================================
// 2. List Pods Tool
// =========================================================================

// ListPodsArgs defines the parameter struct for listing pods.
// The ADK framework parses Go tags (`json:"..."` and `description:"..."`) using reflection.
// These tags are automatically converted into a JSON Schema that tells the LLM:
// - What parameters to send.
// - Their types and descriptions.
type ListPodsArgs struct {
	Namespace string `json:"namespace" description:"Namespace to list pods. If empty, lists across all namespaces."`
}

// PodInfo defines a simplified diagnostic snapshot of a Pod, returned to the model.
type PodInfo struct {
	Name      string `json:"name"`
	Namespace string `json:"namespace"`
	Status    string `json:"status"` // E.g. Running, Pending, Failed
	IP        string `json:"ip"`
	Restarts  int32  `json:"restarts"` // Total restarts across all containers in the Pod
	Ready     string `json:"ready"`    // E.g. "1/2" container readiness indicator
}

type ListPodsResult struct {
	Pods []PodInfo `json:"pods"`
}

// ListPods lists pods inside the requested namespace (or all namespaces if empty).
func (tb *K8sToolbox) ListPods(ctx tool.Context, args ListPodsArgs) (ListPodsResult, error) {
	// Query the pods list. If args.Namespace is empty, it acts as all-namespaces.
	podList, err := tb.Clientset.CoreV1().Pods(args.Namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return ListPodsResult{}, fmt.Errorf("failed to list pods: %w", err)
	}

	pods := make([]PodInfo, 0, len(podList.Items))
	for _, pod := range podList.Items {
		var restarts int32
		var readyContainers int
		var totalContainers = len(pod.Spec.Containers)

		// Accumulate restart counts and readiness status across init and app containers.
		for _, status := range pod.Status.ContainerStatuses {
			restarts += status.RestartCount
			if status.Ready {
				readyContainers++
			}
		}

		pods = append(pods, PodInfo{
			Name:      pod.Name,
			Namespace: pod.Namespace,
			Status:    string(pod.Status.Phase),
			IP:        pod.Status.PodIP,
			Restarts:  restarts,
			Ready:     fmt.Sprintf("%d/%d", readyContainers, totalContainers),
		})
	}

	return ListPodsResult{Pods: pods}, nil
}

// =========================================================================
// 3. Get Pod Logs Tool
// =========================================================================

// GetPodLogsArgs defines the parameters needed to retrieve container logs.
// The name field maps to the Pod's name, namespace maps to its namespace, and the model
// can optionally specify a target container name and a tail limit.
type GetPodLogsArgs struct {
	Namespace     string `json:"namespace" description:"Namespace of the pod"`
	Name          string `json:"name" description:"Name of the pod"`
	ContainerName string `json:"containerName" description:"Name of the container (optional if pod has only one container)"`
	TailLines     *int64 `json:"tailLines" description:"Number of recent log lines to return (optional)"`
}

type GetPodLogsResult struct {
	Logs string `json:"logs"`
}

// GetPodLogs fetches logs from the specified pod/container.
func (tb *K8sToolbox) GetPodLogs(ctx tool.Context, args GetPodLogsArgs) (GetPodLogsResult, error) {
	opts := &corev1.PodLogOptions{}
	if args.ContainerName != "" {
		opts.Container = args.ContainerName
	}
	if args.TailLines != nil {
		opts.TailLines = args.TailLines
	} else {
		// Default to returning the last 100 log lines to avoid flooding prompt context.
		defaultTail := int64(100)
		opts.TailLines = &defaultTail
	}

	// Initiate the logs stream request to the cluster API.
	req := tb.Clientset.CoreV1().Pods(args.Namespace).GetLogs(args.Name, opts)
	stream, err := req.Stream(ctx)
	if err != nil {
		return GetPodLogsResult{}, fmt.Errorf("failed to stream pod logs: %w", err)
	}
	defer stream.Close()

	// Read stream output into a memory buffer.
	buf := new(bytes.Buffer)
	_, err = io.Copy(buf, stream)
	if err != nil {
		return GetPodLogsResult{}, fmt.Errorf("failed to read pod logs stream: %w", err)
	}

	return GetPodLogsResult{Logs: buf.String()}, nil
}

// =========================================================================
// 4. Describe Pod Tool
// =========================================================================

type DescribePodArgs struct {
	Namespace string `json:"namespace" description:"Namespace of the pod"`
	Name      string `json:"name" description:"Name of the pod"`
}

// ContainerState defines basic properties, restarts, and reasons for a container's state.
type ContainerState struct {
	Name         string `json:"name"`
	Ready        bool   `json:"ready"`
	RestartCount int32  `json:"restartCount"`
	State        string `json:"state"`            // Waiting, Running, or Terminated
	Reason       string `json:"reason,omitempty"`  // E.g. CrashLoopBackOff, ErrImagePull
	Message      string `json:"message,omitempty"` // Detailed error description from the container engine
}

// DescribePodResult mimics a simplified `kubectl describe pod` command output.
// It bundles pod phase, node, container status structures, conditions, and pod-specific events.
type DescribePodResult struct {
	Name       string           `json:"name"`
	Namespace  string           `json:"namespace"`
	Phase      string           `json:"phase"`
	NodeName   string           `json:"nodeName"`
	Containers []ContainerState `json:"containers"`
	Conditions []string         `json:"conditions"`
	Events     []string         `json:"events"`
}

// DescribePod collects diagnostic metrics, container status arrays, and event logs for a specific pod.
func (tb *K8sToolbox) DescribePod(ctx tool.Context, args DescribePodArgs) (DescribePodResult, error) {
	// Query pod details.
	pod, err := tb.Clientset.CoreV1().Pods(args.Namespace).Get(ctx, args.Name, metav1.GetOptions{})
	if err != nil {
		return DescribePodResult{}, fmt.Errorf("failed to get pod details: %w", err)
	}

	// Map container statuses.
	containers := make([]ContainerState, 0, len(pod.Status.ContainerStatuses))
	for _, cs := range pod.Status.ContainerStatuses {
		stateStr := "Unknown"
		reason := ""
		message := ""

		if cs.State.Waiting != nil {
			stateStr = "Waiting"
			reason = cs.State.Waiting.Reason
			message = cs.State.Waiting.Message
		} else if cs.State.Running != nil {
			stateStr = "Running"
		} else if cs.State.Terminated != nil {
			stateStr = "Terminated"
			reason = cs.State.Terminated.Reason
			message = cs.State.Terminated.Message
		}

		containers = append(containers, ContainerState{
			Name:         cs.Name,
			Ready:        cs.Ready,
			RestartCount: cs.RestartCount,
			State:        stateStr,
			Reason:       reason,
			Message:      message,
		})
	}

	// Filter active pod conditions.
	conditions := make([]string, 0, len(pod.Status.Conditions))
	for _, c := range pod.Status.Conditions {
		if c.Status == corev1.ConditionTrue {
			conditions = append(conditions, string(c.Type))
		} else {
			conditions = append(conditions, fmt.Sprintf("%s(False:%s)", c.Type, c.Reason))
		}
	}

	// Retrieve event logs targeting this specific pod object.
	eventList, err := tb.Clientset.CoreV1().Events(args.Namespace).List(ctx, metav1.ListOptions{
		FieldSelector: fmt.Sprintf("involvedObject.name=%s,involvedObject.kind=Pod", args.Name),
	})

	events := make([]string, 0)
	if err == nil {
		for _, e := range eventList.Items {
			events = append(events, fmt.Sprintf("[%s] %s: %s", e.Type, e.Reason, e.Message))
		}
	}

	return DescribePodResult{
		Name:       pod.Name,
		Namespace:  pod.Namespace,
		Phase:      string(pod.Status.Phase),
		NodeName:   pod.Spec.NodeName,
		Containers: containers,
		Conditions: conditions,
		Events:     events,
	}, nil
}

// =========================================================================
// 5. Get Events Tool
// =========================================================================

type GetEventsArgs struct {
	Namespace  string `json:"namespace" description:"Namespace to query events. If empty, queries all namespaces."`
	ObjectName string `json:"objectName,omitempty" description:"Optional name of the object (e.g., my-pod) to filter events."`
}

type EventInfo struct {
	Namespace      string `json:"namespace"`
	Type           string `json:"type"` // Normal or Warning
	Reason         string `json:"reason"`
	Message        string `json:"message"`
	Object         string `json:"object"`
	FirstTimestamp string `json:"firstTimestamp,omitempty"`
	LastTimestamp  string `json:"lastTimestamp,omitempty"`
	Count          int32  `json:"count,omitempty"`
}

type GetEventsResult struct {
	Events []EventInfo `json:"events"`
}

// GetEvents queries the event stream in the cluster, optionally filtering by object name.
func (tb *K8sToolbox) GetEvents(ctx tool.Context, args GetEventsArgs) (GetEventsResult, error) {
	opts := metav1.ListOptions{}
	if args.ObjectName != "" {
		opts.FieldSelector = fmt.Sprintf("involvedObject.name=%s", args.ObjectName)
	}

	eventList, err := tb.Clientset.CoreV1().Events(args.Namespace).List(ctx, opts)
	if err != nil {
		return GetEventsResult{}, fmt.Errorf("failed to list events: %w", err)
	}

	events := make([]EventInfo, 0, len(eventList.Items))
	for _, e := range eventList.Items {
		firstTime := ""
		if !e.FirstTimestamp.IsZero() {
			firstTime = e.FirstTimestamp.Time.Format("2006-01-02 15:04:05")
		}
		lastTime := ""
		if !e.LastTimestamp.IsZero() {
			lastTime = e.LastTimestamp.Time.Format("2006-01-02 15:04:05")
		}

		events = append(events, EventInfo{
			Namespace:      e.Namespace,
			Type:           e.Type,
			Reason:         e.Reason,
			Message:        e.Message,
			Object:         fmt.Sprintf("%s/%s", e.InvolvedObject.Kind, e.InvolvedObject.Name),
			FirstTimestamp: firstTime,
			LastTimestamp:  lastTime,
			Count:          e.Count,
		})
	}

	return GetEventsResult{Events: events}, nil
}

// =========================================================================
// 6. Get Object YAML Tool
// =========================================================================

type GetObjectYAMLArgs struct {
	Kind      string `json:"kind" description:"The Kind of the object (e.g. Pod, Service, Deployment, ReplicaSet, StatefulSet, DaemonSet, ConfigMap, Secret, Event, Ingress, PVC, Job, CronJob)"`
	Namespace string `json:"namespace" description:"The namespace of the object"`
	Name      string `json:"name" description:"The name of the object"`
}

type GetObjectYAMLResult struct {
	YAML string `json:"yaml"`
}

// GetObjectYAML retrieves the raw configuration YAML of a Kubernetes object.
// This is crucial for checking configurations (such as mismatching port names, volumes, or incorrect images).
func (tb *K8sToolbox) GetObjectYAML(ctx tool.Context, args GetObjectYAMLArgs) (GetObjectYAMLResult, error) {
	var obj any
	var err error

	// Normalize resource kind string.
	kind := strings.ToLower(args.Kind)
	switch kind {
	case "pod", "pods":
		obj, err = tb.Clientset.CoreV1().Pods(args.Namespace).Get(ctx, args.Name, metav1.GetOptions{})
	case "service", "services", "svc":
		obj, err = tb.Clientset.CoreV1().Services(args.Namespace).Get(ctx, args.Name, metav1.GetOptions{})
	case "deployment", "deployments", "deploy":
		obj, err = tb.Clientset.AppsV1().Deployments(args.Namespace).Get(ctx, args.Name, metav1.GetOptions{})
	case "replicaset", "replicasets", "rs":
		obj, err = tb.Clientset.AppsV1().ReplicaSets(args.Namespace).Get(ctx, args.Name, metav1.GetOptions{})
	case "statefulset", "statefulsets", "sts":
		obj, err = tb.Clientset.AppsV1().StatefulSets(args.Namespace).Get(ctx, args.Name, metav1.GetOptions{})
	case "daemonset", "daemonsets", "ds":
		obj, err = tb.Clientset.AppsV1().DaemonSets(args.Namespace).Get(ctx, args.Name, metav1.GetOptions{})
	case "configmap", "configmaps", "cm":
		obj, err = tb.Clientset.CoreV1().ConfigMaps(args.Namespace).Get(ctx, args.Name, metav1.GetOptions{})
	case "secret", "secrets":
		// Secrets must be redacted to prevent sensitive data from leaking into the LLM context.
		s, getErr := tb.Clientset.CoreV1().Secrets(args.Namespace).Get(ctx, args.Name, metav1.GetOptions{})
		if getErr == nil {
			sCopy := s.DeepCopy()
			for k := range sCopy.Data {
				sCopy.Data[k] = []byte("[REDACTED]")
			}
			obj = sCopy
		} else {
			err = getErr
		}
	case "event", "events":
		obj, err = tb.Clientset.CoreV1().Events(args.Namespace).Get(ctx, args.Name, metav1.GetOptions{})
	case "ingress", "ingresses", "ing":
		obj, err = tb.Clientset.NetworkingV1().Ingresses(args.Namespace).Get(ctx, args.Name, metav1.GetOptions{})
	case "persistentvolumeclaim", "persistentvolumeclaims", "pvc":
		obj, err = tb.Clientset.CoreV1().PersistentVolumeClaims(args.Namespace).Get(ctx, args.Name, metav1.GetOptions{})
	case "job", "jobs":
		obj, err = tb.Clientset.BatchV1().Jobs(args.Namespace).Get(ctx, args.Name, metav1.GetOptions{})
	case "cronjob", "cronjobs":
		obj, err = tb.Clientset.BatchV1().CronJobs(args.Namespace).Get(ctx, args.Name, metav1.GetOptions{})
	default:
		return GetObjectYAMLResult{}, fmt.Errorf("unsupported resource kind %q. Supported kinds: Pod, Service, Deployment, ReplicaSet, StatefulSet, DaemonSet, ConfigMap, Secret, Event, Ingress, PVC, Job, CronJob", args.Kind)
	}

	if err != nil {
		return GetObjectYAMLResult{}, fmt.Errorf("failed to get %s %q in namespace %q: %w", args.Kind, args.Name, args.Namespace, err)
	}

	// Convert the Kubernetes API struct to YAML bytes.
	yamlBytes, err := yaml.Marshal(obj)
	if err != nil {
		return GetObjectYAMLResult{}, fmt.Errorf("failed to marshal object to YAML: %w", err)
	}

	return GetObjectYAMLResult{YAML: string(yamlBytes)}, nil
}

// =========================================================================
// Tool Registration Functions
// =========================================================================

// RegisterAllTools constructs and registers the tool methods as ADK-compatible tool definitions.
// The ADK framework uses these registered definitions to present tool metadata
// (JSON schemas) to the LLM agent, and routes actual tool calls from the model back to these functions.
func RegisterAllTools(tb *K8sToolbox) ([]tool.Tool, error) {
	var tools []tool.Tool

	// 1. Register ListNamespaces
	t1, err := functiontool.New(functiontool.Config{
		Name:        "k8s_list_namespaces",
		Description: "Lists all namespaces in the Kubernetes cluster",
	}, tb.ListNamespaces)
	if err != nil {
		return nil, err
	}
	tools = append(tools, t1)

	// 2. Register ListPods
	t2, err := functiontool.New(functiontool.Config{
		Name:        "k8s_list_pods",
		Description: "Lists pods in a namespace, including names, status, IP, container readiness, and restarts.",
	}, tb.ListPods)
	if err != nil {
		return nil, err
	}
	tools = append(tools, t2)

	// 3. Register GetPodLogs
	t3, err := functiontool.New(functiontool.Config{
		Name:        "k8s_get_pod_logs",
		Description: "Retrieves container logs for a pod in a namespace.",
	}, tb.GetPodLogs)
	if err != nil {
		return nil, err
	}
	tools = append(tools, t3)

	// 4. Register DescribePod
	t4, err := functiontool.New(functiontool.Config{
		Name:        "k8s_describe_pod",
		Description: "Gets details about a pod status, container states, phase, node name, conditions, and pod-specific events.",
	}, tb.DescribePod)
	if err != nil {
		return nil, err
	}
	tools = append(tools, t4)

	// 5. Register GetEvents
	t5, err := functiontool.New(functiontool.Config{
		Name:        "k8s_get_events",
		Description: "Retrieves recent event streams from a namespace or the entire cluster, optionally filtered by object name.",
	}, tb.GetEvents)
	if err != nil {
		return nil, err
	}
	tools = append(tools, t5)

	// 6. Register GetObjectYAML
	t6, err := functiontool.New(functiontool.Config{
		Name:        "k8s_get_object_yaml",
		Description: "Retrieves the raw YAML configuration of a Kubernetes object by Kind, Namespace, and Name.",
	}, tb.GetObjectYAML)
	if err != nil {
		return nil, err
	}
	tools = append(tools, t6)

	return tools, nil
}
