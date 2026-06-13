package k8stools

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"sigs.k8s.io/yaml"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
)

type K8sToolbox struct {
	Clientset *kubernetes.Clientset
}

func NewK8sToolbox() (*K8sToolbox, error) {
	clientset, err := getKubeClient()
	if err != nil {
		return nil, err
	}
	return &K8sToolbox{Clientset: clientset}, nil
}

func getKubeClient() (*kubernetes.Clientset, error) {
	// Try in-cluster config first (if running inside a pod)
	if config, err := rest.InClusterConfig(); err == nil {
		return kubernetes.NewForConfig(config)
	}

	// Fallback to local kubeconfig
	kubeconfig := os.Getenv("KUBECONFIG")
	if kubeconfig == "" {
		if home := homedir.HomeDir(); home != "" {
			kubeconfig = filepath.Join(home, ".kube", "config")
		}
	}
	if kubeconfig == "" {
		return nil, fmt.Errorf("kubeconfig file path not found")
	}

	config, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
	if err != nil {
		return nil, fmt.Errorf("failed to build kubeconfig: %w", err)
	}
	return kubernetes.NewForConfig(config)
}

// 1. List Namespaces Tool
type ListNamespacesArgs struct{}

type ListNamespacesResult struct {
	Namespaces []string `json:"namespaces"`
}

func (tb *K8sToolbox) ListNamespaces(ctx tool.Context, args ListNamespacesArgs) (ListNamespacesResult, error) {
	nsList, err := tb.Clientset.CoreV1().Namespaces().List(ctx, metav1.ListOptions{})
	if err != nil {
		return ListNamespacesResult{}, fmt.Errorf("failed to list namespaces: %w", err)
	}

	namespaces := make([]string, 0, len(nsList.Items))
	for _, ns := range nsList.Items {
		namespaces = append(namespaces, ns.Name)
	}

	return ListNamespacesResult{Namespaces: namespaces}, nil
}

// 2. List Pods Tool
type ListPodsArgs struct {
	Namespace string `json:"namespace" description:"Namespace to list pods. If empty, lists across all namespaces."`
}

type PodInfo struct {
	Name      string `json:"name"`
	Namespace string `json:"namespace"`
	Status    string `json:"status"`
	IP        string `json:"ip"`
	Restarts  int32  `json:"restarts"`
	Ready     string `json:"ready"`
}

type ListPodsResult struct {
	Pods []PodInfo `json:"pods"`
}

func (tb *K8sToolbox) ListPods(ctx tool.Context, args ListPodsArgs) (ListPodsResult, error) {
	podList, err := tb.Clientset.CoreV1().Pods(args.Namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return ListPodsResult{}, fmt.Errorf("failed to list pods: %w", err)
	}

	pods := make([]PodInfo, 0, len(podList.Items))
	for _, pod := range podList.Items {
		var restarts int32
		var readyContainers int
		var totalContainers = len(pod.Spec.Containers)

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

// 3. Get Pod Logs Tool
type GetPodLogsArgs struct {
	Namespace     string `json:"namespace" description:"Namespace of the pod"`
	PodName       string `json:"podName" description:"Name of the pod"`
	ContainerName string `json:"containerName" description:"Name of the container (optional if pod has only one container)"`
	TailLines     *int64 `json:"tailLines" description:"Number of recent log lines to return (optional)"`
}

type GetPodLogsResult struct {
	Logs string `json:"logs"`
}

func (tb *K8sToolbox) GetPodLogs(ctx tool.Context, args GetPodLogsArgs) (GetPodLogsResult, error) {
	opts := &corev1.PodLogOptions{}
	if args.ContainerName != "" {
		opts.Container = args.ContainerName
	}
	if args.TailLines != nil {
		opts.TailLines = args.TailLines
	} else {
		defaultTail := int64(100)
		opts.TailLines = &defaultTail
	}

	req := tb.Clientset.CoreV1().Pods(args.Namespace).GetLogs(args.PodName, opts)
	stream, err := req.Stream(ctx)
	if err != nil {
		return GetPodLogsResult{}, fmt.Errorf("failed to stream pod logs: %w", err)
	}
	defer stream.Close()

	buf := new(bytes.Buffer)
	_, err = io.Copy(buf, stream)
	if err != nil {
		return GetPodLogsResult{}, fmt.Errorf("failed to read pod logs stream: %w", err)
	}

	return GetPodLogsResult{Logs: buf.String()}, nil
}

// 4. Describe Pod Tool
type DescribePodArgs struct {
	Namespace string `json:"namespace" description:"Namespace of the pod"`
	PodName   string `json:"podName" description:"Name of the pod"`
}

type ContainerState struct {
	Name         string `json:"name"`
	Ready        bool   `json:"ready"`
	RestartCount int32  `json:"restartCount"`
	State        string `json:"state"`
	Reason       string `json:"reason,omitempty"`
	Message      string `json:"message,omitempty"`
}

type DescribePodResult struct {
	Name       string           `json:"name"`
	Namespace  string           `json:"namespace"`
	Phase      string           `json:"phase"`
	NodeName   string           `json:"nodeName"`
	Containers []ContainerState `json:"containers"`
	Conditions []string         `json:"conditions"`
	Events     []string         `json:"events"`
}

func (tb *K8sToolbox) DescribePod(ctx tool.Context, args DescribePodArgs) (DescribePodResult, error) {
	pod, err := tb.Clientset.CoreV1().Pods(args.Namespace).Get(ctx, args.PodName, metav1.GetOptions{})
	if err != nil {
		return DescribePodResult{}, fmt.Errorf("failed to get pod details: %w", err)
	}

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

	conditions := make([]string, 0, len(pod.Status.Conditions))
	for _, c := range pod.Status.Conditions {
		if c.Status == corev1.ConditionTrue {
			conditions = append(conditions, string(c.Type))
		} else {
			conditions = append(conditions, fmt.Sprintf("%s(False:%s)", c.Type, c.Reason))
		}
	}

	// Get events related to this pod
	eventList, err := tb.Clientset.CoreV1().Events(args.Namespace).List(ctx, metav1.ListOptions{
		FieldSelector: fmt.Sprintf("involvedObject.name=%s,involvedObject.kind=Pod", args.PodName),
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

// 5. Get Events Tool
type GetEventsArgs struct {
	Namespace  string `json:"namespace" description:"Namespace to query events. If empty, queries all namespaces."`
	ObjectName string `json:"objectName,omitempty" description:"Optional name of the object (e.g., my-pod) to filter events."`
}

type EventInfo struct {
	Namespace      string `json:"namespace"`
	Type           string `json:"type"`
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

// 6. Get Object YAML Tool
type GetObjectYAMLArgs struct {
	Kind      string `json:"kind" description:"The Kind of the object (e.g. Pod, Service, Deployment, ReplicaSet, StatefulSet, DaemonSet, ConfigMap, Secret, Event, Ingress, PVC)"`
	Namespace string `json:"namespace" description:"The namespace of the object"`
	Name      string `json:"name" description:"The name of the object"`
}

type GetObjectYAMLResult struct {
	YAML string `json:"yaml"`
}

func (tb *K8sToolbox) GetObjectYAML(ctx tool.Context, args GetObjectYAMLArgs) (GetObjectYAMLResult, error) {
	var obj any
	var err error

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
	default:
		return GetObjectYAMLResult{}, fmt.Errorf("unsupported resource kind %q. Supported kinds: Pod, Service, Deployment, ReplicaSet, StatefulSet, DaemonSet, ConfigMap, Secret, Event, Ingress, PVC", args.Kind)
	}

	if err != nil {
		return GetObjectYAMLResult{}, fmt.Errorf("failed to get %s %q in namespace %q: %w", args.Kind, args.Name, args.Namespace, err)
	}

	yamlBytes, err := yaml.Marshal(obj)
	if err != nil {
		return GetObjectYAMLResult{}, fmt.Errorf("failed to marshal object to YAML: %w", err)
	}

	return GetObjectYAMLResult{YAML: string(yamlBytes)}, nil
}

func RegisterAllTools(tb *K8sToolbox) ([]tool.Tool, error) {
	var tools []tool.Tool

	t1, err := functiontool.New(functiontool.Config{
		Name:        "k8s_list_namespaces",
		Description: "Lists all namespaces in the Kubernetes cluster",
	}, tb.ListNamespaces)
	if err != nil {
		return nil, err
	}
	tools = append(tools, t1)

	t2, err := functiontool.New(functiontool.Config{
		Name:        "k8s_list_pods",
		Description: "Lists pods in a namespace, including names, status, IP, container readiness, and restarts.",
	}, tb.ListPods)
	if err != nil {
		return nil, err
	}
	tools = append(tools, t2)

	t3, err := functiontool.New(functiontool.Config{
		Name:        "k8s_get_pod_logs",
		Description: "Retrieves container logs for a pod in a namespace.",
	}, tb.GetPodLogs)
	if err != nil {
		return nil, err
	}
	tools = append(tools, t3)

	t4, err := functiontool.New(functiontool.Config{
		Name:        "k8s_describe_pod",
		Description: "Gets details about a pod status, container states, phase, node name, conditions, and pod-specific events.",
	}, tb.DescribePod)
	if err != nil {
		return nil, err
	}
	tools = append(tools, t4)

	t5, err := functiontool.New(functiontool.Config{
		Name:        "k8s_get_events",
		Description: "Retrieves recent event streams from a namespace or the entire cluster, optionally filtered by object name.",
	}, tb.GetEvents)
	if err != nil {
		return nil, err
	}
	tools = append(tools, t5)

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
