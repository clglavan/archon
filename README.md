# Archon Autonomous Agents

A repository hosting custom, autonomous AI agents.

## Agents

### ☸️ Kubernetes Triage Debugger (`archon-k8s`)
An automated warning event monitoring daemon that uses the Agent Development Kit (ADK) Go framework and a local LM Studio host ([magos-k8s-0.6b](https://huggingface.co/clglavan/magos-k8s-0.6b)) to triage Kubernetes warning events in real-time.

#### Quick Start
1. Navigate to the agent subfolder:
   ```bash
   cd archon-k8s
   ```
2. Copy the configuration template:
   ```bash
   cp config.env.example config.env
   ```
3. Open `config.env` and populate your parameters (LM Studio endpoint, polling rate, alert TTL window, webhooks, or bot credentials).
4. Run the daemon:
   ```bash
   go run main.go
   ```
5. Or compile and execute the binary:
   ```bash
   go build -o k8s-debugger
   ./k8s-debugger
   ```
