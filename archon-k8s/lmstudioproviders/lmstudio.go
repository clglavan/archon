package lmstudioproviders

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"os"
	"strconv"
	"strings"

	// google.golang.org/adk/model defines the LLM interface of the ADK framework.
	// Implementing this interface lets our custom wrapper act as a standard LLM inside ADK.
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// LMStudioModel implements the model.LLM interface.
// It wraps a local LM Studio server running an OpenAI-compatible REST API.
type LMStudioModel struct {
	modelName          string
	baseURL            string
	defaultTemperature *float32
}

// NewLMStudioModel constructs the model adapter.
// It retrieves the default temperature from the environment variables (loaded from config.env).
// For agent/triage use cases, a low temperature (like 0.0) is desired to ensure deterministic tool selection.
func NewLMStudioModel(modelName, baseURL string) *LMStudioModel {
	if baseURL == "" {
		baseURL = "http://localhost:1234"
	}
	var defaultTemp *float32
	if tempStr := os.Getenv("TEMPERATURE"); tempStr != "" {
		if val, err := strconv.ParseFloat(tempStr, 32); err == nil {
			t := float32(val)
			defaultTemp = &t
		}
	} else {
		// Fallback to 0.0 for precise tool calling.
		t := float32(0.0)
		defaultTemp = &t
	}
	return &LMStudioModel{
		modelName:          modelName,
		baseURL:            baseURL,
		defaultTemperature: defaultTemp,
	}
}

func (m *LMStudioModel) Name() string {
	return m.modelName
}

// =========================================================================
// OpenAI API Compatibility Structs
// =========================================================================

// OpenAIChatRequest matches the standard body parameters for OpenAI's /v1/chat/completions endpoint.
type OpenAIChatRequest struct {
	Model       string          `json:"model"`
	Messages    []OpenAIMessage `json:"messages"`
	Stream      bool            `json:"stream"`
	Tools       []OpenAITool    `json:"tools,omitempty"`
	Temperature *float32        `json:"temperature,omitempty"`
	TopP        *float32        `json:"top_p,omitempty"`
	MaxTokens   *int32          `json:"max_tokens,omitempty"`
}

// OpenAIMessage maps a message role, text content, and potential tool calls/results.
type OpenAIMessage struct {
	Role       string           `json:"role"` // user, assistant, system, or tool
	Content    string           `json:"content"`
	ToolCalls  []OpenAIToolCall `json:"tool_calls,omitempty"`   // Provided by assistant when calling tools
	ToolCallID string           `json:"tool_call_id,omitempty"` // Provided by tool role to associate with a call
	Name       string           `json:"name,omitempty"`         // Name of the tool function
}

type OpenAIToolCall struct {
	ID       string             `json:"id"`
	Type     string             `json:"type"` // always "function"
	Function OpenAIFunctionCall `json:"function"`
}

type OpenAIFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON arguments string
}

type OpenAITool struct {
	Type     string         `json:"type"` // always "function"
	Function OpenAIFunction `json:"function"`
}

type OpenAIFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"` // JSON Schema map
}

// OpenAIChatResponse maps the response of a non-streaming completion call.
type OpenAIChatResponse struct {
	ID      string             `json:"id"`
	Model   string             `json:"model"`
	Choices []OpenAIChatChoice `json:"choices"`
}

type OpenAIChatChoice struct {
	Index        int           `json:"index"`
	Message      OpenAIMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

// OpenAIStreamResponse maps individual SSE data chunks returned during streaming completions.
type OpenAIStreamResponse struct {
	ID      string               `json:"id"`
	Model   string               `json:"model"`
	Choices []OpenAIStreamChoice `json:"choices"`
}

type OpenAIStreamChoice struct {
	Index        int         `json:"index"`
	Delta        OpenAIDelta `json:"delta"` // Delta content updates
	FinishReason string      `json:"finish_reason,omitempty"`
}

// OpenAIDelta stores partial text updates or incremental tool calls in a stream chunk.
type OpenAIDelta struct {
	Role      string                `json:"role,omitempty"`
	Content   string                `json:"content,omitempty"`
	ToolCalls []OpenAIToolCallDelta `json:"tool_calls,omitempty"`
}

type OpenAIToolCallDelta struct {
	Index    int                 `json:"index"`
	ID       string              `json:"id,omitempty"`
	Type     string              `json:"type,omitempty"`
	Function OpenAIFunctionDelta `json:"function"`
}

type OpenAIFunctionDelta struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"` // Incremental JSON string chunk
}

// AccumulatedToolCall accumulates streaming function call chunks in memory.
// Since arguments are streamed token-by-token, we use a strings.Builder to concatenate them.
type AccumulatedToolCall struct {
	ID        string
	Name      string
	Arguments strings.Builder
}

// =========================================================================
// GenerateContent Implementation
// =========================================================================

// GenerateContent compiles ADK LLMRequest fields, sends them to the local LM Studio
// completions endpoint, and yields LLMResponses back to the ADK framework.
func (m *LMStudioModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		// 1. Convert historical content (ADK structures) to OpenAI chat messages.
		msgs := convertContentToOpenAIMessages(req.Contents)

		// Prepend system instructions if configured by the agent (e.g. system instructions defined in main.go).
		if req.Config != nil && req.Config.SystemInstruction != nil {
			var systemText []string
			for _, part := range req.Config.SystemInstruction.Parts {
				if part.Text != "" {
					systemText = append(systemText, part.Text)
				}
			}
			if len(systemText) > 0 {
				msgs = append([]OpenAIMessage{{
					Role:    "system",
					Content: strings.Join(systemText, "\n"),
				}}, msgs...)
			}
		}

		// 2. Prepare the OpenAI Request object.
		openaiReq := OpenAIChatRequest{
			Model:    m.modelName,
			Messages: msgs,
			Stream:   stream,
		}

		// Apply generation configurations (temperature, topP, maxTokens, and tools schema).
		if req.Config != nil {
			if req.Config.Temperature != nil {
				openaiReq.Temperature = req.Config.Temperature
			} else {
				openaiReq.Temperature = m.defaultTemperature
			}
			if req.Config.TopP != nil {
				openaiReq.TopP = req.Config.TopP
			}
			if req.Config.MaxOutputTokens > 0 {
				openaiReq.MaxTokens = &req.Config.MaxOutputTokens
			}
			if len(req.Config.Tools) > 0 {
				openaiReq.Tools = convertToolsToOpenAI(req.Config.Tools)
			}
		} else {
			openaiReq.Temperature = m.defaultTemperature
		}

		payloadBytes, err := json.Marshal(openaiReq)
		if err != nil {
			yield(nil, fmt.Errorf("failed to marshal openai request: %w", err))
			return
		}

		// 3. Dispatch the HTTP POST request to the local server.
		url := fmt.Sprintf("%s/v1/chat/completions", m.baseURL)
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payloadBytes))
		if err != nil {
			yield(nil, fmt.Errorf("failed to create http request: %w", err))
			return
		}
		httpReq.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(httpReq)
		if err != nil {
			yield(nil, fmt.Errorf("failed to execute lm-studio request: %w", err))
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			bodyBytes, _ := io.ReadAll(resp.Body)
			yield(nil, fmt.Errorf("lm-studio returned status %d: %s", resp.StatusCode, string(bodyBytes)))
			return
		}

		// 4. Handle Non-Streaming response mode.
		if !stream {
			bodyBytes, err := io.ReadAll(resp.Body)
			if err != nil {
				yield(nil, fmt.Errorf("failed to read response body: %w", err))
				return
			}
			var openaiResp OpenAIChatResponse
			if err := json.Unmarshal(bodyBytes, &openaiResp); err != nil {
				yield(nil, fmt.Errorf("failed to unmarshal chat response: %w", err))
				return
			}

			if len(openaiResp.Choices) == 0 {
				yield(nil, fmt.Errorf("lm-studio returned empty choices list"))
				return
			}

			// Convert OpenAI response format back to ADK LLMResponse and yield it.
			yield(convertOpenAIResponseToLLMResponse(&openaiResp), nil)
			return
		}

		// 5. Handle Streaming response mode.
		// We read SSE lines chunk by chunk and yield partial text delta responses.
		// If tool call chunks appear, we aggregate them in a map instead of yielding
		// them instantly, as arguments are delivered incrementally.
		scanner := bufio.NewScanner(resp.Body)
		accumTools := make(map[int]*AccumulatedToolCall)

		for scanner.Scan() {
			select {
			case <-ctx.Done():
				yield(nil, ctx.Err())
				return
			default:
			}

			line := scanner.Text()
			// Standard Server-Sent Events streams return data fields prefixing "data: "
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")
			data = strings.TrimSpace(data)
			if data == "[DONE]" {
				break
			}

			var chunk OpenAIStreamResponse
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				continue
			}

			for _, choice := range chunk.Choices {
				// Yield text content updates immediately.
				if choice.Delta.Content != "" {
					yield(&model.LLMResponse{
						Content: &genai.Content{
							Role: "model",
							Parts: []*genai.Part{{
								Text: choice.Delta.Content,
							}},
						},
						Partial:      true,
						TurnComplete: false,
					}, nil)
				}

				// Accumulate streaming tool call tokens.
				for _, tc := range choice.Delta.ToolCalls {
					at, exists := accumTools[tc.Index]
					if !exists {
						at = &AccumulatedToolCall{}
						accumTools[tc.Index] = at
					}
					if tc.ID != "" {
						at.ID = tc.ID
					}
					if tc.Function.Name != "" {
						at.Name = tc.Function.Name
					}
					if tc.Function.Arguments != "" {
						at.Arguments.WriteString(tc.Function.Arguments)
					}
				}
			}
		}

		if err := scanner.Err(); err != nil {
			yield(nil, fmt.Errorf("error reading stream choices: %w", err))
			return
		}

		// 6. Yield final aggregated tool call structures to ADK framework if any were collected.
		if len(accumTools) > 0 {
			var parts []*genai.Part
			for _, at := range accumTools {
				argsMap := unmarshalArgs(at.Arguments.String())
				parts = append(parts, &genai.Part{
					FunctionCall: &genai.FunctionCall{
						ID:   at.ID,
						Name: at.Name,
						Args: argsMap,
					},
				})
			}
			yield(&model.LLMResponse{
				Content: &genai.Content{
					Role:  "model",
					Parts: parts,
				},
				Partial:      false,
				TurnComplete: true,
			}, nil)
		} else {
			// Yield empty turn complete chunk indicating the response stream ended cleanly.
			yield(&model.LLMResponse{
				Content: &genai.Content{
					Role: "model",
				},
				Partial:      false,
				TurnComplete: true,
			}, nil)
		}
	}
}

// =========================================================================
// Conversions and Helper Functions
// =========================================================================

// convertContentToOpenAIMessages maps ADK conversation arrays to OpenAI message templates.
// It reconstructs the sequence of users' queries, models' thought responses, tool invocations,
// and tool response values back to the expected API role arrays.
func convertContentToOpenAIMessages(contents []*genai.Content) []OpenAIMessage {
	var msgs []OpenAIMessage

	// Map to reconcile tool responses with their originating tool call IDs.
	toolCallIDMap := make(map[string][]string)

	for _, c := range contents {
		role := strings.ToLower(c.Role)
		if role == "model" {
			role = "assistant"
		} else if role == "" {
			role = "user"
		}

		var textParts []string
		var toolCalls []OpenAIToolCall
		var isToolResponse bool
		var toolName string
		var toolResponseStr string
		var matchedID string

		for _, part := range c.Parts {
			if part.Text != "" {
				textParts = append(textParts, part.Text)
			}
			if part.FunctionCall != nil {
				id := part.FunctionCall.ID
				if id == "" {
					id = fmt.Sprintf("call_%s", part.FunctionCall.Name)
				}
				toolCallIDMap[part.FunctionCall.Name] = append(toolCallIDMap[part.FunctionCall.Name], id)
				toolCalls = append(toolCalls, OpenAIToolCall{
					ID:   id,
					Type: "function",
					Function: OpenAIFunctionCall{
						Name:      part.FunctionCall.Name,
						Arguments: marshalArgs(part.FunctionCall.Args),
					},
				})
			}
			if part.FunctionResponse != nil {
				isToolResponse = true
				toolName = part.FunctionResponse.Name
				respBytes, _ := json.Marshal(part.FunctionResponse.Response)
				toolResponseStr = string(respBytes)

				// Pull matching call ID for the tool.
				ids := toolCallIDMap[toolName]
				if len(ids) > 0 {
					matchedID = ids[0]
					toolCallIDMap[toolName] = ids[1:]
				} else {
					matchedID = fmt.Sprintf("call_%s", toolName)
				}
			}
		}

		if isToolResponse {
			msgs = append(msgs, OpenAIMessage{
				Role:       "tool",
				ToolCallID: matchedID,
				Name:       toolName,
				Content:    toolResponseStr,
			})
		} else if len(toolCalls) > 0 {
			msgs = append(msgs, OpenAIMessage{
				Role:      role,
				Content:   strings.Join(textParts, "\n"),
				ToolCalls: toolCalls,
			})
		} else {
			msgs = append(msgs, OpenAIMessage{
				Role:    role,
				Content: strings.Join(textParts, "\n"),
			})
		}
	}
	return msgs
}

// convertToolsToOpenAI converts ADK Tool definitions to standard OpenAI parameter format.
func convertToolsToOpenAI(genaiTools []*genai.Tool) []OpenAITool {
	var tools []OpenAITool
	for _, gt := range genaiTools {
		for _, fd := range gt.FunctionDeclarations {
			tools = append(tools, OpenAITool{
				Type: "function",
				Function: OpenAIFunction{
					Name:        fd.Name,
					Description: fd.Description,
					Parameters:  convertSchema(fd.Parameters),
				},
			})
		}
	}
	return tools
}

// convertSchema maps ADK/GenAI Schema to generic map JSON schema representations.
func convertSchema(s *genai.Schema) map[string]any {
	if s == nil {
		return nil
	}
	m := make(map[string]any)
	if s.Type != "" {
		m["type"] = strings.ToLower(string(s.Type))
	}
	if s.Description != "" {
		m["description"] = s.Description
	}
	if s.Properties != nil {
		props := make(map[string]any)
		for k, v := range s.Properties {
			props[k] = convertSchema(v)
		}
		m["properties"] = props
	}
	if len(s.Required) > 0 {
		m["required"] = s.Required
	}
	if s.Items != nil {
		m["items"] = convertSchema(s.Items)
	}
	return m
}

func marshalArgs(args map[string]any) string {
	if args == nil {
		return "{}"
	}
	b, _ := json.Marshal(args)
	return string(b)
}

func unmarshalArgs(argsStr string) map[string]any {
	var m map[string]any
	if err := json.Unmarshal([]byte(argsStr), &m); err != nil {
		return make(map[string]any)
	}
	return m
}

// convertOpenAIResponseToLLMResponse converts a complete OpenAI response back to ADK LLMResponse structures.
func convertOpenAIResponseToLLMResponse(resp *OpenAIChatResponse) *model.LLMResponse {
	var parts []*genai.Part
	choice := resp.Choices[0]

	if choice.Message.Content != "" {
		parts = append(parts, &genai.Part{
			Text: choice.Message.Content,
		})
	}

	for _, tc := range choice.Message.ToolCalls {
		argsMap := unmarshalArgs(tc.Function.Arguments)
		parts = append(parts, &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   tc.ID,
				Name: tc.Function.Name,
				Args: argsMap,
			},
		})
	}

	return &model.LLMResponse{
		Content: &genai.Content{
			Role:  "model",
			Parts: parts,
		},
		Partial:      false,
		TurnComplete: true,
	}
}
