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
	// We return a Go iterator function. The caller (ADK) will range over this iterator
	// to receive chunks of the model's response as they arrive.
	return func(yield func(*model.LLMResponse, error) bool) {
		
		// 1. CONVERT HISTORY: Convert the ADK's message history structures into the
		// flat array of OpenAI-compatible role/message structures expected by LM Studio.
		msgs := convertContentToOpenAIMessages(req.Contents)

		// 2. SYSTEM PROMPT: If the agent configuration includes a System Instruction (e.g. prompt rules),
		// extract all its text parts and prepend them as a "system" role message to the history list.
		if req.Config != nil && req.Config.SystemInstruction != nil {
			var systemText []string
			for _, part := range req.Config.SystemInstruction.Parts {
				if part.Text != "" {
					systemText = append(systemText, part.Text)
				}
			}
			// Prepend the system prompt so the model reads it first before processing the conversation.
			if len(systemText) > 0 {
				msgs = append([]OpenAIMessage{{
					Role:    "system",
					Content: strings.Join(systemText, "\n"),
				}}, msgs...)
			}
		}

		// 3. BUILD PAYLOAD: Construct the OpenAI chat completions request body.
		openaiReq := OpenAIChatRequest{
			Model:    m.modelName,
			Messages: msgs,
			Stream:   stream,
		}

		// Configure generation hyperparameters like Temperature, TopP, and MaxTokens.
		if req.Config != nil {
			// If the framework passes an explicit temperature, use it. Otherwise, fall back
			// to the default temperature we resolved from the environment variables (e.g., 0.0).
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
			// Map the agent's Go tool schemas into JSON schema declarations.
			if len(req.Config.Tools) > 0 {
				openaiReq.Tools = convertToolsToOpenAI(req.Config.Tools)
			}
		} else {
			// Fall back to default precise temperature if no config structure is provided.
			openaiReq.Temperature = m.defaultTemperature
		}

		// Encode request parameters to JSON bytes.
		payloadBytes, err := json.Marshal(openaiReq)
		if err != nil {
			yield(nil, fmt.Errorf("failed to marshal openai request: %w", err))
			return
		}

		// 4. DISPATCH HTTP: Create a POST request pointing to the LM Studio Local Server.
		url := fmt.Sprintf("%s/v1/chat/completions", m.baseURL)
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payloadBytes))
		if err != nil {
			yield(nil, fmt.Errorf("failed to create http request: %w", err))
			return
		}
		httpReq.Header.Set("Content-Type", "application/json")

		// Execute the HTTP call.
		resp, err := http.DefaultClient.Do(httpReq)
		if err != nil {
			yield(nil, fmt.Errorf("failed to execute lm-studio request: %w", err))
			return
		}
		defer resp.Body.Close()

		// Validate status code. LM Studio returns 200 OK on successful generation initialization.
		if resp.StatusCode != http.StatusOK {
			bodyBytes, _ := io.ReadAll(resp.Body)
			yield(nil, fmt.Errorf("lm-studio returned status %d: %s", resp.StatusCode, string(bodyBytes)))
			return
		}

		// 5. NON-STREAMING ROUTINE: If streaming mode is disabled, read the entire body,
		// decode the complete assistant response, convert it to ADK format, and return.
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

			// Convert complete result structure to ADK LLMResponse and yield it.
			yield(convertOpenAIResponseToLLMResponse(&openaiResp), nil)
			return
		}

		// 6. STREAMING ROUTINE: Read Server-Sent Events (SSE) stream lines chunk-by-chunk.
		// Since Go buffers reader buffers, we use bufio.Scanner to check lines ending with newlines.
		scanner := bufio.NewScanner(resp.Body)
		
		// accumTools gathers tool calls delta tokens. Since arguments are streamed in
		// incremental text chunks, we collect them in memory until the stream completes.
		accumTools := make(map[int]*AccumulatedToolCall)

		for scanner.Scan() {
			// Periodically check if the execution context was canceled (e.g. due to loop detection limit).
			select {
			case <-ctx.Done():
				yield(nil, ctx.Err())
				return
			default:
			}

			line := scanner.Text()
			// OpenAI streaming streams use lines starting with the SSE spec prefix: "data: "
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")
			data = strings.TrimSpace(data)
			
			// The stream finishes when the server transmits the "[DONE]" sentinel.
			if data == "[DONE]" {
				break
			}

			// Parse the JSON payload inside the data chunk.
			var chunk OpenAIStreamResponse
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				continue
			}

			for _, choice := range chunk.Choices {
				// If this chunk contains text generation, yield it immediately as a partial update.
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

				// If this chunk contains tool call deltas, aggregate the tokens.
				for _, tc := range choice.Delta.ToolCalls {
					at, exists := accumTools[tc.Index]
					if !exists {
						at = &AccumulatedToolCall{}
						accumTools[tc.Index] = at
					}
					// Map call ID if provided (typically sent in the first chunk).
					if tc.ID != "" {
						at.ID = tc.ID
					}
					// Map function name if provided (typically sent in the first chunk).
					if tc.Function.Name != "" {
						at.Name = tc.Function.Name
					}
					// Append the arguments JSON character chunk to the builder buffer.
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

		// 7. YIELD FINAL TOOL CALLS: Once the stream finishes scanning all chunks, check if
		// we collected any tool calls. If so, build their complete arguments maps and yield them.
		if len(accumTools) > 0 {
			var parts []*genai.Part
			for _, at := range accumTools {
				// Parse the aggregated arguments string (which is a JSON string) into a Go map.
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
			// Yield an empty final completion chunk so ADK runner knows the turn finished.
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
	// Since OpenAI expects the tool response role ("tool") to explicitly refer to the
	// tool call's unique ID ("tool_call_id"), we store and track generated IDs per tool name.
	toolCallIDMap := make(map[string][]string)

	for _, c := range contents {
		// Map ADK roles back to OpenAI standard roles.
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

		// Iterate through message parts (a single message can contain text, tool calls, or responses).
		for _, part := range c.Parts {
			if part.Text != "" {
				// Text components of the message.
				textParts = append(textParts, part.Text)
			}
			if part.FunctionCall != nil {
				// The assistant is requesting to call a tool. Generate/extract a unique call ID.
				id := part.FunctionCall.ID
				if id == "" {
					id = fmt.Sprintf("call_%s", part.FunctionCall.Name)
				}
				// Save call ID so we can match it when the corresponding tool response arrives.
				toolCallIDMap[part.FunctionCall.Name] = append(toolCallIDMap[part.FunctionCall.Name], id)
				
				// Structure the tool call payload according to OpenAI schema specifications.
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
				// The tool has executed, yielding its results back.
				isToolResponse = true
				toolName = part.FunctionResponse.Name
				// Serialize the response payload map into JSON string.
				respBytes, _ := json.Marshal(part.FunctionResponse.Response)
				toolResponseStr = string(respBytes)

				// Pull matching call ID for the tool name from our cache queue.
				ids := toolCallIDMap[toolName]
				if len(ids) > 0 {
					matchedID = ids[0]
					toolCallIDMap[toolName] = ids[1:]
				} else {
					matchedID = fmt.Sprintf("call_%s", toolName)
				}
			}
		}

		// Construct the final message depending on content types.
		if isToolResponse {
			// A tool response role maps to the "tool" role in OpenAI API.
			msgs = append(msgs, OpenAIMessage{
				Role:       "tool",
				ToolCallID: matchedID,
				Name:       toolName,
				Content:    toolResponseStr,
			})
		} else if len(toolCalls) > 0 {
			// An assistant role message presenting text answers and tool calls.
			msgs = append(msgs, OpenAIMessage{
				Role:      role,
				Content:   strings.Join(textParts, "\n"),
				ToolCalls: toolCalls,
			})
		} else {
			// Standard conversational chat message.
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
	// Loop over all registered tools passed by the framework.
	for _, gt := range genaiTools {
		// Map each Go function declaration inside the tool list to OpenAI Tool structures.
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
	// Guard against nil schema references.
	if s == nil {
		return nil
	}
	m := make(map[string]any)
	// Map datatype name (Object, String, Integer, Array, etc.)
	if s.Type != "" {
		m["type"] = strings.ToLower(string(s.Type))
	}
	// Map parameter description.
	if s.Description != "" {
		m["description"] = s.Description
	}
	// Map child schema properties if the datatype is an object.
	if s.Properties != nil {
		props := make(map[string]any)
		for k, v := range s.Properties {
			props[k] = convertSchema(v)
		}
		m["properties"] = props
	}
	// Map required keys list.
	if len(s.Required) > 0 {
		m["required"] = s.Required
	}
	// Map array items schema if datatype is an array.
	if s.Items != nil {
		m["items"] = convertSchema(s.Items)
	}
	return m
}

func marshalArgs(args map[string]any) string {
	// Fall back to empty JSON object if arguments map is nil.
	if args == nil {
		return "{}"
	}
	b, _ := json.Marshal(args)
	return string(b)
}

func unmarshalArgs(argsStr string) map[string]any {
	var m map[string]any
	// Parse JSON arguments string back to a Go key-value map.
	if err := json.Unmarshal([]byte(argsStr), &m); err != nil {
		// Return empty map on parsing failures.
		return make(map[string]any)
	}
	return m
}

// convertOpenAIResponseToLLMResponse converts a complete OpenAI response back to ADK LLMResponse structures.
func convertOpenAIResponseToLLMResponse(resp *OpenAIChatResponse) *model.LLMResponse {
	var parts []*genai.Part
	choice := resp.Choices[0]

	// Extract standard text responses if populated.
	if choice.Message.Content != "" {
		parts = append(parts, &genai.Part{
			Text: choice.Message.Content,
		})
	}

	// Extract any tool calls requested by the model.
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

	// Package content items into ADK's standard LLMResponse structure.
	return &model.LLMResponse{
		Content: &genai.Content{
			Role:  "model",
			Parts: parts,
		},
		Partial:      false,
		TurnComplete: true,
	}
}
