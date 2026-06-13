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
	"strings"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// LMStudioModel implements model.LLM interface using OpenAI-compatible endpoints.
type LMStudioModel struct {
	modelName string
	baseURL   string
}

func NewLMStudioModel(modelName, baseURL string) *LMStudioModel {
	if baseURL == "" {
		baseURL = "http://localhost:1234"
	}
	return &LMStudioModel{
		modelName: modelName,
		baseURL:   baseURL,
	}
}

func (m *LMStudioModel) Name() string {
	return m.modelName
}

type OpenAIChatRequest struct {
	Model    string          `json:"model"`
	Messages []OpenAIMessage `json:"messages"`
	Stream   bool            `json:"stream"`
	Tools    []OpenAITool    `json:"tools,omitempty"`
}

type OpenAIMessage struct {
	Role       string           `json:"role"`
	Content    string           `json:"content"`
	ToolCalls  []OpenAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
	Name       string           `json:"name,omitempty"`
}

type OpenAIToolCall struct {
	ID       string             `json:"id"`
	Type     string             `json:"type"`
	Function OpenAIFunctionCall `json:"function"`
}

type OpenAIFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON string in standard OpenAI
}

type OpenAITool struct {
	Type     string           `json:"type"`
	Function OpenAIFunction   `json:"function"`
}

type OpenAIFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

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

type OpenAIStreamResponse struct {
	ID      string               `json:"id"`
	Model   string               `json:"model"`
	Choices []OpenAIStreamChoice `json:"choices"`
}

type OpenAIStreamChoice struct {
	Index        int         `json:"index"`
	Delta        OpenAIDelta `json:"delta"`
	FinishReason string      `json:"finish_reason,omitempty"`
}

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
	Arguments string `json:"arguments,omitempty"`
}

type AccumulatedToolCall struct {
	ID        string
	Name      string
	Arguments strings.Builder
}

func (m *LMStudioModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		msgs := convertContentToOpenAIMessages(req.Contents)

		// Prepend system instructions if configured
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

		openaiReq := OpenAIChatRequest{
			Model:    m.modelName,
			Messages: msgs,
			Stream:   stream,
		}

		if req.Config != nil && len(req.Config.Tools) > 0 {
			openaiReq.Tools = convertToolsToOpenAI(req.Config.Tools)
		}

		payloadBytes, err := json.Marshal(openaiReq)
		if err != nil {
			yield(nil, fmt.Errorf("failed to marshal openai request: %w", err))
			return
		}

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

			yield(convertOpenAIResponseToLLMResponse(&openaiResp), nil)
			return
		}

		// Handle streaming response
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
				// Some chunk lines might be empty or metadata
				continue
			}

			for _, choice := range chunk.Choices {
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

		// Yield accumulated tool calls at the end of stream
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
			// Yield empty final turn completion chunk
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

func convertContentToOpenAIMessages(contents []*genai.Content) []OpenAIMessage {
	var msgs []OpenAIMessage

	// Trace tool call IDs to align them with tool responses chronologically
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

				// Match by name
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
