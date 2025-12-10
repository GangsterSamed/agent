package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/rs/zerolog"
)

const (
	envOpenAIAPIKey    = "OPENAI_API_KEY"
	envOpenAIModel     = "OPENAI_MODEL"
	defaultOpenAIModel = "gpt-4o-mini"

	openAIAPIURL      = "https://api.openai.com/v1/chat/completions"
	openAIMaxTokens   = 900
	openAITimeoutSecs = 60

	openAIMaxRetries     = 3
	openAIRetryBaseDelay = 500 * time.Millisecond
	openAIMaxRequestSize = 200000 // ~200KB
)

type openAIClient struct {
	apiKey string
	model  string
	http   *http.Client
	logger zerolog.Logger
}

type openAIPayload struct {
	Model       string          `json:"model"`
	Messages    []openAIMessage `json:"messages"`
	Tools       []openAITool    `json:"tools,omitempty"`
	ToolChoice  string          `json:"tool_choice,omitempty"`
	Temperature float64         `json:"temperature"`
	MaxTokens   int             `json:"max_tokens"`
}

type openAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAITool struct {
	Type     string         `json:"type"`
	Function openAIFunction `json:"function"`
}

type openAIFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

type openAIResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role      string `json:"role"`
			Content   string `json:"content"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls,omitempty"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error,omitempty"`
}

func NewOpenAIFromEnv() (Client, error) {
	key := strings.TrimSpace(os.Getenv(envOpenAIAPIKey))
	if key == "" {
		return nil, fmt.Errorf("missing %s", envOpenAIAPIKey)
	}
	model := strings.TrimSpace(os.Getenv(envOpenAIModel))
	if model == "" {
		model = defaultOpenAIModel
	}
	model = strings.Trim(model, "\"'")
	return &openAIClient{
		apiKey: key,
		model:  model,
		http: &http.Client{
			Timeout: openAITimeoutSecs * time.Second,
		},
		logger: zerolog.Nop(),
	}, nil
}

func NewOpenAIWithLogger(logger zerolog.Logger) (Client, error) {
	client, err := NewOpenAIFromEnv()
	if err != nil {
		return nil, err
	}
	if oc, ok := client.(*openAIClient); ok {
		oc.logger = logger
	}
	return client, nil
}

func (c *openAIClient) Name() string {
	return c.model
}

func (c *openAIClient) Generate(ctx context.Context, req Request) (Response, error) {
	// Validate input
	if len(req.Messages) == 0 {
		return Response{}, errors.New("no messages")
	}

	// Validate and sanitize message content
	for i, m := range req.Messages {
		if len(m.Content) > openAIMaxRequestSize {
			c.logger.Warn().Int("message_idx", i).Int("size", len(m.Content)).Msg("message too large, truncating")
			req.Messages[i].Content = m.Content[:openAIMaxRequestSize] + "... [truncated]"
		}
	}

	// Validate system prompt size
	if len(req.System) > openAIMaxRequestSize {
		c.logger.Warn().Int("size", len(req.System)).Msg("system prompt too large, truncating")
		req.System = req.System[:openAIMaxRequestSize] + "... [truncated]"
	}

	var lastErr error
	for attempt := 0; attempt <= openAIMaxRetries; attempt++ {
		if attempt > 0 {
			// Exponential backoff
			delay := openAIRetryBaseDelay * time.Duration(1<<uint(attempt-1))
			c.logger.Info().
				Int("attempt", attempt).
				Dur("delay", delay).
				Msg("retrying OpenAI API call")
			select {
			case <-ctx.Done():
				return Response{}, ctx.Err()
			case <-time.After(delay):
			}
		}

		// Build messages - OpenAI requires system message as first message with role "system"
		messages := make([]openAIMessage, 0, len(req.Messages)+1)
		if req.System != "" {
			messages = append(messages, openAIMessage{
				Role:    "system",
				Content: req.System,
			})
		}
		for _, m := range req.Messages {
			messages = append(messages, openAIMessage{
				Role:    m.Role,
				Content: m.Content,
			})
		}

		// Convert tools to OpenAI format
		tools := make([]openAITool, 0, len(req.Tools))
		for _, t := range req.Tools {
			tools = append(tools, openAITool{
				Type: "function",
				Function: openAIFunction{
					Name:        t.Name,
					Description: t.Description,
					Parameters:  t.InputSchema,
				},
			})
		}

		payload := openAIPayload{
			Model:       c.model,
			Messages:    messages,
			Temperature: float64(req.Temperature),
			MaxTokens:   max(req.MaxTokens, openAIMaxTokens),
		}
		if len(tools) > 0 {
			payload.Tools = tools
			payload.ToolChoice = "auto" // Let model decide when to use tools
		}

		body, err := json.Marshal(payload)
		if err != nil {
			return Response{}, fmt.Errorf("marshal payload: %w", err)
		}

		// Log request details
		c.logger.Debug().
			Str("model", c.model).
			Int("messages", len(messages)).
			Int("tools", len(tools)).
			Int("payload_size", len(body)).
			Int("max_tokens", payload.MaxTokens).
			Msg("OpenAI API request")

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, openAIAPIURL, bytes.NewReader(body))
		if err != nil {
			return Response{}, fmt.Errorf("create request: %w", err)
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)

		resp, err := c.http.Do(httpReq)
		if err != nil {
			lastErr = fmt.Errorf("http request: %w", err)
			if attempt < openAIMaxRetries {
				continue
			}
			return Response{}, lastErr
		}

		data, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			lastErr = fmt.Errorf("read response: %w", err)
			if attempt < openAIMaxRetries {
				continue
			}
			return Response{}, lastErr
		}

		// Log response details
		c.logger.Debug().
			Int("status", resp.StatusCode).
			Int("response_size", len(data)).
			Msg("OpenAI API response")

		if resp.StatusCode >= 400 {
			var apiResp openAIResponse
			rawError := string(data)
			if err := json.Unmarshal(data, &apiResp); err != nil || apiResp.Error == nil {
				// If we can't parse error, return raw response
				errorMsg := rawError
				if len(errorMsg) > 500 {
					errorMsg = errorMsg[:500] + "..."
				}
				lastErr = fmt.Errorf("openai %d: %s (raw, parse err: %v)", resp.StatusCode, errorMsg, err)
			} else {
				errorMsg := apiResp.Error.Message
				if errorMsg == "" {
					errorMsg = rawError
					if len(errorMsg) > 500 {
						errorMsg = errorMsg[:500] + "..."
					}
				}
				lastErr = fmt.Errorf("openai %d: %s (type: %s, code: %s)", resp.StatusCode, errorMsg, apiResp.Error.Type, apiResp.Error.Code)
			}

			c.logger.Error().
				Int("status", resp.StatusCode).
				Str("error_type", apiResp.Error.Type).
				Str("error_msg", apiResp.Error.Message).
				Str("raw_response", rawError).
				Int("attempt", attempt).
				Msg("OpenAI API error")

			// Retry on 429 (rate limit) and 5xx errors
			if (resp.StatusCode == 429 || resp.StatusCode >= 500) && attempt < openAIMaxRetries {
				continue
			}
			// Don't retry on 4xx errors (except 429)
			return Response{}, lastErr
		}

		var apiResp openAIResponse
		if err := json.Unmarshal(data, &apiResp); err != nil {
			return Response{}, fmt.Errorf("parse response: %w (raw: %s)", err, string(data))
		}

		if len(apiResp.Choices) == 0 {
			return Response{}, fmt.Errorf("no choices in response")
		}

		choice := apiResp.Choices[0]

		// Handle tool calls - OpenAI returns tool calls in message, we need to extract them
		if len(choice.Message.ToolCalls) > 0 {
			// If model wants to call a tool, extract the first tool call
			toolCall := choice.Message.ToolCalls[0]
			c.logger.Debug().
				Str("tool_name", toolCall.Function.Name).
				Str("tool_args", truncateString(toolCall.Function.Arguments, 200)).
				Msg("OpenAI tool call")
			// Return tool call as JSON in format: {"action": "tool_name", "input": {...}}
			toolResponse := map[string]interface{}{
				"action": toolCall.Function.Name,
				"input":  map[string]interface{}{},
			}
			// Parse arguments JSON
			if toolCall.Function.Arguments != "" {
				var args map[string]interface{}
				if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err == nil {
					toolResponse["input"] = args
				}
			}
			// Convert to JSON string
			jsonBytes, err := json.Marshal(toolResponse)
			if err != nil {
				return Response{}, fmt.Errorf("marshal tool call: %w", err)
			}
			return Response{Text: string(jsonBytes)}, nil
		}

		// Regular text response
		text := choice.Message.Content
		if text == "" {
			return Response{}, fmt.Errorf("empty response content")
		}

		c.logger.Debug().
			Str("finish_reason", choice.FinishReason).
			Int("prompt_tokens", apiResp.Usage.PromptTokens).
			Int("completion_tokens", apiResp.Usage.CompletionTokens).
			Int("total_tokens", apiResp.Usage.TotalTokens).
			Str("response_preview", truncateString(text, 200)).
			Msg("OpenAI API success")

		return Response{Text: text}, nil
	}

	return Response{}, fmt.Errorf("max retries exceeded: %w", lastErr)
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
