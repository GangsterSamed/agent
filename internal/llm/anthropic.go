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
	envAPIKey    = "ANTHROPIC_API_KEY"
	envModel     = "ANTHROPIC_MODEL"
	defaultModel = "claude-sonnet-4-5-20250929"

	apiURL      = "https://api.anthropic.com/v1/messages"
	apiVersion  = "2023-06-01"
	apiBeta     = "tools-2024-04-04"
	maxTokens   = 900
	timeoutSecs = 60

	maxRetries     = 3
	retryBaseDelay = 500 * time.Millisecond
	maxRequestSize = 200000 // ~200KB limit for safety
)

type Client interface {
	Generate(ctx context.Context, req Request) (Response, error)
	Name() string
}

type Request struct {
	System      string
	Messages    []Message
	Tools       []Tool
	Temperature float32
	MaxTokens   int
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"input_schema"`
}

type Response struct {
	Text string
}

type anthropicClient struct {
	apiKey string
	model  string
	http   *http.Client
	logger zerolog.Logger
}

func NewAnthropicFromEnv() (Client, error) {
	key := strings.TrimSpace(os.Getenv(envAPIKey))
	if key == "" {
		return nil, fmt.Errorf("missing %s", envAPIKey)
	}
	model := strings.TrimSpace(os.Getenv(envModel))
	if model == "" {
		model = defaultModel
	}
	model = strings.Trim(model, "\"'")
	return &anthropicClient{
		apiKey: key,
		model:  model,
		http: &http.Client{
			Timeout: timeoutSecs * time.Second,
		},
		logger: zerolog.Nop(), // Will be set by caller if needed
	}, nil
}

// NewAnthropicWithLogger creates client with logger for detailed tracing
func NewAnthropicWithLogger(logger zerolog.Logger) (Client, error) {
	client, err := NewAnthropicFromEnv()
	if err != nil {
		return nil, err
	}
	if ac, ok := client.(*anthropicClient); ok {
		ac.logger = logger
	}
	return client, nil
}

func (c *anthropicClient) Name() string { return c.model }

func (c *anthropicClient) Generate(ctx context.Context, req Request) (Response, error) {
	// Validate input
	if len(req.Messages) == 0 {
		return Response{}, errors.New("no messages")
	}

	// Validate and sanitize message content
	for i, m := range req.Messages {
		if len(m.Content) > maxRequestSize {
			c.logger.Warn().Int("message_idx", i).Int("size", len(m.Content)).Msg("message too large, truncating")
			req.Messages[i].Content = m.Content[:maxRequestSize] + "... [truncated]"
		}
	}

	// Validate system prompt size
	if len(req.System) > maxRequestSize {
		c.logger.Warn().Int("size", len(req.System)).Msg("system prompt too large, truncating")
		req.System = req.System[:maxRequestSize] + "... [truncated]"
	}

	var lastErr error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			// Exponential backoff
			delay := retryBaseDelay * time.Duration(1<<uint(attempt-1))
			c.logger.Info().
				Int("attempt", attempt).
				Dur("delay", delay).
				Msg("retrying Anthropic API call")
			select {
			case <-ctx.Done():
				return Response{}, ctx.Err()
			case <-time.After(delay):
			}
		}

		payload := anthropicPayload{
			Model:       c.model,
			MaxTokens:   max(req.MaxTokens, maxTokens),
			Temperature: float64(req.Temperature),
		}
		if req.System != "" {
			payload.System = req.System
		}
		for _, m := range req.Messages {
			payload.Messages = append(payload.Messages, anthropicMessage{
				Role:    m.Role,
				Content: []anthropicContent{{Type: "text", Text: m.Content}},
			})
		}
		for _, t := range req.Tools {
			payload.Tools = append(payload.Tools, anthropicTool(t))
		}

		body, err := json.Marshal(payload)
		if err != nil {
			return Response{}, fmt.Errorf("marshal payload: %w", err)
		}

		// Log request details (without sensitive data)
		c.logger.Debug().
			Str("model", c.model).
			Int("messages", len(payload.Messages)).
			Int("tools", len(payload.Tools)).
			Int("payload_size", len(body)).
			Int("max_tokens", payload.MaxTokens).
			Msg("Anthropic API request")

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, apiURL, bytes.NewReader(body))
		if err != nil {
			return Response{}, fmt.Errorf("create request: %w", err)
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("x-api-key", c.apiKey)
		httpReq.Header.Set("anthropic-version", apiVersion)
		if apiBeta != "" {
			httpReq.Header.Set("anthropic-beta", apiBeta)
		}

		resp, err := c.http.Do(httpReq)
		if err != nil {
			lastErr = fmt.Errorf("http request: %w", err)
			// Retry on network errors
			if attempt < maxRetries {
				continue
			}
			return Response{}, lastErr
		}

		data, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			lastErr = fmt.Errorf("read response: %w", err)
			if attempt < maxRetries {
				continue
			}
			return Response{}, lastErr
		}

		// Log response details
		c.logger.Debug().
			Int("status", resp.StatusCode).
			Int("response_size", len(data)).
			Msg("Anthropic API response")

		if resp.StatusCode >= 400 {
			var apiErr anthropicError
			rawError := string(data)
			if err := json.Unmarshal(data, &apiErr); err != nil {
				// If we can't parse error, return raw response
				errorMsg := rawError
				if len(errorMsg) > 500 {
					errorMsg = errorMsg[:500] + "..."
				}
				lastErr = fmt.Errorf("anthropic %d: %s (raw, parse err: %v)", resp.StatusCode, errorMsg, err)
			} else {
				errorMsg := apiErr.Error()
				if errorMsg == "" {
					errorMsg = rawError
					if len(errorMsg) > 500 {
						errorMsg = errorMsg[:500] + "..."
					}
				}
				lastErr = fmt.Errorf("anthropic %d: %s (type: %s)", resp.StatusCode, errorMsg, apiErr.Type)
			}

			// Log error details with raw response for debugging
			c.logger.Error().
				Int("status", resp.StatusCode).
				Str("error_type", apiErr.Type).
				Str("error_msg", apiErr.Message).
				Str("raw_response", rawError).
				Int("attempt", attempt).
				Msg("Anthropic API error")

			// Check if it's API usage limit error - don't retry
			if resp.StatusCode == 400 {
				if apiErr.Type == "invalid_request_error" &&
					strings.Contains(apiErr.Message, "API usage limits") {
					// Don't retry - user has reached their limit
					c.logger.Warn().
						Str("error_type", apiErr.Type).
						Str("error_msg", apiErr.Message).
						Msg("API usage limit reached - skipping retries")
					return Response{}, fmt.Errorf("API usage limit reached: %s", apiErr.Message)
				}
			}

			// Retry on 429 (rate limit) and 5xx errors
			if (resp.StatusCode == 429 || resp.StatusCode >= 500) && attempt < maxRetries {
				continue
			}
			// Don't retry on 4xx errors (except 429)
			return Response{}, lastErr
		}

		var ar anthropicResponse
		if err := json.Unmarshal(data, &ar); err != nil {
			lastErr = fmt.Errorf("parse response: %w", err)
			if attempt < maxRetries {
				continue
			}
			return Response{}, lastErr
		}

		var buf bytes.Buffer
		for _, content := range ar.Content {
			if content.Type == "text" {
				buf.WriteString(content.Text)
			}
		}

		c.logger.Debug().
			Int("response_length", buf.Len()).
			Msg("Anthropic API success")

		return Response{Text: buf.String()}, nil
	}

	return Response{}, fmt.Errorf("max retries exceeded: %w", lastErr)
}

type anthropicPayload struct {
	Model       string             `json:"model"`
	System      string             `json:"system,omitempty"`
	Messages    []anthropicMessage `json:"messages"`
	Tools       []anthropicTool    `json:"tools,omitempty"`
	MaxTokens   int                `json:"max_tokens"`
	Temperature float64            `json:"temperature"`
}

type anthropicMessage struct {
	Role    string             `json:"role"`
	Content []anthropicContent `json:"content"`
}

type anthropicContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type anthropicTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"input_schema"`
}

type anthropicResponse struct {
	Content []anthropicContent `json:"content"`
}

type anthropicError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

func (e anthropicError) Error() string {
	if e.Message != "" {
		return e.Message
	}
	return e.Type
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
