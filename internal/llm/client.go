package llm

import (
	"fmt"
	"os"
	"strings"

	"github.com/rs/zerolog"
)

const (
	envProvider = "LLM_PROVIDER" // "anthropic" or "openai"
)

// NewClientFromEnv creates a client based on LLM_PROVIDER env var
// Defaults to Anthropic if not specified
func NewClientFromEnv() (Client, error) {
	provider := strings.ToLower(strings.TrimSpace(os.Getenv(envProvider)))
	if provider == "" {
		provider = "anthropic" // Default
	}

	switch provider {
	case "openai":
		return NewOpenAIFromEnv()
	case "anthropic":
		return NewAnthropicFromEnv()
	default:
		return nil, fmt.Errorf("unknown LLM provider: %s (use 'anthropic' or 'openai')", provider)
	}
}

// NewClientWithLogger creates a client with logger based on LLM_PROVIDER env var
func NewClientWithLogger(logger zerolog.Logger) (Client, error) {
	provider := strings.ToLower(strings.TrimSpace(os.Getenv(envProvider)))
	if provider == "" {
		provider = "anthropic" // Default
	}

	switch provider {
	case "openai":
		return NewOpenAIWithLogger(logger)
	case "anthropic":
		return NewAnthropicWithLogger(logger)
	default:
		return nil, fmt.Errorf("unknown LLM provider: %s (use 'anthropic' or 'openai')", provider)
	}
}
