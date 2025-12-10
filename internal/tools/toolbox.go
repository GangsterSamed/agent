package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/polzovatel/ai-agent-for-browser-fast/internal/browser"
)

type Toolbox interface {
	Describe() []Tool
	Invoke(ctx context.Context, name string, input map[string]any) (Result, error)
}

type Tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"input_schema"`
}

type Result struct {
	Observation string
}

type PromptFunc func(ctx context.Context, message string) (string, error)

type standard struct {
	ctrl   browser.Controller
	prompt PromptFunc
	tools  []Tool
}

func New(ctrl browser.Controller, prompt PromptFunc) Toolbox {
	return &standard{
		ctrl:   ctrl,
		prompt: prompt,
		tools: []Tool{
			newTool("navigate", "Open URL", schema{"url": str("url to open")}, []string{"url"}),
			newTool("click_text", "Click element by visible text", schema{"text": str("text to click"), "exact": boolean("exact match")}, []string{"text"}),
			newTool("click_role", "Click element by role (button/link/checkbox/radio/option) and name", schema{"role": str("aria role"), "name": str("visible label"), "exact": boolean("exact name match")}, []string{"role"}),
			newTool("click_selector", "Click element by CSS selector (use from snapshot elements)", schema{"selector": str("CSS selector")}, []string{"selector"}),
			newTool("click_text_fuzzy", "Click element by partial text match (fallback when exact match fails)", schema{"text": str("partial text to match")}, []string{"text"}),
			newTool("click_coordinates", "Click at specific coordinates from element bbox (last resort fallback)", schema{"x": integer("x coordinate"), "y": integer("y coordinate")}, []string{"x", "y"}),
			newTool("fill", "Fill input by CSS selector", schema{"selector": str("CSS selector"), "text": str("text to type")}, []string{"selector", "text"}),
			newTool("scroll_page", "Scroll page up/down/top/bottom (use sparingly, max 1-2 times)", schema{"direction": str("down|up|top|bottom|page_down|page_up"), "distance": integer("pixels")}, nil),
			newTool("scroll_to_element", "Scroll element into view before clicking", schema{"selector": str("CSS selector")}, []string{"selector"}),
			newTool("wait_for", "Wait for selector visible", schema{"selector": str("CSS selector"), "timeout_ms": integer("timeout ms")}, []string{"selector"}),
			newTool("wait_for_emails", "Wait for email elements to appear (for email clients)", schema{"timeout_ms": integer("timeout ms")}, nil),
			newTool("request_user_input", "Ask user for extra info (codes, confirm)", schema{"prompt": str("question to user")}, []string{"prompt"}),
			newTool("save_state", "Save current storage state", schema{"path": str("path to save")}, []string{"path"}),
		},
	}
}

func (s *standard) Describe() []Tool {
	return append([]Tool(nil), s.tools...)
}

func (s *standard) Invoke(ctx context.Context, name string, input map[string]any) (Result, error) {
	switch name {
	case "navigate":
		url, err := requiredString(input, "url")
		if err != nil {
			return Result{}, err
		}
		if err := s.ctrl.Navigate(ctx, url); err != nil {
			return Result{}, err
		}
		return Result{Observation: fmt.Sprintf("opened %s", url)}, nil

	case "click_text":
		text, err := requiredString(input, "text")
		if err != nil {
			return Result{}, err
		}
		exact := optionalBool(input, "exact")
		if err := s.ctrl.ClickText(ctx, text, exact); err != nil {
			return Result{}, err
		}
		return Result{Observation: fmt.Sprintf("clicked text %q", text)}, nil

	case "click_role":
		role, err := requiredString(input, "role")
		if err != nil {
			return Result{}, err
		}
		name := optionalString(input, "name")
		exact := optionalBool(input, "exact")
		if err := s.ctrl.ClickRole(ctx, role, name, exact); err != nil {
			return Result{}, err
		}
		return Result{Observation: fmt.Sprintf("clicked role=%s name=%s", role, name)}, nil

	case "click_selector":
		sel, err := requiredString(input, "selector")
		if err != nil {
			return Result{}, err
		}
		// Sanitize selector: remove newlines, fix quotes, validate
		sel = sanitizeSelector(sel)
		if sel == "" {
			return Result{}, fmt.Errorf("selector is invalid or empty after sanitization")
		}
		// Check if element exists before clicking (non-trivial solution)
		// Use WaitFor with short timeout to verify element exists
		if err := s.ctrl.WaitFor(ctx, sel, 2*time.Second); err != nil {
			// Element doesn't exist or not visible - return error
			return Result{}, fmt.Errorf("element not found or not visible: %w", err)
		}
		// Try scrolling to element first
		if err := s.ctrl.ScrollToElement(ctx, sel); err != nil {
			// If scroll fails, try click anyway
		}
		if err := s.ctrl.Click(ctx, sel); err != nil {
			return Result{}, err
		}
		return Result{Observation: fmt.Sprintf("clicked selector %s", sel)}, nil

	case "click_text_fuzzy":
		text, err := requiredString(input, "text")
		if err != nil {
			return Result{}, err
		}
		if err := s.ctrl.ClickByTextFuzzy(ctx, text); err != nil {
			return Result{}, err
		}
		return Result{Observation: fmt.Sprintf("clicked fuzzy text %s", text)}, nil

	case "click_coordinates":
		x, err := requiredInt(input, "x")
		if err != nil {
			return Result{}, err
		}
		y, err := requiredInt(input, "y")
		if err != nil {
			return Result{}, err
		}
		if err := s.ctrl.ClickByCoordinates(ctx, float64(x), float64(y)); err != nil {
			return Result{}, err
		}
		return Result{Observation: fmt.Sprintf("clicked at coordinates (%d, %d)", x, y)}, nil

	case "scroll_to_element":
		sel, err := requiredString(input, "selector")
		if err != nil {
			return Result{}, err
		}
		if err := s.ctrl.ScrollToElement(ctx, sel); err != nil {
			return Result{}, err
		}
		return Result{Observation: fmt.Sprintf("scrolled to element %s", sel)}, nil

	case "wait_for_emails":
		timeout := optionalInt(input, "timeout_ms")
		if timeout <= 0 {
			timeout = 10000
		}
		if err := s.ctrl.WaitForEmailElements(ctx, time.Duration(timeout)*time.Millisecond); err != nil {
			return Result{}, err
		}
		return Result{Observation: "email elements appeared"}, nil

	case "fill":
		sel, err := requiredString(input, "selector")
		if err != nil {
			return Result{}, err
		}
		text, err := requiredString(input, "text")
		if err != nil {
			return Result{}, err
		}
		if err := s.ctrl.Fill(ctx, sel, text); err != nil {
			return Result{}, err
		}
		return Result{Observation: fmt.Sprintf("filled %s", sel)}, nil

	case "scroll_page":
		dir := optionalString(input, "direction")
		dist := optionalInt(input, "distance")
		if err := s.ctrl.Scroll(ctx, dir, dist); err != nil {
			return Result{}, err
		}
		return Result{Observation: fmt.Sprintf("scrolled %s %d", dir, dist)}, nil

	case "wait_for":
		sel, err := requiredString(input, "selector")
		if err != nil {
			return Result{}, err
		}
		timeout := optionalInt(input, "timeout_ms")
		if timeout <= 0 {
			timeout = 5000
		}
		if err := s.ctrl.WaitFor(ctx, sel, time.Duration(timeout)*time.Millisecond); err != nil {
			return Result{}, err
		}
		return Result{Observation: fmt.Sprintf("waited %s", sel)}, nil

	case "request_user_input":
		if s.prompt == nil {
			return Result{}, fmt.Errorf("prompt unavailable")
		}
		msg, err := requiredString(input, "prompt")
		if err != nil {
			return Result{}, err
		}
		answer, err := s.prompt(ctx, msg)
		if err != nil {
			return Result{}, err
		}
		return Result{Observation: answer}, nil

	case "save_state":
		path, err := requiredString(input, "path")
		if err != nil {
			return Result{}, err
		}
		if err := s.ctrl.SaveState(ctx, path); err != nil {
			return Result{}, err
		}
		return Result{Observation: fmt.Sprintf("state saved to %s", path)}, nil
	default:
		return Result{}, fmt.Errorf("unknown tool %s", name)
	}
}

// Helpers for schema and extraction.
type schema map[string]any

func newTool(name, desc string, props schema, required []string) Tool {
	return Tool{
		Name:        name,
		Description: desc,
		InputSchema: map[string]any{
			"type":       "object",
			"properties": props,
			"required":   required,
		},
	}
}

func str(desc string) map[string]any { return map[string]any{"type": "string", "description": desc} }

func boolean(desc string) map[string]any {
	return map[string]any{"type": "boolean", "description": desc}
}

func integer(desc string) map[string]any {
	return map[string]any{"type": "integer", "description": desc}
}

func requiredString(input map[string]any, key string) (string, error) {
	val, ok := input[key]
	if !ok {
		return "", fmt.Errorf("field %s required", key)
	}
	switch v := val.(type) {
	case string:
		if strings.TrimSpace(v) == "" {
			return "", fmt.Errorf("field %s empty", key)
		}
		return v, nil
	case json.Number:
		return v.String(), nil
	default:
		return "", fmt.Errorf("field %s must be string", key)
	}
}

func optionalString(input map[string]any, key string) string {
	val, ok := input[key]
	if !ok {
		return ""
	}
	switch v := val.(type) {
	case string:
		return v
	case json.Number:
		return v.String()
	default:
		return ""
	}
}

func optionalBool(input map[string]any, key string) bool {
	val, ok := input[key]
	if !ok {
		return false
	}
	switch v := val.(type) {
	case bool:
		return v
	case string:
		return strings.EqualFold(v, "true")
	default:
		return false
	}
}

func requiredInt(input map[string]any, key string) (int, error) {
	val, ok := input[key]
	if !ok {
		return 0, fmt.Errorf("field %s required", key)
	}
	switch v := val.(type) {
	case float64:
		return int(v), nil
	case int:
		return v, nil
	case int64:
		return int(v), nil
	case json.Number:
		i, err := v.Int64()
		if err != nil {
			return 0, fmt.Errorf("field %s must be integer: %w", key, err)
		}
		return int(i), nil
	default:
		return 0, fmt.Errorf("field %s must be integer", key)
	}
}

func optionalInt(input map[string]any, key string) int {
	val, ok := input[key]
	if !ok {
		return 0
	}
	switch v := val.(type) {
	case float64:
		return int(v)
	case int:
		return v
	case int64:
		return int(v)
	case json.Number:
		i, _ := v.Int64()
		return int(i)
	default:
		return 0
	}
}

// sanitizeSelector cleans CSS selector from invalid characters
func sanitizeSelector(sel string) string {
	if sel == "" {
		return ""
	}
	// Remove newlines and carriage returns
	sel = strings.ReplaceAll(sel, "\n", " ")
	sel = strings.ReplaceAll(sel, "\r", " ")
	// Remove tabs
	sel = strings.ReplaceAll(sel, "\t", " ")
	// Collapse multiple spaces
	sel = strings.Join(strings.Fields(sel), " ")
	// Remove or escape problematic characters in attribute values
	// For aria-label* and similar, we need to escape quotes properly
	// Simple approach: if selector contains unescaped newlines in quotes, remove that part
	if strings.Contains(sel, "aria-label*=") {
		// Try to fix aria-label with newlines
		parts := strings.Split(sel, "aria-label*=")
		if len(parts) == 2 {
			// Take only first 50 chars of the value to avoid newlines
			valuePart := parts[1]
			if idx := strings.Index(valuePart, "]"); idx > 0 {
				value := valuePart[:idx]
				// Remove newlines and limit length
				value = strings.ReplaceAll(value, "\n", " ")
				value = strings.ReplaceAll(value, "\r", " ")
				if len(value) > 50 {
					value = value[:50]
				}
				// Reconstruct
				sel = parts[0] + "aria-label*=" + value + valuePart[idx:]
			}
		}
	}
	// Final cleanup: remove any remaining problematic sequences
	sel = strings.TrimSpace(sel)
	return sel
}
