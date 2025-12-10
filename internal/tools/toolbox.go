package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/playwright-community/playwright-go"
	"github.com/polzovatel/ai-agent-for-browser-fast/internal/browser"
)

type Toolbox interface {
	Describe() []Tool
	Invoke(ctx context.Context, name string, input map[string]any) (Result, error)
	WaitForStableDOM(ctx context.Context, timeout time.Duration) error
	Page() playwright.Page // For checking element existence
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
			newTool("click_by_index", "Click element by index from snapshot (PREFERRED - use index from elements list, e.g. [1], [2], [3])", schema{"index": integer("element index from snapshot (1-based)")}, []string{"index"}),
			newTool("click_text", "Click element by visible text", schema{"text": str("text to click"), "exact": boolean("exact match")}, []string{"text"}),
			newTool("click_role", "Click element by role (button/link/checkbox/radio/option) and name", schema{"role": str("aria role"), "name": str("visible label"), "exact": boolean("exact name match")}, []string{"role"}),
			newTool("click_selector", "Click element by CSS selector (fallback when index not available)", schema{"selector": str("CSS selector")}, []string{"selector"}),
			newTool("click_text_fuzzy", "Click element by partial text match (fallback when exact match fails)", schema{"text": str("partial text to match")}, []string{"text"}),
			newTool("click_coordinates", "Click at specific coordinates from element bbox (last resort fallback)", schema{"x": integer("x coordinate"), "y": integer("y coordinate")}, []string{"x", "y"}),
			newTool("fill", "Fill input by CSS selector", schema{"selector": str("CSS selector"), "text": str("text to type")}, []string{"selector", "text"}),
			newTool("scroll_page", "Scroll page up/down/top/bottom. Distance is optional - if not provided, uses viewport height (~600-1000px). Use sparingly, max 1-2 times.", schema{"direction": str("down|up|top|bottom|page_down|page_up"), "distance": integer("pixels, optional (defaults to viewport height if not provided)")}, nil),
			newTool("scroll_to_element", "Scroll element into view before clicking", schema{"selector": str("CSS selector")}, []string{"selector"}),
			newTool("wait_for", "Wait for selector visible", schema{"selector": str("CSS selector"), "timeout_ms": integer("timeout ms")}, []string{"selector"}),
			newTool("wait_for_emails", "Wait for email elements to appear (for email clients)", schema{"timeout_ms": integer("timeout ms")}, nil),
			newTool("wait_for_lazy_content", "Wait for lazy-loaded content to appear after scroll (for YouTube, Medium, Twitter)", schema{"selector": str("CSS selector to wait for"), "timeout_ms": integer("timeout ms")}, []string{"selector"}),
			newTool("read_page", "Read text from page or element by selector (use when snapshot doesn't show target elements, especially for iframe content)", schema{"selector": str("CSS selector (empty for full page)"), "max_chars": integer("max characters to return")}, nil),
			newTool("collect_texts", "Collect texts AND selectors from elements by selector (use when snapshot doesn't show target elements, especially for iframe content). Returns both text and selector for each element so you can click them.", schema{"selector": str("CSS selector"), "attribute": str("attribute name instead of text"), "limit": integer("max elements to collect")}, []string{"selector"}),
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
		// Use WaitFor with adequate timeout for SPA and lazy loading (5s instead of 2s)
		if err := s.ctrl.WaitFor(ctx, sel, 5*time.Second); err != nil {
			// Element doesn't exist or not visible - return error
			return Result{}, fmt.Errorf("element not found or not visible: %w", err)
		}
		// Try scrolling to element first
		if err := s.ctrl.ScrollToElement(ctx, sel); err != nil {
			// If scroll fails, try click anyway
		}
		// For Twitter-like sites: hover before click to reveal hidden elements
		// Try hover first, but don't fail if it doesn't work
		_ = s.ctrl.Hover(ctx, sel)
		time.Sleep(200 * time.Millisecond) // Brief pause for hover effects
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

	case "wait_for_lazy_content":
		// Wait for lazy-loaded content (YouTube, Medium, Twitter pattern)
		// After scroll, content may load asynchronously
		selector, err := requiredString(input, "selector")
		if err != nil {
			return Result{}, err
		}
		timeout := optionalInt(input, "timeout_ms")
		if timeout <= 0 {
			timeout = 5000 // Default 5s for lazy loading
		}
		// Wait a bit for lazy loading to trigger, then check
		time.Sleep(500 * time.Millisecond)
		if err := s.ctrl.WaitFor(ctx, selector, time.Duration(timeout)*time.Millisecond); err != nil {
			return Result{}, fmt.Errorf("lazy content not loaded: %w", err)
		}
		return Result{Observation: fmt.Sprintf("lazy content appeared: %s", selector)}, nil

	case "read_page":
		selector := optionalString(input, "selector")
		maxChars := optionalInt(input, "max_chars")
		if maxChars <= 0 {
			maxChars = 5000 // Default from ChatGPT recommendation
		}

		// Read from main frame
		content, err := s.ctrl.Read(ctx, selector)
		if err != nil {
			// Try to continue with frames even if main frame fails
			content = ""
		}

		// Improved: read from all frames (ChatGPT recommendation)
		// This is critical for iframe content like Yandex Mail
		page := s.ctrl.Page()
		frames := page.Frames()
		for _, frame := range frames {
			if frame == page.MainFrame() {
				continue // Already read from main frame
			}
			// Try to get innerText from frame body
			val, err := frame.Evaluate("() => { const b = document.body; return b ? b.innerText : ''; }")
			if err == nil {
				if frameText, ok := val.(string); ok && strings.TrimSpace(frameText) != "" {
					content += "\n\nFRAME:\n" + frameText
				}
			}
		}

		if len(content) > maxChars {
			content = content[:maxChars] + "..."
		}
		return Result{Observation: content}, nil

	case "collect_texts":
		selector, err := requiredString(input, "selector")
		if err != nil {
			return Result{}, err
		}
		attribute := optionalString(input, "attribute")
		limit := optionalInt(input, "limit")
		if limit <= 0 {
			limit = 50
		}
		// Improved: collect from all frames (including iframes) for better iframe support
		// CRITICAL: Return both text AND selector for each element so LLM can click them
		page := s.ctrl.Page()
		frames := page.Frames()

		type itemData struct {
			Text     string `json:"text"`
			Selector string `json:"selector"`
			Index    int    `json:"index"`
		}
		items := make([]itemData, 0, limit)

		// JavaScript function to build selector for nth element (more reliable)
		buildElementSelectorScript := `(selector, index) => {
			try {
				const elements = document.querySelectorAll(selector);
				if (index >= elements.length) return selector;
				const el = elements[index];
				
				// Try to build unique selector
				if (el.id) return "#" + el.id;
				
				// Try data attributes
				const testId = el.getAttribute("data-testid");
				if (testId) {
					const siblings = Array.from(el.parentElement?.children || []);
					const sameTestId = siblings.filter(c => c.getAttribute("data-testid") === testId);
					if (sameTestId.length > 1) {
						const idx = sameTestId.indexOf(el) + 1;
						return "[data-testid=\"" + testId + "\"]:nth-of-type(" + idx + ")";
					}
					return "[data-testid=\"" + testId + "\"]";
				}
				
				// Fallback to nth-of-type
				const role = el.getAttribute("role");
				if (role) {
					const roleSiblings = Array.from(el.parentElement?.children || []).filter(c => c.getAttribute("role") === role);
					if (roleSiblings.length > 1) {
						const idx = roleSiblings.indexOf(el) + 1;
						return "[role=\"" + role + "\"]:nth-of-type(" + idx + ")";
					}
					return "[role=\"" + role + "\"]";
				}
				
				// Last resort: use original selector with nth-of-type
				return selector + ":nth-of-type(" + (index + 1) + ")";
			} catch (e) {
				return selector + ":nth-of-type(" + (index + 1) + ")";
			}
		}`

		// Try main frame first
		loc := page.Locator(selector)
		count, err := loc.Count()
		if err == nil && count > 0 {
			maxCount := int(count)
			if limit > 0 && maxCount > limit {
				maxCount = limit
			}
			for i := 0; i < maxCount && len(items) < limit; i++ {
				item := loc.Nth(i)
				var text string
				if attribute != "" {
					attr, err := item.GetAttribute(attribute)
					if err != nil {
						continue
					}
					text = attr
				} else {
					text, err = item.InnerText()
					if err != nil {
						continue
					}
				}
				if text != "" {
					// Build selector for this specific element using JavaScript (more reliable)
					var selStr string
					elementSelector, err := page.Evaluate(buildElementSelectorScript, selector, i)
					if err == nil {
						if s, ok := elementSelector.(string); ok && s != "" {
							selStr = s
						}
					}
					// Validate selector - must not contain undefined or NaN
					if selStr == "" || strings.Contains(selStr, "undefined") || strings.Contains(selStr, "NaN") {
						// Fallback to simple nth-of-type
						if i == 0 {
							selStr = selector + ":first-of-type"
						} else {
							selStr = fmt.Sprintf("%s:nth-of-type(%d)", selector, i+1)
						}
					}
					// Assign index (1-based, like browser-use)
					itemIndex := len(items) + 1
					items = append(items, itemData{
						Text:     text,
						Selector: selStr,
						Index:    itemIndex,
					})
				}
			}
		}

		// Try all iframes if we haven't reached the limit
		if len(items) < limit {
			for _, frame := range frames {
				if frame == page.MainFrame() {
					continue
				}
				if len(items) >= limit {
					break
				}
				iframeLoc := frame.Locator(selector)
				iframeCount, err := iframeLoc.Count()
				if err != nil || iframeCount == 0 {
					continue
				}
				maxCount := int(iframeCount)
				remaining := limit - len(items)
				if maxCount > remaining {
					maxCount = remaining
				}
				for i := 0; i < maxCount; i++ {
					item := iframeLoc.Nth(i)
					var text string
					if attribute != "" {
						attr, err := item.GetAttribute(attribute)
						if err != nil {
							continue
						}
						text = attr
					} else {
						text, err = item.InnerText()
						if err != nil {
							continue
						}
					}
					if text != "" {
						// Build selector for this specific element in iframe using JavaScript
						elementSelector, err := frame.Evaluate(buildElementSelectorScript, selector, i)
						selStr := ""
						if err == nil {
							if s, ok := elementSelector.(string); ok && s != "" {
								selStr = s
							}
						}
						// Validate selector - must not contain undefined or NaN
						if selStr == "" || strings.Contains(selStr, "undefined") || strings.Contains(selStr, "NaN") {
							// Fallback
							if i == 0 {
								selStr = selector + ":first-of-type"
							} else {
								selStr = fmt.Sprintf("%s:nth-of-type(%d)", selector, i+1)
							}
						}
						// Assign index (1-based, like browser-use)
						itemIndex := len(items) + 1
						items = append(items, itemData{
							Text:     text,
							Selector: selStr,
							Index:    itemIndex,
						})
					}
				}
			}
		}

		// Format response to make it clear to LLM that selectors are available
		var responseBuilder strings.Builder

		if len(items) == 0 {
			responseBuilder.WriteString("❌ No items found with selector. Try different selector like [data-testid*='message'] or [role='option']\n")
			return Result{Observation: responseBuilder.String()}, nil
		}

		responseBuilder.WriteString(fmt.Sprintf("✅ Found %d email items. ", len(items)))
		responseBuilder.WriteString("🚨 CRITICAL: Each item has an 'index' field - USE click_by_index!\n\n")
		responseBuilder.WriteString("❌ DO NOT use click_selector or generate your own selector!\n")
		responseBuilder.WriteString("✅ DO use click_by_index with the 'index' field from items[] below (e.g., click_by_index with index=1)!\n\n")

		// Show first 5 items with their selectors prominently
		maxShow := 5
		if len(items) < maxShow {
			maxShow = len(items)
		}
		for i := 0; i < maxShow; i++ {
			item := items[i]
			textPreview := item.Text
			// Get first line of text for preview
			if newlineIdx := strings.Index(textPreview, "\n"); newlineIdx > 0 {
				textPreview = textPreview[:newlineIdx]
			}
			if len(textPreview) > 60 {
				textPreview = textPreview[:60] + "..."
			}
			// Validate selector before showing
			if item.Selector == "" || strings.Contains(item.Selector, "undefined") || strings.Contains(item.Selector, "NaN") {
				// Fallback to simple nth-of-type
				item.Selector = fmt.Sprintf("%s:nth-of-type(%d)", selector, i+1)
			}
			responseBuilder.WriteString(fmt.Sprintf("[%d] text=\"%s\" → USE click_by_index with index=%d\n", item.Index, textPreview, item.Index))
		}
		if len(items) > maxShow {
			responseBuilder.WriteString(fmt.Sprintf("... and %d more emails (all have selectors in JSON)\n", len(items)-maxShow))
		}

		responseBuilder.WriteString("\n📋 ACTION REQUIRED: Use click_by_index with index from items[0].index to open first email!\n")
		if len(items) > 0 {
			responseBuilder.WriteString(fmt.Sprintf("Example: click_by_index with index=%d\n\n", items[0].Index))
		}

		// Also include full JSON for programmatic access
		payload := map[string]any{
			"items":       items,
			"count":       len(items),
			"instruction": "Use click_by_index with items[].index to click on elements. DO NOT use click_selector!",
		}
		if len(items) > 0 {
			payload["example"] = fmt.Sprintf("click_selector with selector=\"%s\"", items[0].Selector)
		} else {
			payload["example"] = "No items found - try different selector"
		}
		encoded, err := json.Marshal(payload)
		if err != nil {
			return Result{}, err
		}
		responseBuilder.WriteString("Full JSON: " + string(encoded))

		return Result{Observation: responseBuilder.String()}, nil

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
		// If distance is 0 or not provided, Scroll() will use default (viewport height)
		// We'll get the actual distance used from the scroll result
		actualDist, err := s.ctrl.Scroll(ctx, dir, dist)
		if err != nil {
			return Result{}, err
		}
		if actualDist == 0 {
			actualDist = dist // Fallback to original if Scroll didn't return actual distance
			if actualDist == 0 {
				actualDist = 600 // Default fallback
			}
		}
		return Result{Observation: fmt.Sprintf("scrolled %s %d", dir, actualDist)}, nil

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
	// Ensure required is always an array (not nil) for OpenAI compatibility
	requiredArray := required
	if requiredArray == nil {
		requiredArray = []string{}
	}
	return Tool{
		Name:        name,
		Description: desc,
		InputSchema: map[string]any{
			"type":       "object",
			"properties": props,
			"required":   requiredArray,
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
	// CRITICAL FIX: Remove escaped quotes first (\" -> ")
	// LLM generates selectors with escaped quotes which are invalid for CSS
	sel = strings.ReplaceAll(sel, `\"`, `"`)

	// Remove newlines and carriage returns
	sel = strings.ReplaceAll(sel, "\n", " ")
	sel = strings.ReplaceAll(sel, "\r", " ")
	// Remove tabs
	sel = strings.ReplaceAll(sel, "\t", " ")
	// Collapse multiple spaces
	sel = strings.Join(strings.Fields(sel), " ")

	// Fix aria-label values that might be too long or contain newlines
	if strings.Contains(sel, "aria-label*=") {
		parts := strings.Split(sel, "aria-label*=")
		if len(parts) == 2 {
			valuePart := parts[1]
			if idx := strings.Index(valuePart, "]"); idx > 0 {
				value := valuePart[:idx]
				// Remove newlines and limit length to prevent invalid selectors
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

func (s *standard) WaitForStableDOM(ctx context.Context, timeout time.Duration) error {
	return s.ctrl.WaitForStableDOM(ctx, timeout)
}

func (s *standard) Page() playwright.Page {
	return s.ctrl.Page()
}
