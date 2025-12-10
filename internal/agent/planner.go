package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/polzovatel/ai-agent-for-browser-fast/internal/llm"
	"github.com/polzovatel/ai-agent-for-browser-fast/internal/snapshot"
	"github.com/polzovatel/ai-agent-for-browser-fast/internal/tools"
)

const systemPrompt = `You are a fast, deterministic browser agent.
CRITICAL RULES:
1. Use ONLY the provided tools.
2. Respond with a SINGLE JSON object and NOTHING else: {"action": "...", "input": {...}}
3. BEFORE any action, ALWAYS check snapshot.elements array - it contains visible clickable elements.
4. Prefer click_role with name (from snapshot.elements[].text or aria-label) > click_selector > click_text.
5. Use scroll_page if target elements are not visible in current snapshot. Heavy SPAs (like email clients) may need multiple scrolls to load content.
6. After clicking a folder/link, the page will update - check new snapshot.elements for the content.
7. If task is done: {"action":"finish","input":{"message":"..."}}.
8. For risky actions (payment/delete): ask user via request_user_input.`

type Planner interface {
	Next(ctx context.Context, state State) (Decision, error)
}

type State struct {
	Task    string
	Step    int
	History []HistoryItem
	Summary snapshot.Summary
	Tools   []tools.Tool
	Memory  *TaskMemory // Persistent memory for task context
}

type HistoryItem struct {
	Action   string `json:"action"`
	Result   string `json:"result"`
	Selector string `json:"selector,omitempty"` // For click_selector actions
	URL      string `json:"url,omitempty"`      // URL context for the action
}

type Decision struct {
	ActionName  string
	ActionInput map[string]any
	Finish      bool
	Message     string
}

type fastPlanner struct {
	llm llm.Client
}

func NewPlanner(client llm.Client) Planner {
	return &fastPlanner{llm: client}
}

func (p *fastPlanner) Next(ctx context.Context, state State) (Decision, error) {
	guidance := fmt.Sprintf("SNAPSHOT: URL=%s, Title=%s, Elements=%d. ", state.Summary.URL, state.Summary.Title, len(state.Summary.Elements))
	if len(state.Summary.Elements) > 0 {
		guidance += "EXAMPLE ELEMENTS (first 8): "
		maxExamples := 8
		if len(state.Summary.Elements) < maxExamples {
			maxExamples = len(state.Summary.Elements)
		}
		for i := 0; i < maxExamples; i++ {
			el := state.Summary.Elements[i]
			guidance += fmt.Sprintf("[%d] role=%s text=%q selector=%s; ", i+1, el.Role, el.Text, el.Sel)
		}
		guidance += fmt.Sprintf("Check snapshot.elements for target items. Use click_role with name from element.text/attr OR click_selector with element.selector. If target not found, use scroll_page to load more content (heavy SPAs load dynamically).")
	} else {
		guidance += "No elements found - page may be loading. Use wait_for or navigate."
	}

	payload := map[string]any{
		"task":     state.Task,
		"step":     state.Step,
		"page":     state.Summary.ToMap(),
		"history":  state.History,
		"tools":    state.Tools,
		"format":   map[string]string{"action": "name", "input": "object"},
		"guidance": guidance,
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return Decision{}, err
	}
	msg := fmt.Sprintf("STATE:\n%s\n\nOUTPUT FORMAT (strict JSON only, no text outside): {\"action\":\"...\",\"input\":{}}\n", string(raw))
	resp, err := p.llm.Generate(ctx, llm.Request{
		System:      systemPrompt,
		Messages:    []llm.Message{{Role: "user", Content: msg}},
		Tools:       toLLMTools(state.Tools),
		Temperature: 0.0,
		MaxTokens:   400,
	})
	if err != nil {
		return Decision{}, err
	}
	dec, err := parseDecision(resp.Text)
	if err != nil {
		return Decision{}, fmt.Errorf("%w: raw=%q", err, resp.Text)
	}
	return dec, nil
}

func parseDecision(text string) (Decision, error) {
	jsonStr, err := extractJSON(text)
	if err != nil {
		return Decision{}, err
	}
	var parsed struct {
		Action string         `json:"action"`
		Input  map[string]any `json:"input"`
	}
	if err := json.Unmarshal([]byte(jsonStr), &parsed); err != nil {
		return Decision{}, fmt.Errorf("llm json parse: %w", err)
	}
	if parsed.Input == nil {
		parsed.Input = map[string]any{}
	}
	dec := Decision{
		ActionName:  strings.TrimSpace(parsed.Action),
		ActionInput: parsed.Input,
	}
	if dec.ActionName == "finish" {
		dec.Finish = true
		if msg, ok := parsed.Input["message"].(string); ok {
			dec.Message = msg
		} else if m, ok := parsed.Input["result"].(string); ok {
			dec.Message = m
		} else if t, ok := parsed.Input["text"].(string); ok {
			dec.Message = t
		}
	}
	if dec.Finish && dec.Message == "" {
		dec.Message = fmt.Sprintf("task finished: %v", parsed.Input)
	}
	return dec, nil
}

func extractJSON(text string) (string, error) {
	depth := 0
	start := -1
	inStr := false
	esc := false
	for i := 0; i < len(text); i++ {
		ch := text[i]
		if esc {
			esc = false
			continue
		}
		switch ch {
		case '\\':
			if inStr {
				esc = true
			}
		case '"':
			inStr = !inStr
		case '{':
			if !inStr {
				if depth == 0 {
					start = i
				}
				depth++
			}
		case '}':
			if !inStr && depth > 0 {
				depth--
				if depth == 0 && start != -1 {
					return text[start : i+1], nil
				}
			}
		}
	}
	return "", fmt.Errorf("json not found")
}

func toLLMTools(ts []tools.Tool) []llm.Tool {
	res := make([]llm.Tool, 0, len(ts))
	for _, t := range ts {
		res = append(res, llm.Tool{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.InputSchema,
		})
	}
	return res
}
