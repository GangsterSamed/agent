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

// buildSystemPrompt creates a system prompt based on browser-use-reference pattern
func buildSystemPrompt(task string) string {
	return `You are an autonomous browser agent that solves tasks in a real browser. Your ultimate goal is accomplishing the task provided in <user_request>.

<language_settings>
- Always respond in the same language as the user request.
- If the user request is in Russian, respond in Russian (including all messages, summaries, and user prompts).
- If the user request is in English, respond in English.
- When using request_user_input, use the same language as the user request.
</language_settings>

<user_request>
USER REQUEST: This is your ultimate objective and always remains visible.
- This has the highest priority. Make the user happy.
- If the user request is very specific - then carefully follow each step and don't skip or hallucinate steps.
- If the task is open ended you can plan yourself how to get it done.
- The <user_request> is the ultimate goal. If the user specifies explicit steps, they have always the highest priority.
</user_request>

<agent_history>
Agent history will be given as a list of step information as follows:
<step_N>:
Evaluation of Previous Step: Assessment of last action
Memory: Your memory of this step (track progress here - e.g., "Processed item 1/10", "Completed step 2/5")
Next Goal: Your goal for this step
Action Results: Your actions and their results
</step_N>

Use the Memory fields from previous steps to track what you've already done and avoid repeating the same actions.
</agent_history>

<output_format>
You must ALWAYS respond with valid JSON:
{
  "thinking": "Structured reasoning about current state, history, and what to do next",
  "evaluation_previous_goal": "One-sentence analysis of last action: Success/Failure/Uncertain",
  "memory": "1-3 sentences tracking progress (pages visited, items found, steps completed)",
  "next_goal": "Next immediate goal in one clear sentence",
  "action": "action_name",
  "input": {...}
}
</output_format>

<browser_rules>
- Only interact with elements that have an index in the elements list provided
- Use click_by_index with index from elements list when available
- CRITICAL: Always use elements from the CURRENT <browser_state> snapshot, NOT from history. If an element is not in the current snapshot, it doesn't exist anymore - the page has changed. Check the current snapshot before every action.
- CRITICAL: The browser state is automatically updated after each action. You will receive the new page state in the next step. If the page changes after an action, the sequence continues and you get the new state automatically - you do NOT need to use wait or wait_for actions to wait for page changes.
- CRITICAL: After clicking a button or submitting a form, DO NOT use wait action to check if the page changed. The page state is automatically updated in the next step - just proceed to the next action or check the new snapshot that will be provided.
- CRITICAL: After filling a form field (fill_by_index or fill), the page may change (new elements may appear, buttons may change). Always check the CURRENT snapshot in the next step to see what elements are available NOW, not what was available in previous steps.
- If page changes after an action, analyze if new elements appeared that need interaction
- Scroll ONLY if there is content above or below the current viewport
- If expected elements are missing, try scrolling, navigating back, or checking if they appear in the next step's snapshot (the page state is automatically updated after each action)
- Use wait action ONLY if you explicitly need to wait for user action (e.g., user solving captcha) or for a specific timeout. DO NOT use wait to wait for page loading, to check if login was successful, or to verify page changes - the page state is updated automatically after each action.
- If action sequence was interrupted, complete remaining actions in next step
- Don't login into a page if you don't have to. If the task requires login and you don't have credentials, use request_user_input to ask the user for them
- CRITICAL: Before using fill_by_index or fill, you MUST have the data. If you see a textbox field (role="textbox" in elements list) and you don't have the value to fill it with, you MUST use request_user_input FIRST to ask the user for the data. DO NOT attempt to fill a field without data - this will cause a timeout. DO NOT use placeholder values like "your_password_here", "enter_password", "your_email", etc. - these are NOT real data and will be rejected. The sequence is: (1) See textbox field -> (2) Use request_user_input("Please provide [login/email/password/etc]") -> (3) After receiving the value, use fill_by_index with that EXACT value (not a placeholder) -> (4) Click submit/next button
- CRITICAL: Before requesting data from user, ALWAYS check your Memory fields in history. If you already requested and received data (e.g., password, login), DO NOT request it again. Use the data you already received from previous request_user_input actions. Check history to see what data you already have.
- CRITICAL CAPTCHA RULE: If you detect a captcha page (URL contains "captcha" or "showcaptcha", page title contains "робот" or "robot", or you see text like "Я не робот", "I'm not a robot", "Вы не робот?"), you MUST IMMEDIATELY use request_user_input tool to ask the user to solve it. DO NOT click on any captcha elements (checkbox, button, link, or any element on captcha page). DO NOT attempt to solve it automatically. DO NOT use click_role, click_selector, click_by_index, or any click action on captcha pages. The ONLY action allowed on captcha pages is request_user_input. After the user responds with "done" (or "готово", "yes"), check if the page changed (URL changed or new elements appeared) - if yes, continue with your task. The user's response "done" is a confirmation that captcha is solved, NOT data to use in fill tool
- To find an input field: check snapshot for role="textbox" elements, or use collect_texts with selector "input[type='text'], input[type='email'], input[type='password'], textarea, [role='textbox']" to get the field selector
- To read content from a page, use read_page tool with appropriate selector
- To find interactive elements not visible in snapshot, use collect_texts tool
- CRITICAL: If clicking on a link leads to unexpected results (like opening search instead of detail view), use collect_texts to find parent container elements that contain that link, then click on the parent element instead
- If snapshot shows only links but you need to interact with list items, use collect_texts to explore the page structure and find the actual interactive elements
</browser_rules>

<reasoning_rules>
You must reason explicitly at every step:
- Analyze agent history to track progress toward the task. History shows your previous Memory fields - use them to track what you've already done (e.g., "processed item 1/10", "completed step 2/5")
- Analyze history to understand what data you already have. When you see request_user_input in history, the Result field contains the data you received. When you see fill_by_index or fill in history, check if the input text matches data from a previous request_user_input - if it matches, that data was already used. Do not request the same data twice - check history first.
- Analyze the most recent action result and clearly state what you tried to achieve
- Explicitly judge success/failure/uncertainty of the last action. Never assume an action succeeded just because it appears executed. Verify by checking if the page state changed as expected (URL changed, new elements appeared, content changed)
- CRITICAL: If an action timed out or failed, but the URL or page state changed (e.g., URL changed significantly, or new elements appeared that indicate success), this means the user completed the action manually. Update your understanding: the task progressed, continue with the next step. DO NOT retry the failed action or request data that is no longer needed.
- If expected change is missing, mark the last action as failed and plan recovery
- Analyze whether you are stuck (repeating same actions without progress). Check your Memory fields in history - if you see the same Memory repeated, you're stuck. Consider alternative approaches: scrolling, different selectors, or different pages
- Decide what concise, actionable context should be stored in memory to inform future reasoning. CRITICAL: Track progress explicitly (e.g., "Processed item 1/10", "Completed step 2/5", "Found 3 matching results"). This helps you avoid repeating the same actions.
- Always reason about the <user_request>. Make sure to carefully analyze the specific steps and information required. E.g. specific filters, specific form fields, specific information to search. Make sure to always compare the current trajectory with the user request and think carefully if that's how the user requested it.
- When ready to finish, state you are preparing to call finish and communicate completion/results to the user.
- Before taking any action, check if the <user_request> is already fully completed by comparing what was requested with what has been accomplished in your history. If all parts of the <user_request> are done, call finish immediately.
</reasoning_rules>

<action_rules>
- You are allowed to use ONE action per step.
- If the page changes after an action, the sequence continues and you get the new state automatically in the next step.
- The browser state is automatically updated after each action - you will see the new page state in the next step without needing to wait.
- DO NOT use wait action after clicking buttons, submitting forms, or navigating - the page state is automatically updated. Only use wait if you explicitly need to wait for user action (e.g., captcha) or a specific timeout.
</action_rules>
<efficiency_guidelines>
- You can combine actions where it makes sense (e.g., input + click for form submission) - but remember you can only use ONE action per step
- Don't try multiple different paths in one step. Always have one clear goal per step
- Don't chain actions that change browser state multiple times - you need to see if each action succeeded
</efficiency_guidelines>

<task_completion>
Call finish action when:
- Task is fully completed (all requirements met)
- Maximum steps reached
- It's absolutely impossible to continue

You must call the finish action in one of two cases:
- When you have fully completed the USER REQUEST.
- When you reach the final allowed step (max_steps), even if the task is incomplete.
- If it is ABSOLUTELY IMPOSSIBLE to continue.

The finish action is your opportunity to terminate and share your findings with the user.
- Set success=true only if the full USER REQUEST has been completed with no missing components.
- If any part of the request is missing, incomplete, or uncertain, set success=false.
- When calling finish action, you MUST provide a detailed summary in the "input" field with a "message" key. The "message" field is REQUIRED when action is "finish". Describe what was accomplished, what steps were taken, and any important results or outcomes. The message should help the user understand what was done and what they can do next.
- Always compare the current trajectory with the user request and think carefully if that's how the user requested it.

</task_completion>`
}

type Planner interface {
	Next(ctx context.Context, state State) (Decision, error)
}

type State struct {
	Task    string
	Step    int
	History []HistoryItem
	Summary snapshot.Summary
	Tools   []tools.Tool
}

type HistoryItem struct {
	Action                 string `json:"action"`
	Result                 string `json:"result"`
	Selector               string `json:"selector,omitempty"`                 // For click_selector actions
	URL                    string `json:"url,omitempty"`                      // URL context for the action
	EvaluationPreviousGoal string `json:"evaluation_previous_goal,omitempty"` // Analysis of last action
	Memory                 string `json:"memory,omitempty"`                   // Progress tracking
	NextGoal               string `json:"next_goal,omitempty"`                // Next immediate goal
}

type Decision struct {
	ActionName             string
	ActionInput            map[string]any
	Finish                 bool
	Message                string
	Thinking               string // Reasoning about current state
	EvaluationPreviousGoal string // Analysis of last action
	Memory                 string // Progress tracking
	NextGoal               string // Next immediate goal
}

type fastPlanner struct {
	llm llm.Client
}

func NewPlanner(client llm.Client) Planner {
	return &fastPlanner{llm: client}
}

func (p *fastPlanner) Next(ctx context.Context, state State) (Decision, error) {
	// Build dynamic system prompt based on task type
	systemPrompt := buildSystemPrompt(state.Task)

	// Minimal guidance - just page info, let agent figure out the rest
	guidance := fmt.Sprintf("URL: %s | Title: %s | Elements: %d\n", state.Summary.URL, state.Summary.Title, len(state.Summary.Elements))

	if len(state.Summary.Elements) > 0 {
		// Interactive roles that should be shown (like browser-use-reference shows all interactive elements)
		actionableRoles := map[string]bool{
			"button": true, "link": true, "textbox": true, "checkbox": true,
			"radio": true, "combobox": true, "listitem": true, "menuitem": true,
			"tab": true, "option": true, "article": true, "row": true,
			"list": true, "listbox": true, "treeitem": true, "cell": true,
		}

		// Show all interactive elements first (like browser-use-reference)
		// They show ALL interactive elements, not just first 15
		hasTextbox := false
		hasLoginButton := false
		for i := range state.Summary.Elements {
			el := &state.Summary.Elements[i]
			roleLower := strings.ToLower(el.Role)
			if actionableRoles[roleLower] {
				guidance += fmt.Sprintf("[%d]%s:%q\n", el.Index, el.Role, truncateText(el.Text, 50))
				if roleLower == "textbox" {
					hasTextbox = true
				}
				// Check for login button/link (universal pattern)
				textLower := strings.ToLower(el.Text)
				if (roleLower == "button" || roleLower == "link") && (strings.Contains(textLower, "войти") || strings.Contains(textLower, "login") || strings.Contains(textLower, "sign in") || strings.Contains(textLower, "log in")) {
					hasLoginButton = true
				}
			}
		}

		// Universal rule: if you see a login button/link but no login form (textbox), click the button first
		if hasLoginButton && !hasTextbox {
			guidance += "\nIMPORTANT: You see a login button or link on the page, but no login form (textbox fields). You should click the login button/link first to open the login form, then request credentials if needed.\n"
		}

		// If we're on login/auth page but don't see textbox in snapshot, remind agent to use collect_texts
		if !hasTextbox && (strings.Contains(state.Summary.URL, "auth") || strings.Contains(state.Summary.URL, "login") || strings.Contains(strings.ToLower(state.Summary.Title), "authorization") || strings.Contains(strings.ToLower(state.Summary.Title), "log in")) {
			guidance += "\nCRITICAL: You are on a login/authorization page but don't see textbox fields in the snapshot. You MUST use collect_texts with selector \"input[type='text'], input[type='email'], input[type='password'], textarea, [role='textbox']\" to find input fields. After collect_texts returns elements with indices, use request_user_input to ask for data, then use fill_by_index with the index from collect_texts result and the value from request_user_input.\n"
		} else if hasTextbox && (strings.Contains(state.Summary.URL, "auth") || strings.Contains(state.Summary.URL, "login") || strings.Contains(strings.ToLower(state.Summary.Title), "authorization") || strings.Contains(strings.ToLower(state.Summary.Title), "log in")) {
			guidance += "\nCRITICAL: You see textbox fields on a login/authorization page. If you don't have the login/email/password data, you MUST use request_user_input FIRST to ask the user for it, then use fill_by_index with the received value.\n"
		}

		// Then show non-interactive elements (up to 50 more to keep context manageable)
		nonInteractiveCount := 0
		maxNonInteractive := 50
		for i := range state.Summary.Elements {
			if nonInteractiveCount >= maxNonInteractive {
				break
			}
			el := &state.Summary.Elements[i]
			roleLower := strings.ToLower(el.Role)
			if !actionableRoles[roleLower] {
				guidance += fmt.Sprintf("[%d]%s:%q\n", el.Index, el.Role, truncateText(el.Text, 50))
				nonInteractiveCount++
			}
		}
	}

	// Format history like browser-use-reference: <step_N>:\nEvaluation: ...\nMemory: ...\nNext Goal: ...\nAction Results: ...
	historyFormatted := formatHistory(state.History)

	// Format message like browser-use-reference: highlight user_request prominently (like browser-use-reference does)
	msg := fmt.Sprintf(`<user_request>
%s
</user_request>

<agent_state>
Step: %d
</agent_state>

<browser_state>
URL: %s
Title: %s
Elements: %d interactive elements available
%s
</browser_state>

<agent_history>
%s
</agent_history>

OUTPUT FORMAT (strict JSON only, no text outside):
{
  "thinking": "...",
  "evaluation_previous_goal": "...",
  "memory": "...",
  "next_goal": "...",
  "action": "tool_name",
  "input": {}
}

If you need to finish the task, set "action": "finish" and provide "input": {"message": "Your detailed summary here"}.
The "message" field is REQUIRED when action is "finish" - describe what was accomplished, what steps were taken, and any important results.

IMPORTANT: Use ONE action per step. Do NOT use multi_tool_use.parallel. Execute actions sequentially: first fill the field, then click the button in the next step.`,
		state.Task,
		state.Step,
		state.Summary.URL,
		state.Summary.Title,
		len(state.Summary.Elements),
		guidance,
		historyFormatted)
	resp, err := p.llm.Generate(ctx, llm.Request{
		System:      systemPrompt,
		Messages:    []llm.Message{{Role: "user", Content: msg}},
		Tools:       toLLMTools(state.Tools),
		Temperature: 0.0,
		MaxTokens:   2000, // Increased for detailed reasoning (thinking/evaluation/memory)
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
		Thinking               string      `json:"thinking"`
		EvaluationPreviousGoal string      `json:"evaluation_previous_goal"`
		Memory                 string      `json:"memory"`
		NextGoal               string      `json:"next_goal"`
		Action                 string      `json:"action"`
		Input                  interface{} `json:"input"` // Can be map or array for multi_tool_use.parallel
	}
	if err := json.Unmarshal([]byte(jsonStr), &parsed); err != nil {
		return Decision{}, fmt.Errorf("llm json parse: %w", err)
	}

	// Handle multi_tool_use.parallel: extract first action from array
	var actionInput map[string]any
	if parsed.Action == "multi_tool_use.parallel" {
		// Input is an array of actions: [{"name":"fill_by_index","index":9,"text":"..."},{"name":"click_by_index","index":2}]
		if inputArr, ok := parsed.Input.([]interface{}); ok && len(inputArr) > 0 {
			if firstAction, ok := inputArr[0].(map[string]interface{}); ok {
				// Extract action name and input from first action
				if name, ok := firstAction["name"].(string); ok {
					parsed.Action = name
					// Remove "name" from map, rest is input
					actionInput = make(map[string]any)
					for k, v := range firstAction {
						if k != "name" {
							actionInput[k] = v
						}
					}
				}
			}
		}
		if actionInput == nil {
			return Decision{}, fmt.Errorf("multi_tool_use.parallel: failed to extract first action from input array")
		}
	} else {
		// Normal case: input is a map
		if parsed.Input == nil {
			actionInput = make(map[string]any)
		} else if inputMap, ok := parsed.Input.(map[string]any); ok {
			actionInput = inputMap
		} else {
			actionInput = make(map[string]any)
		}
	}

	// Remove "functions." prefix if present (OpenAI sometimes adds this prefix)
	actionName := strings.TrimSpace(parsed.Action)
	if strings.HasPrefix(actionName, "functions.") {
		actionName = strings.TrimPrefix(actionName, "functions.")
	}

	dec := Decision{
		ActionName:             actionName,
		ActionInput:            actionInput,
		Thinking:               strings.TrimSpace(parsed.Thinking),
		EvaluationPreviousGoal: strings.TrimSpace(parsed.EvaluationPreviousGoal),
		Memory:                 strings.TrimSpace(parsed.Memory),
		NextGoal:               strings.TrimSpace(parsed.NextGoal),
	}

	if dec.ActionName == "finish" {
		dec.Finish = true
		if msg, ok := actionInput["message"].(string); ok && strings.TrimSpace(msg) != "" {
			dec.Message = strings.TrimSpace(msg)
		} else if m, ok := actionInput["result"].(string); ok && strings.TrimSpace(m) != "" {
			dec.Message = strings.TrimSpace(m)
		} else if t, ok := actionInput["text"].(string); ok && strings.TrimSpace(t) != "" {
			dec.Message = strings.TrimSpace(t)
		}
		// Validate: if finish=true, message must be provided (like ai-agent-for-browser does)
		if dec.Finish && dec.Message == "" {
			return Decision{}, fmt.Errorf("finish action requires 'message' field in input (got: %v)", actionInput)
		}
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
					jsonStr := text[start : i+1]
					// Remove JSON comments (// ... and /* ... */) before parsing
					jsonStr = removeJSONComments(jsonStr)
					return jsonStr, nil
				}
			}
		}
	}
	return "", fmt.Errorf("json not found")
}

// removeJSONComments removes // and /* */ comments from JSON string
func removeJSONComments(jsonStr string) string {
	var result strings.Builder
	inStr := false
	esc := false
	i := 0
	for i < len(jsonStr) {
		ch := jsonStr[i]
		if esc {
			result.WriteByte(ch)
			esc = false
			i++
			continue
		}
		if ch == '\\' && inStr {
			result.WriteByte(ch)
			esc = true
			i++
			continue
		}
		if ch == '"' {
			inStr = !inStr
			result.WriteByte(ch)
			i++
			continue
		}
		if !inStr {
			// Check for // comment
			if i < len(jsonStr)-1 && jsonStr[i] == '/' && jsonStr[i+1] == '/' {
				// Skip to end of line
				for i < len(jsonStr) && jsonStr[i] != '\n' {
					i++
				}
				continue
			}
			// Check for /* comment
			if i < len(jsonStr)-1 && jsonStr[i] == '/' && jsonStr[i+1] == '*' {
				// Skip to */
				i += 2
				for i < len(jsonStr)-1 {
					if jsonStr[i] == '*' && jsonStr[i+1] == '/' {
						i += 2
						break
					}
					i++
				}
				continue
			}
		}
		result.WriteByte(ch)
		i++
	}
	return result.String()
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

func truncateText(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// formatHistory formats history items like browser-use-reference:
// <step_N>:
// Evaluation of Previous Step: ...
// Memory: ...
// Next Goal: ...
// Action Results: ...
func formatHistory(history []HistoryItem) string {
	if len(history) == 0 {
		return ""
	}

	var parts []string
	for i, item := range history {
		stepNum := i + 1
		var content []string

		if item.EvaluationPreviousGoal != "" {
			content = append(content, "Evaluation of Previous Step: "+item.EvaluationPreviousGoal)
		}
		if item.Memory != "" {
			content = append(content, "Memory: "+item.Memory)
		}
		if item.NextGoal != "" {
			content = append(content, "Next Goal: "+item.NextGoal)
		}

		actionResult := fmt.Sprintf("Action Results: %s -> %s", item.Action, item.Result)
		if item.Selector != "" {
			actionResult += fmt.Sprintf(" (selector: %s)", item.Selector)
		}
		if item.URL != "" {
			actionResult += fmt.Sprintf(" (URL: %s)", item.URL)
		}
		content = append(content, actionResult)

		parts = append(parts, fmt.Sprintf("<step_%d>:\n%s\n</step_%d>", stepNum, strings.Join(content, "\n"), stepNum))
	}
	return strings.Join(parts, "\n\n")
}
