package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/polzovatel/ai-agent-for-browser-fast/internal/llm"
	"github.com/polzovatel/ai-agent-for-browser-fast/internal/snapshot"
)

// SubAgent is a specialized agent for specific task types
type SubAgent interface {
	// CanHandle returns true if this agent can handle the given task
	CanHandle(task string) bool
	// Next returns the next action decision, similar to Planner but with specialized logic
	Next(ctx context.Context, state State) (Decision, error)
	// Name returns the agent's name for logging
	Name() string
}

// EmailAgent specializes in email-related tasks (reading, deleting, searching emails)
type EmailAgent struct {
	llm llm.Client
}

func NewEmailAgent(client llm.Client) SubAgent {
	return &EmailAgent{llm: client}
}

func (e *EmailAgent) Name() string {
	return "EmailAgent"
}

func (e *EmailAgent) CanHandle(task string) bool {
	taskLower := strings.ToLower(task)
	emailKeywords := []string{
		"письм", "email", "mail", "почт", "спам", "inbox",
		"прочитай", "read", "удали", "delete", "удалить",
		"последние", "last", "новые", "new",
	}
	for _, keyword := range emailKeywords {
		if strings.Contains(taskLower, keyword) {
			return true
		}
	}
	return false
}

const emailSystemPrompt = `You are a specialized email agent for browser automation.
Your expertise: reading, searching, and managing emails in web mail clients (Gmail, Yandex Mail, Outlook, etc.).

CRITICAL EMAIL-SPECIFIC RULES:
1. Email lists are often in IFRAMES or shadow DOM - elements may not be immediately visible.
2. Look for email-specific patterns in snapshot.elements:
   - Elements with text containing email subjects, senders, dates
   - Elements with role="listitem", "article", "row", or similar
   - Elements with data attributes like data-subject, data-sender, data-id
   - Elements with selectors containing "mail", "message", "letter", "item"
3. Email clients use virtual scrolling - you may need to scroll multiple times to load emails.
4. After scrolling, wait 1-2 seconds for emails to load, then check snapshot again.
5. To read an email: click on the email element (usually the subject or sender text).
6. To delete emails: look for delete buttons/icons (trash, delete icon, or "удалить" text).
7. For spam: look for "spam", "спам" text or spam-related buttons.
8. Use click_role with name from element.text OR click_selector with element.selector.
9. If no emails visible after 3-4 scrolls, try clicking on inbox/folder links first.
10. Respond with SINGLE JSON: {"action":"...","input":{...}} or {"action":"finish","input":{"message":"..."}}`

func (e *EmailAgent) Next(ctx context.Context, state State) (Decision, error) {
	// Handle empty snapshot on first step - just navigate
	if state.Summary.URL == "" || state.Summary.URL == "about:blank" {
		return Decision{
			ActionName:  "navigate",
			ActionInput: map[string]any{"url": "https://mail.yandex.ru"},
		}, nil
	}

	// Enhanced guidance for email tasks with memory context
	guidance := e.buildEmailGuidance(state)

	// Add memory context if available (passed via state if orchestrator provides it)
	memoryContext := ""
	if len(state.History) > 0 {
		scrollCount := e.countScrolls(state.History)
		if scrollCount > 0 {
			memoryContext += fmt.Sprintf("Already scrolled %d times. ", scrollCount)
		}
		// Check if we've seen emails before
		emailRows := e.findEmailRows(state.Summary.Elements)
		if len(emailRows) > 0 {
			memoryContext += fmt.Sprintf("Found %d email rows in current snapshot. ", len(emailRows))
		}
	}
	if memoryContext != "" {
		guidance = memoryContext + guidance
	}

	// Build payload safely - limit size to avoid API errors
	pageMap := state.Summary.ToMap()
	// Limit elements count in payload to avoid too large requests
	if elems, ok := pageMap["elements"].([]snapshot.Element); ok && len(elems) > 50 {
		pageMap["elements"] = elems[:50] // Only first 50 elements
	}

	// Build tools list safely - only include name and description to avoid schema issues
	toolsSafe := make([]map[string]any, 0, len(state.Tools))
	for _, t := range state.Tools {
		toolsSafe = append(toolsSafe, map[string]any{
			"name":        t.Name,
			"description": t.Description,
		})
	}

	payload := map[string]any{
		"task":     state.Task,
		"step":     state.Step,
		"page":     pageMap,
		"history":  state.History,
		"tools":    toolsSafe,
		"format":   map[string]string{"action": "name", "input": "object"},
		"guidance": guidance,
	}

	raw, err := json.Marshal(payload)
	if err != nil {
		return Decision{}, fmt.Errorf("marshal payload: %w", err)
	}

	// Limit message size (Anthropic has limits)
	msgContent := string(raw)
	if len(msgContent) > 50000 { // Rough limit
		msgContent = msgContent[:50000] + "... [truncated]"
	}

	msg := fmt.Sprintf("EMAIL TASK STATE:\n%s\n\nOUTPUT FORMAT (strict JSON only): {\"action\":\"...\",\"input\":{}}\n", msgContent)

	resp, err := e.llm.Generate(ctx, llm.Request{
		System:      emailSystemPrompt,
		Messages:    []llm.Message{{Role: "user", Content: msg}},
		Tools:       toLLMTools(state.Tools),
		Temperature: 0.0,
		MaxTokens:   500, // Slightly more tokens for email tasks
	})
	if err != nil {
		// Fallback: if LLM fails, try basic actions based on state
		return e.fallbackDecision(state, err)
	}

	dec, err := parseDecision(resp.Text)
	if err != nil {
		return Decision{}, fmt.Errorf("email agent decision parse: %w: raw=%q", err, resp.Text)
	}
	return dec, nil
}

func (e *EmailAgent) buildEmailGuidance(state State) string {
	guidance := fmt.Sprintf("EMAIL CLIENT: URL=%s, Title=%s, Elements=%d. ",
		state.Summary.URL, state.Summary.Title, len(state.Summary.Elements))

	// Use email-specific DOM discovery
	emailRows := e.findEmailRows(state.Summary.Elements)
	if len(emailRows) > 0 {
		guidance += fmt.Sprintf("FOUND %d EMAIL ROWS: ", len(emailRows))
		maxShow := 10
		if len(emailRows) < maxShow {
			maxShow = len(emailRows)
		}
		for i := 0; i < maxShow; i++ {
			el := emailRows[i]
			// Show key info: role, first line of text, selector
			textPreview := strings.Split(el.Text, "\n")[0]
			if len(textPreview) > 50 {
				textPreview = textPreview[:50] + "..."
			}
			guidance += fmt.Sprintf("[%d] role=%s text=%q selector=%s; ",
				i+1, el.Role, textPreview, truncateText(el.Sel, 60))
		}
		guidance += "These are email rows - click on them to read. "
	} else {
		guidance += "NO EMAIL ROWS FOUND YET. "
		// Show some elements for debugging
		if len(state.Summary.Elements) > 0 {
			guidance += "Showing first 5 elements for context: "
			maxShow := 5
			if len(state.Summary.Elements) < maxShow {
				maxShow = len(state.Summary.Elements)
			}
			for i := 0; i < maxShow; i++ {
				el := state.Summary.Elements[i]
				textPreview := strings.Split(el.Text, "\n")[0]
				if len(textPreview) > 30 {
					textPreview = textPreview[:30] + "..."
				}
				guidance += fmt.Sprintf("[%d]%s:%q ", i+1, el.Role, textPreview)
			}
		}
	}

	// Check for common email UI elements
	if e.hasInboxFolder(state.Summary.Elements) {
		guidance += "Found inbox/folder links - make sure you're in the right folder. "
	}

	// Check scroll history
	scrollCount := e.countScrolls(state.History)
	if scrollCount > 0 && scrollCount < 5 {
		guidance += fmt.Sprintf("Scrolled %d times - emails may still be loading. Continue scrolling and wait for content. ", scrollCount)
	} else if scrollCount >= 5 && len(emailRows) == 0 {
		guidance += "Scrolled many times but no emails found. Try clicking on inbox/folder links, wait longer, or check if emails are in iframe. "
	}

	if len(emailRows) > 0 {
		guidance += "Use click_role/click_selector on email rows from snapshot. "
	} else {
		// If no email rows found, suggest waiting for them
		guidance += "If emails not visible, try wait_for_emails first, then scroll (email clients use virtual scrolling - wait 1s after scroll). "
	}

	return guidance
}

// findEmailRows is email-specific DOM discovery for Yandex Mail and other email clients
func (e *EmailAgent) findEmailRows(elements []snapshot.Element) []snapshot.Element {
	var emailRows []snapshot.Element
	for _, el := range elements {
		if e.isEmailRow(el) {
			emailRows = append(emailRows, el)
		}
	}
	// Semantic ranking: sort by relevance (most important first)
	e.rankElements(emailRows)
	return emailRows
}

func (e *EmailAgent) findEmailElements(elements []snapshot.Element) []snapshot.Element {
	// Use improved email-specific discovery
	return e.findEmailRows(elements)
}

// isEmailRow checks if element is an email row (Yandex Mail specific patterns)
func (e *EmailAgent) isEmailRow(el snapshot.Element) bool {
	textLower := strings.ToLower(el.Text)
	roleLower := strings.ToLower(el.Role)
	selLower := strings.ToLower(el.Sel)
	attrLower := strings.ToLower(el.Attr)

	// Yandex Mail specific patterns
	// Pattern 1: data-testid with "message" or "mail" or "item"
	if strings.Contains(selLower, "data-testid") {
		if strings.Contains(selLower, "message") || strings.Contains(selLower, "mail") ||
			strings.Contains(selLower, "item") || strings.Contains(selLower, "letter") {
			// Additional check: should have some text content
			if len(el.Text) > 10 {
				return true
			}
		}
	}

	// Pattern 2: role="row" or "listitem" with email-like content
	if roleLower == "row" || roleLower == "listitem" || roleLower == "article" {
		// Check for email indicators in text or attributes
		if strings.Contains(textLower, "@") ||
			strings.Contains(textLower, "от:") || strings.Contains(textLower, "from:") ||
			strings.Contains(textLower, "тема") || strings.Contains(textLower, "subject") ||
			strings.Contains(attrLower, "data-uid") || // Yandex Mail uses data-uid
			strings.Contains(attrLower, "data-subject") ||
			strings.Contains(attrLower, "data-sender") {
			return true
		}
		// If it's a row with substantial text, likely an email
		if len(el.Text) > 30 {
			return true
		}
	}

	// Pattern 3: aria-label with email patterns
	if strings.Contains(attrLower, "aria-label") {
		if strings.Contains(textLower, "@") ||
			strings.Contains(textLower, "от:") || strings.Contains(textLower, "from:") ||
			strings.Contains(textLower, "письм") || strings.Contains(textLower, "mail") {
			return true
		}
	}

	// Pattern 4: Generic email indicators (fallback)
	return e.looksLikeEmail(el)
}

// rankElements sorts elements by semantic importance (email items first, then actions, then UI)
func (e *EmailAgent) rankElements(elements []snapshot.Element) {
	// Simple ranking: elements with more email indicators are more important
	// This is a lightweight semantic ranking
	for i := range elements {
		score := e.elementScore(elements[i])
		// We'll use a simple insertion sort for small arrays
		if i > 0 {
			for j := i; j > 0 && e.elementScore(elements[j-1]) < score; j-- {
				elements[j], elements[j-1] = elements[j-1], elements[j]
			}
		}
	}
}

// elementScore calculates semantic importance score for an element
func (e *EmailAgent) elementScore(el snapshot.Element) int {
	score := 0
	textLower := strings.ToLower(el.Text)
	roleLower := strings.ToLower(el.Role)
	attrLower := strings.ToLower(el.Attr)

	// High priority: email content indicators
	if strings.Contains(textLower, "@") {
		score += 10 // Email address
	}
	if strings.Contains(textLower, "от:") || strings.Contains(textLower, "from:") {
		score += 8 // Sender
	}
	if strings.Contains(textLower, "тема") || strings.Contains(textLower, "subject") {
		score += 8 // Subject
	}
	if strings.Contains(attrLower, "data-subject") || strings.Contains(attrLower, "data-sender") {
		score += 7 // Email data attributes
	}

	// Medium priority: email list items
	if roleLower == "listitem" || roleLower == "article" || roleLower == "row" {
		score += 5
		if len(textLower) > 20 {
			score += 2 // Longer text = more likely to be email content
		}
	}

	// Action buttons (delete, spam, etc.)
	if strings.Contains(textLower, "удалить") || strings.Contains(textLower, "delete") {
		score += 6
	}
	if strings.Contains(textLower, "спам") || strings.Contains(textLower, "spam") {
		score += 6
	}

	// Lower priority: navigation/folder links
	if strings.Contains(textLower, "входящие") || strings.Contains(textLower, "inbox") {
		score += 3
	}

	return score
}

func (e *EmailAgent) looksLikeEmail(el snapshot.Element) bool {
	textLower := strings.ToLower(el.Text)
	roleLower := strings.ToLower(el.Role)
	selLower := strings.ToLower(el.Sel)
	attrLower := strings.ToLower(el.Attr)

	// Email indicators
	emailIndicators := []string{
		"@", // email address
		"subject", "тема", "от:", "from:", "to:", "кому:",
		"дата", "date", "время", "time",
		"письм", "mail", "message", "letter",
		"spam", "спам", "inbox", "входящие",
	}

	// Check text
	for _, indicator := range emailIndicators {
		if strings.Contains(textLower, indicator) {
			return true
		}
	}

	// Check role (email list items)
	emailRoles := []string{"listitem", "article", "row", "option", "menuitem"}
	for _, role := range emailRoles {
		if roleLower == role && len(textLower) > 10 {
			return true
		}
	}

	// Check selector/attributes
	emailSelectors := []string{"mail", "message", "letter", "item", "row", "listitem"}
	for _, sel := range emailSelectors {
		if strings.Contains(selLower, sel) || strings.Contains(attrLower, sel) {
			return true
		}
	}

	// Check for data attributes
	if strings.Contains(attrLower, "data-subject") ||
		strings.Contains(attrLower, "data-sender") ||
		strings.Contains(attrLower, "data-id") {
		return true
	}

	return false
}

func (e *EmailAgent) hasInboxFolder(elements []snapshot.Element) bool {
	for _, el := range elements {
		textLower := strings.ToLower(el.Text)
		folders := []string{"inbox", "входящие", "sent", "отправленные", "spam", "спам", "folder", "папка"}
		for _, folder := range folders {
			if strings.Contains(textLower, folder) {
				return true
			}
		}
	}
	return false
}

func (e *EmailAgent) countScrolls(history []HistoryItem) int {
	count := 0
	for _, item := range history {
		if item.Action == "scroll_page" {
			count++
		}
	}
	return count
}

func truncateText(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// fallbackDecision provides basic fallback actions when LLM fails
func (e *EmailAgent) fallbackDecision(state State, llmErr error) (Decision, error) {
	// If we're on about:blank, navigate
	if state.Summary.URL == "" || state.Summary.URL == "about:blank" {
		return Decision{
			ActionName:  "navigate",
			ActionInput: map[string]any{"url": "https://mail.yandex.ru"},
		}, nil
	}

	urlLower := strings.ToLower(state.Summary.URL)

	// Check if we're viewing a single email (not in inbox list)
	isViewingEmail := strings.Contains(urlLower, "/message/") || strings.Contains(state.Summary.Title, "Письмо")

	if isViewingEmail {
		// If task mentions spam deletion, try to identify and delete spam
		taskLower := strings.ToLower(state.Task)
		if strings.Contains(taskLower, "спам") || strings.Contains(taskLower, "spam") || strings.Contains(taskLower, "удал") {
			// Look for spam indicators in email content
			emailText := strings.ToLower(state.Summary.Visible)
			spamKeywords := []string{"spam", "спам", "unsubscribe", "отписаться", "newsletter", "рассылка", "promo", "promotion"}
			isSpam := false
			for _, keyword := range spamKeywords {
				if strings.Contains(emailText, keyword) {
					isSpam = true
					break
				}
			}

			// If it looks like spam, look for delete button
			if isSpam {
				for _, el := range state.Summary.Elements {
					textLower := strings.ToLower(el.Text)
					attrLower := strings.ToLower(el.Attr)
					if strings.Contains(textLower, "удалить") ||
						strings.Contains(textLower, "delete") ||
						strings.Contains(textLower, "корзина") ||
						strings.Contains(textLower, "trash") ||
						strings.Contains(attrLower, "delete") {
						if el.Sel != "" {
							// Track deletion in memory if available
							if state.Memory != nil {
								state.Memory.EmailsDeleted++
							}
							return Decision{
								ActionName:  "click_selector",
								ActionInput: map[string]any{"selector": el.Sel},
							}, nil
						}
					}
				}
			}
		}

		// We're viewing an email - need to go back to inbox
		// Look for back button or inbox link
		for _, el := range state.Summary.Elements {
			textLower := strings.ToLower(el.Text)
			attrLower := strings.ToLower(el.Attr)
			// Look for back/inbox buttons
			if strings.Contains(textLower, "входящие") ||
				strings.Contains(textLower, "inbox") ||
				strings.Contains(textLower, "назад") ||
				strings.Contains(textLower, "back") ||
				strings.Contains(attrLower, "inbox") ||
				el.Role == "button" && (strings.Contains(textLower, "←") || strings.Contains(textLower, "<")) {
				if el.Sel != "" {
					return Decision{
						ActionName:  "click_selector",
						ActionInput: map[string]any{"selector": el.Sel},
					}, nil
				}
			}
		}
		// If no back button found, try navigating to inbox
		return Decision{
			ActionName:  "navigate",
			ActionInput: map[string]any{"url": "https://mail.yandex.ru/?uid=2289260139#/tabs/relevant"},
		}, nil
	}

	// If we're on mail page but no emails found, try scrolling
	if strings.Contains(urlLower, "mail") {
		// Check if we've read enough emails (task says "последние 10 писем")
		if state.Memory != nil && state.Memory.EmailsRead >= 10 {
			return Decision{
				ActionName: "finish",
				ActionInput: map[string]any{
					"message": fmt.Sprintf("Прочитано %d писем, удалено %d спам-писем",
						state.Memory.EmailsRead, state.Memory.EmailsDeleted),
				},
				Finish: true,
			}, nil
		}

		emailRows := e.findEmailRows(state.Summary.Elements)
		if len(emailRows) == 0 {
			// Try scrolling to load emails
			scrollCount := e.countScrolls(state.History)
			if scrollCount < 5 {
				return Decision{
					ActionName:  "scroll_page",
					ActionInput: map[string]any{"direction": "down", "distance": 300},
				}, nil
			}
			// Try waiting for emails
			return Decision{
				ActionName:  "wait_for_emails",
				ActionInput: map[string]any{"timeout_ms": 10000},
			}, nil
		}

		// If we have emails, try clicking next unprocessed one
		if len(emailRows) > 0 {
			// Get lists of processed/deleted email URLs and selectors from memory
			processedSelectors := make(map[string]bool)
			if state.Memory != nil {
				for _, sel := range state.Memory.ProcessedSelectors {
					processedSelectors[sel] = true
				}
			}

			// Try to find email that hasn't been processed yet
			// Check both selector and URL context to avoid repeats
			var lastSelector string
			var lastURL string
			if len(state.History) > 0 {
				lastAction := state.History[len(state.History)-1]
				lastSelector = lastAction.Selector
				lastURL = lastAction.URL
			}

			// Try to find email with unprocessed selector in current context
			for _, email := range emailRows {
				if email.Sel != "" {
					// Skip if this selector was already processed in this URL context
					selectorKey := email.Sel + ":" + state.Summary.URL
					if processedSelectors[selectorKey] {
						continue
					}
					// Skip if this selector was just used in same URL context
					if email.Sel == lastSelector && state.Summary.URL == lastURL {
						continue
					}
					// This is a new email to process
					return Decision{
						ActionName:  "click_selector",
						ActionInput: map[string]any{"selector": email.Sel},
					}, nil
				}
			}

			// If all emails have same selector pattern or were processed, try first unprocessed
			// Check if we've processed all visible emails
			allProcessed := true
			for _, email := range emailRows {
				if email.Sel != "" {
					selectorKey := email.Sel + ":" + state.Summary.URL
					if !processedSelectors[selectorKey] && !(email.Sel == lastSelector && state.Summary.URL == lastURL) {
						allProcessed = false
						break
					}
				}
			}

			if allProcessed {
				// All visible emails processed - try scrolling or finish
				if state.Memory != nil && state.Memory.EmailsRead >= 10 {
					return Decision{
						ActionName: "finish",
						ActionInput: map[string]any{
							"message": fmt.Sprintf("Прочитано %d писем, удалено %d спам-писем",
								state.Memory.EmailsRead, state.Memory.EmailsDeleted),
						},
						Finish: true,
					}, nil
				}
				// Try scrolling to load more emails
				return Decision{
					ActionName:  "scroll_page",
					ActionInput: map[string]any{"direction": "down", "distance": 300},
				}, nil
			}

			// Fallback: click first email (will be tracked after opening)
			firstEmail := emailRows[0]
			if firstEmail.Sel != "" {
				return Decision{
					ActionName:  "click_selector",
					ActionInput: map[string]any{"selector": firstEmail.Sel},
				}, nil
			}
		}
	}

	// Last resort: return error with context
	return Decision{}, fmt.Errorf("LLM failed and no fallback available: %w (url: %s, elements: %d)",
		llmErr, state.Summary.URL, len(state.Summary.Elements))
}
