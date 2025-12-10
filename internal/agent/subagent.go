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

CHUNKING STRATEGY (CRITICAL):
- Snapshot shows first 50 most relevant elements (fast, token-efficient).
- ALWAYS check snapshot.elements FIRST.
- If email list is NOT in snapshot (especially in iframes), use collect_texts("[data-testid*='message']") or read_page() to explore DOM.
- This is critical for Yandex Mail which loads emails in iframes.
- CRITICAL: Elements in snapshot have INDICES [1], [2], [3]... Use click_by_index with the index number (e.g., click_by_index with index=3).
  This is the PREFERRED method - more reliable than selectors!
- If emails are not in snapshot (iframe), use collect_texts which returns indices. Then use click_by_index with the index from collect_texts result.

CRITICAL EMAIL-SPECIFIC RULES:
1. Email lists are often in IFRAMES or shadow DOM - elements may not be immediately visible in snapshot.
2. Look for email-specific patterns in snapshot.elements:
   - Elements with text containing email subjects, senders, dates
   - Elements with role="listitem", "article", "row", or similar
   - Elements with data attributes like data-subject, data-sender, data-id
   - Elements with selectors containing "mail", "message", "letter", "item"
3. If snapshot doesn't show emails, use collect_texts("[data-testid*='message']") or read_page() to explore iframe content.
4. Email clients use virtual scrolling - you may need to scroll multiple times to load emails.
5. After scrolling, wait 1-2 seconds for emails to load, then check snapshot again.
6. To read an email: click on the email element (usually the subject or sender text).
7. To delete emails: look for delete buttons/icons (trash, delete icon, or "удалить" text).
8. For spam: look for "spam", "спам" text or spam-related buttons.
9. Use click_by_index with element.index (PREFERRED) OR click_role with name from element.text OR click_selector with element.selector (fallback).
10. If no emails visible after 3-4 scrolls, try clicking on inbox/folder links first.
11. TASK COMPLETION: When task is done (e.g., read 10 emails and deleted spam), use {"action":"finish","input":{"message":"Read 10 emails, deleted X spam emails"}}.
12. DO NOT navigate to Spam folder if task is about reading emails from Inbox - stay in Inbox!
13. DO NOT click on folder that is already active (e.g., don't click "Спам" if you're already in Spam folder).
14. DO NOT repeat the same click_by_index action more than 2 times - if it doesn't work, try a different action!
15. Respond with SINGLE JSON: {"action":"...","input":{...}} or {"action":"finish","input":{"message":"..."}}`

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
	if elems, ok := pageMap["elements"].([]snapshot.Element); ok {
		if len(elems) > 50 {
			pageMap["elements"] = elems[:50] // Only first 50 elements
		}
		// Add available indices info to help LLM
		availableIndices := make([]int, 0, len(elems))
		for _, el := range elems {
			availableIndices = append(availableIndices, el.Index)
		}
		if len(availableIndices) > 0 {
			pageMap["available_indices"] = availableIndices
			pageMap["indices_range"] = fmt.Sprintf("%d-%d", availableIndices[0], availableIndices[len(availableIndices)-1])
		}
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

	// Log LLM response for debugging
	preview := resp.Text
	if len(preview) > 200 {
		preview = preview[:200] + "..."
	}
	fmt.Printf("[EmailAgent] LLM response preview: %s\n", preview)

	dec, err := parseDecision(resp.Text)
	if err != nil {
		return Decision{}, fmt.Errorf("email agent decision parse: %w: raw=%q", err, resp.Text)
	}

	fmt.Printf("[EmailAgent] Parsed decision: action=%s\n", dec.ActionName)

	return dec, nil
}

func (e *EmailAgent) buildEmailGuidance(state State) string {
	// Show available indices range
	availableIndices := make([]int, 0, len(state.Summary.Elements))
	for _, el := range state.Summary.Elements {
		availableIndices = append(availableIndices, el.Index)
	}
	indicesRange := ""
	if len(availableIndices) > 0 {
		indicesRange = fmt.Sprintf("Available indices: %d-%d. ", availableIndices[0], availableIndices[len(availableIndices)-1])
	}

	guidance := fmt.Sprintf("EMAIL CLIENT (snapshot: first 50 elements): URL=%s, Title=%s, Elements=%d. %s",
		state.Summary.URL, state.Summary.Title, len(state.Summary.Elements), indicesRange)

	// Detect current folder
	isInSpamFolder := strings.Contains(state.Summary.URL, "#spam") || strings.Contains(state.Summary.Title, "Спам")
	isInInbox := strings.Contains(state.Summary.URL, "#/tabs/relevant") || strings.Contains(state.Summary.URL, "#inbox") || strings.Contains(state.Summary.Title, "Входящие")

	// Count emails processed from history
	emailsRead := 0
	emailsDeleted := 0
	for _, item := range state.History {
		if strings.Contains(item.URL, "/message/") {
			emailsRead++
		}
		if item.Action == "click_selector" && strings.Contains(item.Result, "clicked selector #delete") {
			emailsDeleted++
		}
	}

	// Task completion logic
	if strings.Contains(strings.ToLower(state.Task), "последние 10 писем") || strings.Contains(strings.ToLower(state.Task), "last 10 emails") {
		if emailsRead >= 10 && emailsDeleted > 0 {
			guidance += fmt.Sprintf("✅ TASK COMPLETE: Read %d emails, deleted %d spam emails. Use finish action with message summarizing what was done! ", emailsRead, emailsDeleted)
		} else if emailsRead >= 10 {
			guidance += fmt.Sprintf("✅ TASK COMPLETE: Read %d emails (no spam found). Use finish action! ", emailsRead)
		} else {
			guidance += fmt.Sprintf("Progress: Read %d/10 emails, deleted %d spam. Continue reading emails from INBOX (not Spam folder)! ", emailsRead, emailsDeleted)
		}
	}

	// Prevent clicking on already active folder
	if isInSpamFolder {
		guidance += "🚨 CRITICAL: You are ALREADY in Spam folder! DO NOT click on 'Спам' again - you're already there! "
		guidance += "If task was to read last 10 emails from INBOX and delete spam, you should NOT be in Spam folder. Go back to Inbox! "
	}
	if isInInbox {
		guidance += "You are in INBOX folder. "
	}

	// Detect if we're viewing an email (not email list)
	isViewingEmail := strings.Contains(state.Summary.Title, "Письмо") ||
		strings.Contains(state.Summary.Title, "Письмо «") ||
		strings.Contains(state.Summary.URL, "/message/") ||
		strings.Contains(state.Summary.URL, "#/message/")

	if isViewingEmail {
		guidance += "🚨 CRITICAL: YOU ARE VIEWING AN EMAIL (not email list)! "
		guidance += "The email is already open. You need to: "
		guidance += "1) Check if this email is spam (look for spam indicators in text), "
		guidance += "2) If spam, find and click DELETE button (look for 'удалить', 'delete', trash icon, or button with 'Удалить' text), "
		guidance += "3) If not spam, go BACK to email list (look for 'Назад', 'Back', 'Входящие', or click on inbox link) to read next email. "
		guidance += "DO NOT keep clicking on the email content - it's already open! "

		// Look for delete button in snapshot
		hasDeleteButton := false
		for _, el := range state.Summary.Elements {
			textLower := strings.ToLower(el.Text)
			if strings.Contains(textLower, "удалить") ||
				strings.Contains(textLower, "delete") ||
				strings.Contains(textLower, "trash") ||
				el.Role == "button" && (strings.Contains(textLower, "удал") || strings.Contains(textLower, "del")) {
				guidance += fmt.Sprintf("FOUND DELETE BUTTON: [%d] %s:%q. Use click_by_index with index=%d to delete! ",
					el.Index, el.Role, truncateText(el.Text, 30), el.Index)
				hasDeleteButton = true
				break
			}
		}
		if !hasDeleteButton {
			guidance += "Delete button not found in snapshot - try using read_page or look for buttons with 'удалить' text. "
		}

		// Look for back/inbox button
		for _, el := range state.Summary.Elements {
			textLower := strings.ToLower(el.Text)
			if strings.Contains(textLower, "назад") ||
				strings.Contains(textLower, "back") ||
				strings.Contains(textLower, "входящие") ||
				strings.Contains(textLower, "inbox") {
				textPreview := el.Text
				if len(textPreview) > 30 {
					textPreview = textPreview[:30] + "..."
				}
				guidance += fmt.Sprintf("FOUND BACK/INBOX BUTTON: [%d] %s:%q. Use click_by_index with index=%d to go back to list! ",
					el.Index, el.Role, textPreview, el.Index)
				break
			}
		}
	}

	// Use email-specific DOM discovery
	emailRows := e.findEmailRows(state.Summary.Elements)
	if len(emailRows) > 0 {
		guidance += fmt.Sprintf("FOUND %d EMAIL ROWS IN SNAPSHOT: ", len(emailRows))
		maxShow := 10
		if len(emailRows) < maxShow {
			maxShow = len(emailRows)
		}
		for i := 0; i < maxShow; i++ {
			el := emailRows[i]
			// Show key info: index, role, first line of text (browser-use pattern)
			textPreview := strings.Split(el.Text, "\n")[0]
			if len(textPreview) > 50 {
				textPreview = textPreview[:50] + "..."
			}
			guidance += fmt.Sprintf("[%d] %s:%q; ",
				el.Index, el.Role, textPreview)
		}
		// Show available indices range
		if len(emailRows) > 0 {
			guidance += fmt.Sprintf("Available email indices: %d-%d. ", emailRows[0].Index, emailRows[len(emailRows)-1].Index)
		}
		guidance += "🚨 CRITICAL: Use click_by_index with an index from the list above (e.g., click_by_index with index="
		if len(emailRows) > 0 {
			guidance += fmt.Sprintf("%d", emailRows[0].Index)
		} else {
			guidance += "1"
		}
		guidance += ") to click on emails! DO NOT use indices that are not in the list above! "
	} else {
		// Count scroll attempts in history
		scrollCount := 0
		lastScrollWasUnchanged := false
		for i := len(state.History) - 1; i >= 0 && i >= len(state.History)-5; i-- {
			if state.History[i].Action == "scroll_page" {
				scrollCount++
			}
			if state.History[i].Action == "observation" && strings.Contains(state.History[i].Result, "no changes after scroll") {
				lastScrollWasUnchanged = true
			}
		}

		guidance += "CRITICAL: NO EMAIL ROWS FOUND IN SNAPSHOT! "

		// Check if collect_texts was used recently and returned results
		hasCollectTextsResult := false
		for i := len(state.History) - 1; i >= 0 && i >= len(state.History)-3; i-- {
			if state.History[i].Action == "collect_texts" && strings.Contains(state.History[i].Result, "items") {
				hasCollectTextsResult = true
				break
			}
		}

		// If multiple scrolls failed, FORCE use of collect_texts/read_page
		if scrollCount >= 2 || lastScrollWasUnchanged {
			guidance += "STOP SCROLLING! You've already tried scrolling " + fmt.Sprintf("%d", scrollCount) + " times and snapshot didn't change. "
			guidance += "MANDATORY ACTION: You MUST use collect_texts(\"[role='option']\") or collect_texts(\"[data-testid*='message']\") or read_page() RIGHT NOW to find emails in iframe. "
			guidance += "DO NOT use scroll_page again - it won't help! Emails are in iframe and require collect_texts/read_page to access. "
		} else {
			guidance += "Emails are likely in iframe. YOU MUST use collect_texts(\"[data-testid*='message']\") or collect_texts(\"[role='option']\") or read_page() to explore DOM and find emails. "
			guidance += "Do NOT just scroll - use read_page or collect_texts to find the email list! "
		}

		// CRITICAL: If collect_texts was used and returned results, tell LLM to use indices from result
		if hasCollectTextsResult {
			guidance += "🚨 CRITICAL: You just used collect_texts and got results with INDICES! "
			guidance += "The collect_texts result shows 'index' field for each email (e.g., items[0].index). "
			guidance += "YOU MUST use click_by_index with the EXACT index from collect_texts result (e.g., click_by_index with index from items[0].index). "
			guidance += "DO NOT use click_selector - use click_by_index with the index! "

			// Find the last collect_texts result in history
			for i := len(state.History) - 1; i >= 0 && i >= len(state.History)-3; i-- {
				if state.History[i].Action == "collect_texts" {
					// Extract index example from result if possible
					result := state.History[i].Result
					if strings.Contains(result, "index=") {
						// Try to extract first index as example
						if idx := strings.Index(result, "index="); idx > 0 {
							idxStart := idx + len("index=")
							idxEnd := strings.Index(result[idxStart:], "\n")
							if idxEnd > 0 {
								exampleIndex := result[idxStart : idxStart+idxEnd]
								guidance += fmt.Sprintf("Example: use click_by_index with index=%s from collect_texts result! ", exampleIndex)
							}
						}
					}
					break
				}
			}
		}

		// Show scrollable elements if any (from browser-use pattern)
		scrollableCount := 0
		for _, el := range state.Summary.Elements {
			if el.ScrollInfo != "" {
				scrollableCount++
			}
		}
		if scrollableCount > 0 {
			guidance += fmt.Sprintf("Found %d scrollable containers in snapshot. ", scrollableCount)
		}

		// Show some elements for debugging
		if len(state.Summary.Elements) > 0 {
			guidance += "Showing first 5 snapshot elements for context: "
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
				scrollInfoText := ""
				if el.ScrollInfo != "" {
					scrollInfoText = " (scroll: " + el.ScrollInfo + ")"
				}
				guidance += fmt.Sprintf("[%d]%s:%q%s ", i+1, el.Role, textPreview, scrollInfoText)
			}
		} else {
			guidance += "SNAPSHOT IS COMPLETELY EMPTY - use read_page() immediately to see page content!"
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

	// Exclude layout elements (not actual emails)
	if strings.Contains(selLower, "-header") || strings.Contains(selLower, "-footer") ||
		strings.Contains(selLower, "-layout") && !strings.Contains(selLower, "content") {
		return false
	}

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

		// If we have emails, find one that hasn't failed recently
		if len(emailRows) > 0 {
			// Build set of failed selectors from recent history (last 5 actions)
			failedSelectors := make(map[string]int) // selector -> failure count
			currentInboxURL := state.Summary.URL

			// Check last 5 actions for failed click_selector attempts in inbox
			for i := len(state.History) - 1; i >= 0 && i >= len(state.History)-5; i-- {
				item := state.History[i]
				if item.Action == "click_selector" &&
					strings.Contains(item.Result, "error") &&
					strings.Contains(item.URL, "/tabs/relevant") &&
					strings.Contains(currentInboxURL, "/tabs/relevant") &&
					item.Selector != "" {
					failedSelectors[item.Selector]++
				}
			}

			// Try to find an email that hasn't failed (or failed only once)
			for _, email := range emailRows {
				if email.Sel != "" {
					failCount := failedSelectors[email.Sel]
					// Skip if this selector failed 2+ times
					if failCount < 2 {
						return Decision{
							ActionName:  "click_selector",
							ActionInput: map[string]any{"selector": email.Sel},
						}, nil
					}
				}
			}

			// If all emails failed multiple times, try scrolling to load new emails
			scrollCount := e.countScrolls(state.History)
			if scrollCount < 5 {
				return Decision{
					ActionName:  "scroll_page",
					ActionInput: map[string]any{"direction": "down", "distance": 300},
				}, nil
			}

			// Last resort: try first email anyway (maybe list refreshed)
			firstEmail := emailRows[0]
			if firstEmail.Sel != "" {
				return Decision{
					ActionName:  "click_selector",
					ActionInput: map[string]any{"selector": firstEmail.Sel},
				}, nil
			}
		}

		// If no email rows found but we're in inbox, try more aggressive search
		// Look for any clickable elements that might be emails (even if not detected by isEmailRow)
		if len(emailRows) == 0 && strings.Contains(urlLower, "/tabs/relevant") {
			// Try to find elements with role="option" or role="row" (Yandex Mail specific)
			for _, el := range state.Summary.Elements {
				roleLower := strings.ToLower(el.Role)
				selLower := strings.ToLower(el.Sel)
				// Skip layout elements
				if strings.Contains(selLower, "-header") || strings.Contains(selLower, "-footer") ||
					strings.Contains(selLower, "-layout") && !strings.Contains(selLower, "content") {
					continue
				}
				// Look for email-like elements
				if (roleLower == "option" || roleLower == "row" || roleLower == "listitem") &&
					el.Sel != "" && len(el.Text) > 10 {
					// Found potential email - try clicking it
					return Decision{
						ActionName:  "click_selector",
						ActionInput: map[string]any{"selector": el.Sel},
					}, nil
				}
			}
		}
	}

	// Last resort: return error with context
	return Decision{}, fmt.Errorf("LLM failed and no fallback available: %w (url: %s, elements: %d)",
		llmErr, state.Summary.URL, len(state.Summary.Elements))
}
