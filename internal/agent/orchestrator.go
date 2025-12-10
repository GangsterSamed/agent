package agent

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/rs/zerolog"

	"github.com/polzovatel/ai-agent-for-browser-fast/internal/snapshot"
	"github.com/polzovatel/ai-agent-for-browser-fast/internal/tools"
)

type Config struct {
	MaxSteps int
}

type Task struct {
	Description string
}

type Orchestrator struct {
	cfg       Config
	planner   Planner
	tools     tools.Toolbox
	logger    zerolog.Logger
	subAgents []SubAgent // Specialized agents for different task types
	// Error tracking for adaptive handling
	errorHistory []errorRecord
	// Persistent memory for email tasks
	memory *TaskMemory
}

type TaskMemory struct {
	InInbox       bool
	ScrollCount   int
	EmailsSeen    int
	LastSnapshot  snapshot.Summary
	LastAction    string
	EmailElements []snapshot.Element
}

type errorRecord struct {
	action    string
	errorType string
	step      int
	timestamp time.Time
}

func NewOrchestrator(cfg Config, planner Planner, toolbox tools.Toolbox, logger zerolog.Logger, subAgents ...SubAgent) *Orchestrator {
	return &Orchestrator{
		cfg:       cfg,
		planner:   planner,
		tools:     toolbox,
		logger:    logger,
		subAgents: subAgents,
		memory:    &TaskMemory{},
	}
}

func (o *Orchestrator) Run(ctx context.Context, task Task, snap summaryFunc) error {
	history := make([]HistoryItem, 0, 8)
	for step := 1; step <= o.cfg.MaxSteps; step++ {
		if err := ctx.Err(); err != nil {
			return err
		}

		// Wait for stable DOM after navigation (event-driven, not fixed sleep)
		if len(history) > 0 && history[len(history)-1].Action == "navigate" {
			// Use WaitForStableDOM instead of fixed sleep for better performance
			if err := o.tools.WaitForStableDOM(ctx, 5*time.Second); err != nil {
				o.logger.Debug().Err(err).Msg("wait for stable DOM after navigate")
			}
		}

		// Re-observation loop: always get fresh snapshot at start of each step
		// For email tasks, wait for email elements to appear before snapshot
		if len(o.subAgents) > 0 {
			for _, subAgent := range o.subAgents {
				if subAgent.CanHandle(task.Description) {
					// Wait for email elements to load (for email clients like Yandex Mail)
					_, err := o.tools.Invoke(ctx, "wait_for_emails", map[string]any{"timeout_ms": 5000})
					if err != nil {
						// Ignore error - emails might already be loaded
						o.logger.Debug().Err(err).Msg("wait_for_emails")
					}
					break
				}
			}
		}

		ctxSnap, cancel := snapshot.WithDeadline(ctx, 5*time.Second)
		summary, _ := snap(ctxSnap)
		cancel()

		// ALWAYS log snapshot info for debugging (especially for email tasks)
		elemPreview := ""
		if len(summary.Elements) > 0 {
			maxPreview := 10
			if len(summary.Elements) < maxPreview {
				maxPreview = len(summary.Elements)
			}
			for i := 0; i < maxPreview; i++ {
				el := summary.Elements[i]
				scrollInfoText := ""
				if el.ScrollInfo != "" {
					scrollInfoText = " (scroll:" + el.ScrollInfo + ")"
				}
				// Use element's index (browser-use pattern)
				elemPreview += fmt.Sprintf(" [%d]%s:%q%s", el.Index, el.Role, truncateTextForDebug(el.Text, 40), scrollInfoText)
			}
		} else {
			elemPreview = "EMPTY - no elements found!"
		}
		o.logger.Info().
			Int("step", step).
			Str("url", summary.URL).
			Str("title", summary.Title).
			Int("elements", len(summary.Elements)).
			Str("preview", elemPreview).
			Msg("snapshot")

		state := State{
			Task:    task.Description,
			Step:    step,
			History: last(history, 5),
			Summary: summary,
			Tools:   o.tools.Describe(),
		}

		// Sub-agent architecture: delegate to specialized agent if available
		var dec Decision
		var err error
		subAgent := o.selectSubAgent(task.Description)
		if subAgent != nil {
			o.logger.Info().
				Str("sub_agent", subAgent.Name()).
				Str("task", task.Description).
				Msg("delegating to specialized agent")

			// Non-trivial solution: if EmailAgent and no email rows found, wait for them first
			// Skip on first step (about:blank) or if already navigated
			if subAgent.Name() == "EmailAgent" && summary.URL != "about:blank" && summary.URL != "" && o.shouldWaitForEmails(summary) {
				o.logger.Info().Msg("no email rows found - waiting for emails to appear")
				_, waitErr := o.tools.Invoke(ctx, "wait_for_emails", map[string]any{"timeout_ms": 10000})
				if waitErr == nil {
					// Update snapshot after waiting
					time.Sleep(500 * time.Millisecond)
					ctxSnapWait, cancelWait := snapshot.WithDeadline(ctx, 3*time.Second)
					summaryWait, _ := snap(ctxSnapWait)
					cancelWait()
					summary = summaryWait
					state.Summary = summaryWait
				}
			}

			dec, err = subAgent.Next(ctx, state)
		} else {
			// Use default planner
			dec, err = o.planner.Next(ctx, state)
		}
		if err != nil {
			return fmt.Errorf("planner: %w", err)
		}

		if dec.Finish {
			fmt.Printf("✅ %s\n", dec.Message)
			return nil
		}

		limit := 3
		if dec.ActionName == "scroll_page" {
			limit = 20 // allow many scrolls for heavy SPAs that load content dynamically
		}
		if dec.ActionName == "click_by_index" {
			limit = 2 // Strict limit for click_by_index - prevent loops
		}

		// Prevent clicking on already active folder (e.g., clicking "Спам" when already in Spam folder)
		if dec.ActionName == "click_by_index" || dec.ActionName == "click_selector" {
			index, hasIndex := dec.ActionInput["index"]
			selector, hasSelector := dec.ActionInput["selector"].(string)

			// Check if clicking on Spam folder when already in Spam
			if strings.Contains(summary.URL, "#spam") || strings.Contains(summary.Title, "Спам") {
				if hasIndex {
					// Find element by index
					var foundElement *snapshot.Element
					for i := range summary.Elements {
						if summary.Elements[i].Index == int(index.(float64)) {
							foundElement = &summary.Elements[i]
							break
						}
					}
					if foundElement != nil && strings.Contains(strings.ToLower(foundElement.Text), "спам") {
						o.logger.Warn().
							Int("index", int(index.(float64))).
							Msg("preventing click on already active Spam folder")
						// Skip this action and continue
						history = append(history, HistoryItem{
							Action: "observation",
							Result: "skipped: already in Spam folder, don't click on it again",
							URL:    summary.URL,
						})
						continue
					}
				}
				if hasSelector && (strings.Contains(selector, "Спам") || strings.Contains(strings.ToLower(selector), "spam")) {
					o.logger.Warn().Str("selector", selector).Msg("preventing click on already active Spam folder")
					history = append(history, HistoryItem{
						Action: "observation",
						Result: "skipped: already in Spam folder, don't click on it again",
						URL:    summary.URL,
					})
					continue
				}
			}
		}
		// Pass URL context for tooManyRepeats check
		checkInput := make(map[string]any)
		for k, v := range dec.ActionInput {
			checkInput[k] = v
		}
		checkInput["_url"] = summary.URL
		if tooManyRepeats(history, dec.ActionName, checkInput, limit) {
			return fmt.Errorf("too many repeated actions: %s (limit: %d). Try a different action", dec.ActionName, limit)
		}

		// Security layer: check for destructive actions
		if requiresConfirmation(dec.ActionName, dec.ActionInput) {
			confirmed, err := o.requestConfirmation(ctx, dec.ActionName, dec.ActionInput)
			if err != nil {
				return fmt.Errorf("confirmation request failed: %w", err)
			}
			if !confirmed {
				item := HistoryItem{
					Action: dec.ActionName,
					Result: "cancelled by user",
					URL:    summary.URL,
				}
				if dec.ActionName == "click_selector" {
					if sel, ok := dec.ActionInput["selector"].(string); ok {
						item.Selector = sel
					}
				}
				history = append(history, item)
				fmt.Printf("⚠️  Action cancelled by user: %s\n", dec.ActionName)
				continue
			}
		}

		// Update memory
		o.updateMemory(dec.ActionName, summary)

		// Handle click_by_index: convert to click_selector using element from snapshot (browser-use pattern)
		var foundElement *snapshot.Element // Keep reference for bbox fallback
		if dec.ActionName == "click_by_index" {
			index, ok := dec.ActionInput["index"].(float64)
			if !ok {
				indexInt, okInt := dec.ActionInput["index"].(int)
				if okInt {
					index = float64(indexInt)
				} else {
					return fmt.Errorf("invalid index type for click_by_index")
				}
			}
			indexInt := int(index)

			// Find element by index in snapshot
			for i := range summary.Elements {
				if summary.Elements[i].Index == indexInt {
					foundElement = &summary.Elements[i]
					break
				}
			}

			if foundElement == nil {
				// Build list of available indices for better error message
				availableIndices := make([]int, 0, len(summary.Elements))
				for _, el := range summary.Elements {
					availableIndices = append(availableIndices, el.Index)
				}
				return fmt.Errorf("element with index %d not found in current snapshot. Available indices: %v. Use an index from the current snapshot", indexInt, availableIndices)
			}

			// Browser-use pattern: don't check DOM existence - just try to click
			// If selector doesn't work, fallback to coordinates from bbox
			// This handles virtualized lists and iframes better

			// Convert to click_selector
			o.logger.Debug().
				Int("index", indexInt).
				Str("selector", foundElement.Sel).
				Str("text", truncateTextForDebug(foundElement.Text, 30)).
				Str("bbox", foundElement.BBox).
				Msg("converting click_by_index to click_selector")

			dec.ActionName = "click_selector"
			dec.ActionInput = map[string]any{"selector": foundElement.Sel}
		}

		result, err := o.tools.Invoke(ctx, dec.ActionName, dec.ActionInput)
		if err != nil {
			// Browser-use pattern: if click_selector fails and we have bbox, try coordinates
			if dec.ActionName == "click_selector" && foundElement != nil && foundElement.BBox != "" {
				// Parse bbox: "x,y,width,height" -> center point
				var x, y, w, h float64
				if n, _ := fmt.Sscanf(foundElement.BBox, "%f,%f,%f,%f", &x, &y, &w, &h); n == 4 {
					// Click at center of bbox
					centerX := x + w/2
					centerY := y + h/2
					o.logger.Info().
						Float64("x", centerX).
						Float64("y", centerY).
						Str("bbox", foundElement.BBox).
						Msg("click_selector failed, trying click_coordinates from bbox")

					coordResult, coordErr := o.tools.Invoke(ctx, "click_coordinates", map[string]any{
						"x": int(centerX),
						"y": int(centerY),
					})
					if coordErr == nil {
						// Success with coordinates!
						result = coordResult
						err = nil
						o.logger.Info().Msg("click_coordinates succeeded as fallback")
					}
				}
			}

			if err != nil {
				// Check if error is selector parsing error - skip retry for invalid selectors
				errorType := o.analyzeError(err)
				if errorType == "selector_parse_error" {
					o.logger.Warn().
						Err(err).
						Str("action", dec.ActionName).
						Msg("selector parse error - skipping retry, will try alternative")
					item := HistoryItem{
						Action: dec.ActionName,
						Result: "error: invalid selector",
						URL:    summary.URL,
					}
					if dec.ActionName == "click_selector" {
						if sel, ok := dec.ActionInput["selector"].(string); ok {
							item.Selector = sel
						}
					}
					history = append(history, item)
					// Update snapshot and continue
					time.Sleep(500 * time.Millisecond)
					ctxSnapErr, cancelErr := snapshot.WithDeadline(ctx, 3*time.Second)
					summaryErr, _ := snap(ctxSnapErr)
					cancelErr()
					summary = summaryErr
					continue
				}

				// Record error for adaptive handling
				o.errorHistory = append(o.errorHistory, errorRecord{
					action:    dec.ActionName,
					errorType: errorType,
					step:      step,
					timestamp: time.Now(),
				})
				// Keep only last 10 errors
				if len(o.errorHistory) > 10 {
					o.errorHistory = o.errorHistory[len(o.errorHistory)-10:]
				}

				o.logger.Warn().
					Err(err).
					Str("action", dec.ActionName).
					Str("error_type", errorType).
					Msg("tool error")

				// Re-observation: update snapshot before retry
				time.Sleep(500 * time.Millisecond) // Wait for DOM to settle
				ctxSnapRetry, cancelRetry := snapshot.WithDeadline(ctx, 3*time.Second)
				freshSummary, _ := snap(ctxSnapRetry)
				cancelRetry()

				// Adaptive error handling: try multiple strategies with fresh snapshot
				recoveredAction, recoveredResult, success := o.handleErrorAdaptively(ctx, dec, freshSummary, snap, history, step)
				if success {
					// Successfully recovered from error
					item := HistoryItem{
						Action: recoveredAction,
						Result: recoveredResult.Observation,
						URL:    freshSummary.URL,
					}
					if recoveredAction == "click_selector" {
						// Try to extract selector from original decision
						if sel, ok := dec.ActionInput["selector"].(string); ok {
							item.Selector = sel
						}
					}
					history = append(history, item)
					fmt.Printf("agent[%d]: %s (recovered) -> %s\n", step, recoveredAction, truncate(recoveredAction, recoveredResult.Observation))
					// Re-observation loop: update snapshot after successful recovery
					time.Sleep(800 * time.Millisecond)
					ctxSnapAfter, cancelAfter := snapshot.WithDeadline(ctx, 3*time.Second)
					summaryAfter, _ := snap(ctxSnapAfter)
					cancelAfter()
					summary = summaryAfter // Update summary for next iteration
					o.updateMemory(recoveredAction, summaryAfter)
					// Delay after click actions
					if recoveredAction == "click_role" || recoveredAction == "click_selector" || recoveredAction == "click_text" {
						time.Sleep(1 * time.Second)
					}
					continue
				}

				// All recovery strategies failed
				item := HistoryItem{
					Action: dec.ActionName,
					Result: "error: " + err.Error(),
					URL:    summary.URL,
				}
				if dec.ActionName == "click_selector" {
					if sel, ok := dec.ActionInput["selector"].(string); ok {
						item.Selector = sel
					}
				}
				history = append(history, item)
				// Re-observation: update snapshot even after error to see what changed
				time.Sleep(500 * time.Millisecond)
				ctxSnapErr, cancelErr := snapshot.WithDeadline(ctx, 3*time.Second)
				summaryErr, _ := snap(ctxSnapErr)
				cancelErr()
				summary = summaryErr // Update summary for next iteration
				// Don't give up immediately - let planner decide next action with fresh snapshot
				continue
			}
		}
		fmt.Printf("agent[%d]: %s -> %s\n", step, dec.ActionName, truncate(dec.ActionName, result.Observation))
		// Create history item with selector and URL context
		item := HistoryItem{
			Action: dec.ActionName,
			Result: result.Observation,
			URL:    summary.URL,
		}
		if dec.ActionName == "click_selector" {
			if sel, ok := dec.ActionInput["selector"].(string); ok {
				item.Selector = sel
			}
		}
		history = append(history, item)

		// Observation Stabilization: wait after scroll, then check if DOM changed
		if dec.ActionName == "scroll_page" {
			time.Sleep(1000 * time.Millisecond) // Wait for virtual list to render
			ctxSnapStable, cancelStable := snapshot.WithDeadline(ctx, 3*time.Second)
			stableSummary, _ := snap(ctxSnapStable)
			cancelStable()
			// Compare with previous snapshot
			if !o.snapshotChanged(summary, stableSummary) {
				o.logger.Info().Msg("snapshot unchanged after scroll - stopping scroll loop")
				history = append(history, HistoryItem{
					Action: "observation",
					Result: "no changes after scroll - emails may be in iframe, use collect_texts or read_page",
				})
				// Update summary to reflect that scroll didn't help
				summary = stableSummary
			} else {
				summary = stableSummary
			}
		} else {
			// Re-observation loop: update snapshot after every action
			time.Sleep(800 * time.Millisecond) // Wait for DOM to update after action
			ctxSnapAfter, cancelAfter := snapshot.WithDeadline(ctx, 3*time.Second)
			summaryAfter, _ := snap(ctxSnapAfter)
			cancelAfter()

			summary = summaryAfter // Update summary for next iteration
		}

		// Update memory after action
		o.updateMemory(dec.ActionName, summary)

		// Delay after click actions to let heavy SPAs update
		if dec.ActionName == "click_role" || dec.ActionName == "click_selector" || dec.ActionName == "click_text" {
			time.Sleep(1 * time.Second)
		}
	}
	return fmt.Errorf("step limit reached")
}

type summaryFunc func(ctx context.Context) (snapshot.Summary, error)

func last(items []HistoryItem, n int) []HistoryItem {
	if len(items) <= n {
		return items
	}
	return items[len(items)-n:]
}

func truncate(action, s string) string {
	if action == "read_page" {
		return "(read_page data omitted)"
	}
	if len(s) > 160 {
		return s[:160] + "..."
	}
	return s
}

func truncateTextForDebug(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

func tooManyRepeats(history []HistoryItem, action string, input map[string]any, limit int) bool {
	if limit <= 0 {
		return false
	}
	if len(history) < limit {
		return false
	}

	// For click_selector, check action + selector + URL context
	if action == "click_selector" {
		selector, _ := input["selector"].(string)
		currentURL := ""
		if url, ok := input["_url"].(string); ok {
			currentURL = url
		}

		count := 0
		for i := len(history) - 1; i >= 0 && i >= len(history)-limit; i-- {
			if history[i].Action == action &&
				history[i].Selector == selector &&
				history[i].URL == currentURL {
				count++
			}
		}
		return count >= limit
	}

	// For click_by_index, check by action + URL (strict limit to prevent loops)
	if action == "click_by_index" {
		currentURL := ""
		if url, ok := input["_url"].(string); ok {
			currentURL = url
		}

		count := 0
		for i := len(history) - 1; i >= 0 && i >= len(history)-limit; i-- {
			if history[i].Action == action && history[i].URL == currentURL {
				// Same action on same URL - likely a loop
				count++
			}
		}
		return count >= limit
	}

	// For other actions, check only by action name
	for i := 0; i < limit; i++ {
		if history[len(history)-1-i].Action != action {
			return false
		}
	}
	return true
}

// requiresConfirmation checks if an action is destructive and requires user confirmation
func requiresConfirmation(action string, input map[string]any) bool {
	// Check action name for destructive keywords
	destructiveActions := map[string]bool{
		"click_selector": false, // will check by selector/text
		"click_role":     false, // will check by role
		"click_text":     false, // will check by text
		"fill":           false, // will check by context
	}

	if !destructiveActions[action] {
		return false
	}

	// Check input for destructive keywords
	var textToCheck string
	if selector, ok := input["selector"].(string); ok {
		textToCheck = selector
	}
	if role, ok := input["role"].(string); ok {
		textToCheck = role
	}
	if text, ok := input["text"].(string); ok {
		textToCheck = text
	}
	if label, ok := input["label"].(string); ok {
		textToCheck = label
	}

	// Keywords that indicate destructive actions (case-insensitive)
	destructiveKeywords := []string{
		"delete", "удалить", "удалить письмо", "удалить сообщение",
		"payment", "оплатить", "купить", "buy", "purchase", "checkout", "pay",
		"remove", "очистить", "clear",
		"submit", "отправить", "подтвердить",
		"confirm", "подтвердить",
		"spam", "спам",
		"cancel", "отменить",
		"archive", "архив",
		"unsubscribe", "отписаться",
	}

	textToCheckLower := strings.ToLower(fmt.Sprintf("%v", textToCheck))
	for _, keyword := range destructiveKeywords {
		if contains(textToCheckLower, strings.ToLower(keyword)) {
			return true
		}
	}

	return false
}

// requestConfirmation asks user for confirmation before destructive action
func (o *Orchestrator) requestConfirmation(ctx context.Context, action string, input map[string]any) (bool, error) {
	// Build description of the action
	actionDesc := fmt.Sprintf("Action: %s", action)
	if selector, ok := input["selector"].(string); ok {
		actionDesc += fmt.Sprintf(" on selector: %s", selector)
	}
	if role, ok := input["role"].(string); ok {
		actionDesc += fmt.Sprintf(" on role: %s", role)
	}
	if text, ok := input["text"].(string); ok {
		actionDesc += fmt.Sprintf(" on text: %s", text)
	}

	prompt := fmt.Sprintf("⚠️  SECURITY CHECK: This action may be destructive:\n%s\n\nDo you want to proceed? (yes/no): ", actionDesc)

	// Use request_user_input tool to ask user
	result, err := o.tools.Invoke(ctx, "request_user_input", map[string]any{
		"prompt": prompt,
	})
	if err != nil {
		return false, err
	}

	// Parse response
	response := result.Observation
	if contains(response, "yes") || contains(response, "да") || contains(response, "y") {
		return true, nil
	}
	return false, nil
}

func contains(s, substr string) bool {
	if len(s) < len(substr) {
		return false
	}
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// analyzeError categorizes error type for adaptive handling
func (o *Orchestrator) analyzeError(err error) string {
	errStr := strings.ToLower(err.Error())
	switch {
	case strings.Contains(errStr, "badstring") || strings.Contains(errStr, "unsupported token") || strings.Contains(errStr, "parsing selector"):
		return "selector_parse_error"
	case strings.Contains(errStr, "timeout"):
		return "timeout"
	case strings.Contains(errStr, "not found") || strings.Contains(errStr, "not visible"):
		return "element_not_found"
	case strings.Contains(errStr, "not clickable") || strings.Contains(errStr, "not interactable"):
		return "not_interactable"
	case strings.Contains(errStr, "stale") || strings.Contains(errStr, "detached"):
		return "stale_element"
	case strings.Contains(errStr, "network") || strings.Contains(errStr, "connection"):
		return "network_error"
	default:
		return "unknown"
	}
}

// handleErrorAdaptively tries multiple recovery strategies based on error type
func (o *Orchestrator) handleErrorAdaptively(ctx context.Context, dec Decision, summary snapshot.Summary, snap summaryFunc, history []HistoryItem, step int) (string, tools.Result, bool) {
	// Don't retry if we've already tried too many times for this action
	if o.hasRecentRetries(dec.ActionName, 2) {
		return "", tools.Result{}, false
	}

	errorType := o.errorHistory[len(o.errorHistory)-1].errorType

	// Strategy 1: Wait and retry (for timeout/stale element)
	if errorType == "timeout" || errorType == "stale_element" {
		o.logger.Info().Str("strategy", "wait_retry").Msg("trying wait and retry")
		time.Sleep(2 * time.Second)
		// Get fresh snapshot
		ctxSnap, cancel := snapshot.WithDeadline(ctx, 3*time.Second)
		freshSummary, _ := snap(ctxSnap)
		cancel()
		_ = freshSummary // use freshSummary if needed
		// Retry original action
		retryResult, err := o.tools.Invoke(ctx, dec.ActionName, dec.ActionInput)
		if err == nil {
			return dec.ActionName, retryResult, true
		}
	}

	// Strategy 2: Try alternative action methods (for click actions)
	if dec.ActionName == "click_selector" || dec.ActionName == "click_role" || dec.ActionName == "click_text" {
		alternatives := o.generateAlternatives(dec, summary)
		for _, alt := range alternatives {
			o.logger.Info().
				Str("original", dec.ActionName).
				Str("alternative", alt.action).
				Msg("trying alternative action")
			altResult, err := o.tools.Invoke(ctx, alt.action, alt.input)
			if err == nil {
				return alt.action, altResult, true
			}
		}

		// Strategy 2b: Try fuzzy text matching if we have text
		if dec.ActionName == "click_selector" {
			if text := o.extractTextFromSelector(dec, summary); text != "" {
				o.logger.Info().Str("strategy", "fuzzy_text").Str("text", text).Msg("trying fuzzy text match")
				fuzzyResult, err := o.tools.Invoke(ctx, "click_text_fuzzy", map[string]any{"text": text})
				if err == nil {
					return "click_text_fuzzy", fuzzyResult, true
				}
			}
		}

		// Strategy 2c: Try clicking by coordinates from bbox (last resort)
		if coords := o.extractCoordinates(dec, summary); coords.x > 0 && coords.y > 0 {
			o.logger.Info().
				Float64("x", coords.x).
				Float64("y", coords.y).
				Msg("trying click by coordinates")
			coordResult, err := o.tools.Invoke(ctx, "click_coordinates", map[string]any{
				"x": int(coords.x),
				"y": int(coords.y),
			})
			if err == nil {
				return "click_coordinates", coordResult, true
			}
		}
	}

	// Strategy 3: Find similar element in snapshot (for element_not_found)
	if errorType == "element_not_found" {
		similar := o.findSimilarElement(dec, summary)
		if similar.action != "" {
			o.logger.Info().
				Str("original", dec.ActionName).
				Str("similar", similar.action).
				Msg("trying similar element")
			similarResult, err := o.tools.Invoke(ctx, similar.action, similar.input)
			if err == nil {
				return similar.action, similarResult, true
			}
		}
	}

	// Strategy 4: Scroll to make element visible (for not_interactable)
	if errorType == "not_interactable" {
		o.logger.Info().Str("strategy", "scroll_to_element").Msg("trying scroll to element")
		// Try scrolling to element location if we have bbox info
		if err := o.scrollToElement(ctx, dec, summary); err == nil {
			time.Sleep(1 * time.Second)
			retryResult, err := o.tools.Invoke(ctx, dec.ActionName, dec.ActionInput)
			if err == nil {
				return dec.ActionName, retryResult, true
			}
		}
	}

	return "", tools.Result{}, false
}

type alternativeAction struct {
	action string
	input  map[string]any
}

// generateAlternatives creates alternative actions based on current decision and snapshot
func (o *Orchestrator) generateAlternatives(dec Decision, summary snapshot.Summary) []alternativeAction {
	var alternatives []alternativeAction

	switch dec.ActionName {
	case "click_selector":
		if selector, ok := dec.ActionInput["selector"].(string); ok {
			// Try click_text if we can find text in snapshot
			text := o.findTextBySelector(selector, summary)
			if text != "" {
				alternatives = append(alternatives, alternativeAction{
					action: "click_text",
					input:  map[string]any{"text": text},
				})
			}
			// Try click_role if selector has role
			if role := o.extractRoleFromSelector(selector, summary); role != "" {
				alternatives = append(alternatives, alternativeAction{
					action: "click_role",
					input:  map[string]any{"role": role, "label": text},
				})
			}
		}
	case "click_role":
		if role, ok := dec.ActionInput["role"].(string); ok {
			label, _ := dec.ActionInput["label"].(string)
			text := label
			if text == "" {
				// Try to find text from snapshot
				text = o.findTextByRole(role, summary)
			}
			// Try click_selector
			alternatives = append(alternatives, alternativeAction{
				action: "click_selector",
				input:  map[string]any{"selector": fmt.Sprintf("[role='%s']", role)},
			})
			// Try click_text if we have text
			if text != "" {
				alternatives = append(alternatives, alternativeAction{
					action: "click_text",
					input:  map[string]any{"text": text},
				})
			}
		}
	case "click_text":
		if text, ok := dec.ActionInput["text"].(string); ok {
			// Try click_role with common roles
			for _, role := range []string{"button", "link", "menuitem"} {
				alternatives = append(alternatives, alternativeAction{
					action: "click_role",
					input:  map[string]any{"role": role, "label": text},
				})
			}
			// Try click_selector if we can find matching selector
			if sel := o.findSelectorByText(text, summary); sel != "" {
				alternatives = append(alternatives, alternativeAction{
					action: "click_selector",
					input:  map[string]any{"selector": sel},
				})
			}
		}
	}

	return alternatives
}

// findSimilarElement finds a similar element in snapshot when original is not found
func (o *Orchestrator) findSimilarElement(dec Decision, summary snapshot.Summary) alternativeAction {
	// Extract search criteria from original action
	var searchText string
	if text, ok := dec.ActionInput["text"].(string); ok {
		searchText = strings.ToLower(text)
	}
	if selector, ok := dec.ActionInput["selector"].(string); ok {
		searchText = strings.ToLower(selector)
	}

	// Search in snapshot elements for similar text
	for _, elem := range summary.Elements {
		elemText := strings.ToLower(elem.Text)
		if searchText != "" && strings.Contains(elemText, searchText) {
			// Found similar element, try to click it
			if elem.Sel != "" {
				return alternativeAction{
					action: "click_selector",
					input:  map[string]any{"selector": elem.Sel},
				}
			}
			if elem.Role != "" {
				return alternativeAction{
					action: "click_role",
					input:  map[string]any{"role": elem.Role, "label": elem.Text},
				}
			}
		}
	}

	return alternativeAction{}
}

// Helper methods for alternative generation
func (o *Orchestrator) findTextBySelector(selector string, summary snapshot.Summary) string {
	for _, elem := range summary.Elements {
		if elem.Sel == selector || strings.Contains(elem.Sel, selector) {
			return elem.Text
		}
	}
	return ""
}

func (o *Orchestrator) extractRoleFromSelector(selector string, summary snapshot.Summary) string {
	for _, elem := range summary.Elements {
		if elem.Sel == selector || strings.Contains(elem.Sel, selector) {
			return elem.Role
		}
	}
	return ""
}

func (o *Orchestrator) findSelectorByText(text string, summary snapshot.Summary) string {
	textLower := strings.ToLower(text)
	for _, elem := range summary.Elements {
		if strings.Contains(strings.ToLower(elem.Text), textLower) {
			return elem.Sel
		}
	}
	return ""
}

func (o *Orchestrator) findTextByRole(role string, summary snapshot.Summary) string {
	for _, elem := range summary.Elements {
		if elem.Role == role {
			return elem.Text
		}
	}
	return ""
}

func (o *Orchestrator) scrollToElement(ctx context.Context, dec Decision, summary snapshot.Summary) error {
	// Try to find element bbox in snapshot
	for _, elem := range summary.Elements {
		if elem.Sel != "" {
			// Extract bbox and scroll to it
			if elem.BBox != "" {
				// Scroll down a bit to make element visible
				_, err := o.tools.Invoke(ctx, "scroll_page", map[string]any{
					"direction": "down",
					"distance":  300,
				})
				return err
			}
		}
	}
	return fmt.Errorf("element bbox not found")
}

func (o *Orchestrator) hasRecentRetries(action string, maxRetries int) bool {
	count := 0
	for i := len(o.errorHistory) - 1; i >= 0 && i >= len(o.errorHistory)-5; i-- {
		if o.errorHistory[i].action == action {
			count++
		}
	}
	return count >= maxRetries
}

type coordinates struct {
	x, y float64
}

// extractCoordinates tries to extract click coordinates from element bbox
func (o *Orchestrator) extractCoordinates(dec Decision, summary snapshot.Summary) coordinates {
	if dec.ActionName != "click_selector" {
		return coordinates{}
	}
	selector, ok := dec.ActionInput["selector"].(string)
	if !ok {
		return coordinates{}
	}

	// Find element by selector in snapshot
	for _, el := range summary.Elements {
		if el.Sel == selector || strings.Contains(el.Sel, selector) {
			// Parse bbox: "x,y,width,height"
			parts := strings.Split(el.BBox, ",")
			if len(parts) == 4 {
				var x, y, w, h float64
				fmt.Sscanf(parts[0], "%f", &x)
				fmt.Sscanf(parts[1], "%f", &y)
				fmt.Sscanf(parts[2], "%f", &w)
				fmt.Sscanf(parts[3], "%f", &h)
				// Click at center of element
				return coordinates{
					x: x + w/2,
					y: y + h/2,
				}
			}
		}
	}
	return coordinates{}
}

// extractTextFromSelector extracts text from element for fuzzy matching
func (o *Orchestrator) extractTextFromSelector(dec Decision, summary snapshot.Summary) string {
	if dec.ActionName != "click_selector" {
		return ""
	}
	selector, ok := dec.ActionInput["selector"].(string)
	if !ok {
		return ""
	}

	// Find element by selector in snapshot
	for _, el := range summary.Elements {
		if el.Sel == selector || strings.Contains(el.Sel, selector) {
			// Use first line of text, limit length
			text := strings.Split(el.Text, "\n")[0]
			if len(text) > 50 {
				text = text[:50]
			}
			return strings.TrimSpace(text)
		}
	}
	return ""
}

// selectSubAgent chooses the appropriate specialized agent for the task
func (o *Orchestrator) selectSubAgent(task string) SubAgent {
	for _, agent := range o.subAgents {
		if agent.CanHandle(task) {
			return agent
		}
	}
	return nil // No specialized agent found, use default planner
}

// updateMemory updates persistent memory about task progress
func (o *Orchestrator) updateMemory(action string, summary snapshot.Summary) {
	if o.memory == nil {
		o.memory = &TaskMemory{}
	}
	o.memory.LastAction = action
	o.memory.LastSnapshot = summary

	// Check if we're in inbox
	urlLower := strings.ToLower(summary.URL)
	titleLower := strings.ToLower(summary.Title)
	if strings.Contains(urlLower, "inbox") || strings.Contains(urlLower, "входящие") ||
		strings.Contains(titleLower, "входящие") || strings.Contains(titleLower, "inbox") {
		o.memory.InInbox = true
	}

	// Count scrolls
	if action == "scroll_page" {
		o.memory.ScrollCount++
	}
}

// shouldWaitForEmails checks if we should wait for email elements
func (o *Orchestrator) shouldWaitForEmails(summary snapshot.Summary) bool {
	// Don't wait on empty/blank pages
	if summary.URL == "" || summary.URL == "about:blank" {
		return false
	}

	// Check if we're on an email page
	urlLower := strings.ToLower(summary.URL)
	titleLower := strings.ToLower(summary.Title)
	if !strings.Contains(urlLower, "mail") && !strings.Contains(titleLower, "почт") && !strings.Contains(titleLower, "mail") {
		return false
	}

	// Check if we have email-like elements
	emailIndicators := []string{"@", "от:", "from:", "тема", "subject", "письм", "message"}
	hasEmailElements := false
	for _, el := range summary.Elements {
		textLower := strings.ToLower(el.Text)
		attrLower := strings.ToLower(el.Attr)
		for _, indicator := range emailIndicators {
			if strings.Contains(textLower, indicator) || strings.Contains(attrLower, indicator) {
				hasEmailElements = true
				break
			}
		}
		// Check for email-specific attributes
		if strings.Contains(attrLower, "data-testid") &&
			(strings.Contains(attrLower, "message") || strings.Contains(attrLower, "mail")) {
			hasEmailElements = true
			break
		}
	}

	// If we have email elements, don't wait
	if hasEmailElements {
		return false
	}

	// No email indicators found - should wait
	return true
}

// snapshotChanged compares two snapshots to see if DOM changed significantly
func (o *Orchestrator) snapshotChanged(old, new snapshot.Summary) bool {
	// Simple comparison: check if element count changed or URL changed
	if old.URL != new.URL {
		return true
	}
	if len(old.Elements) != len(new.Elements) {
		return true
	}
	// Check if we have new email-like elements
	if len(new.Elements) > len(old.Elements) {
		// More elements = likely new content loaded
		return true
	}
	// Check if element texts changed (simple heuristic)
	if len(old.Elements) > 0 && len(new.Elements) > 0 {
		// Compare first few elements
		maxCompare := 10
		if len(old.Elements) < maxCompare {
			maxCompare = len(old.Elements)
		}
		if len(new.Elements) < maxCompare {
			maxCompare = len(new.Elements)
		}
		for i := 0; i < maxCompare; i++ {
			if old.Elements[i].Text != new.Elements[i].Text {
				return true
			}
		}
	}
	return false
}
