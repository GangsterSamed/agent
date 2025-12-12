package snapshot

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/playwright-community/playwright-go"

	"github.com/polzovatel/ai-agent-for-browser-fast/internal/browser"
)

// Element describes minimal info about interactive node.
type Element struct {
	Index      int    `json:"index"`                 // Interactive index (1-based, like browser-use)
	Role       string `json:"role"`                  // Role from CDP (link, button, etc.)
	Text       string `json:"text"`                  // Text content
	Attr       string `json:"attr"`                  // Attributes
	BBox       string `json:"bbox"`                  // Bounding box
	Sel        string `json:"selector"`              // CSS selector
	ScrollInfo string `json:"scroll_info,omitempty"` // Scroll info like "0.0↑ 2.5↓ 0%" (from browser-use pattern)
	Depth      int    `json:"depth"`                 // Depth in hierarchy (0 = root, for indentation)
	NodeId     string `json:"node_id"`               // CDP node ID (for building hierarchy)
	ParentId   string `json:"parent_id"`             // Parent node ID (for building hierarchy)
}

// Summary is a compact view of current page.
type Summary struct {
	URL       string
	Title     string
	Visible   string
	Elements  []Element
	PageStats PageStatistics // Page statistics like browser-use
}

// PageStatistics contains page-level statistics
type PageStatistics struct {
	Links            int
	Iframes          int
	ScrollContainers int
	Interactive      int
	TotalElements    int
}

// ToMap returns summary as a JSON-friendly map.
func (s Summary) ToMap() map[string]any {
	return map[string]any{
		"url":        s.URL,
		"title":      s.Title,
		"visible":    s.Visible,
		"elements":   s.Elements,
		"page_stats": s.PageStats,
	}
}

// calculatePageStatistics calculates page-level statistics from elements
func calculatePageStatistics(elems []Element) PageStatistics {
	stats := PageStatistics{
		TotalElements: len(elems),
	}

	for _, el := range elems {
		roleLower := strings.ToLower(el.Role)

		// Count links
		if roleLower == "link" || strings.Contains(el.Attr, "href:") {
			stats.Links++
		}

		// Count iframes (usually have role or specific attributes)
		if roleLower == "document" || strings.Contains(el.Attr, "iframe") {
			stats.Iframes++
		}

		// Count scroll containers
		if el.ScrollInfo != "" {
			stats.ScrollContainers++
		}

		// Count interactive elements
		if el.Role != "" && el.Role != "generic" && el.Role != "presentation" {
			stats.Interactive++
		}
	}

	return stats
}

func Collect(ctx context.Context, ctrl browser.Controller) (Summary, error) {
	page := ctrl.Page()
	title, _ := page.Title()
	url := page.URL()

	text, _ := page.InnerText("body")
	if len(text) > 1200 {
		text = text[:1200]
	}

	// Use shorter timeout for snapshot collection to avoid hanging
	snapshotCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	elems, _ := collectInteractive(snapshotCtx, page, 200) // Reduced from 500 to 200 for speed

	// Like browser-use-reference: show ALL interactive elements, don't filter by relevance
	// Filter only non-interactive elements, keep all interactive ones
	actionableRoles := map[string]bool{
		"button": true, "link": true, "textbox": true, "checkbox": true,
		"radio": true, "radiogroup": true, "combobox": true, "listitem": true, "menuitem": true,
		"tab": true, "option": true, "article": true, "row": true,
		"list": true, "listbox": true, "treeitem": true, "cell": true,
	}

	// Separate interactive and non-interactive elements
	interactiveElems := make([]Element, 0)
	nonInteractiveElems := make([]Element, 0)

	for i := range elems {
		roleLower := strings.ToLower(elems[i].Role)
		if actionableRoles[roleLower] {
			interactiveElems = append(interactiveElems, elems[i])
		} else {
			nonInteractiveElems = append(nonInteractiveElems, elems[i])
		}
	}

	// Filter only non-interactive elements (limit to 50 for context)
	filteredNonInteractive := filterAndRankElements(nonInteractiveElems, 50)

	// Combine: ALL interactive + filtered non-interactive
	filteredElems := make([]Element, 0, len(interactiveElems)+len(filteredNonInteractive))
	filteredElems = append(filteredElems, interactiveElems...)
	filteredElems = append(filteredElems, filteredNonInteractive...)

	// Assign interactive indices (1-based, like browser-use)
	for i := range filteredElems {
		filteredElems[i].Index = i + 1
	}

	// Calculate page statistics
	stats := calculatePageStatistics(filteredElems)

	return Summary{
		URL:       url,
		Title:     title,
		Visible:   strings.TrimSpace(text),
		Elements:  filteredElems,
		PageStats: stats,
	}, nil
}

func (s Summary) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "URL: %s\nTITLE: %s\nTEXT: %s\nELEMENTS:\n", s.URL, s.Title, s.Visible)
	for i, el := range s.Elements {
		fmt.Fprintf(&b, "%d) role=%s text=%s attr=%s bbox=%s\n", i+1, el.Role, el.Text, el.Attr, el.BBox)
	}
	return b.String()
}

func collectInteractive(ctx context.Context, page playwright.Page, limit int) ([]Element, error) {
	// Try to use CDP Accessibility.getFullAXTree (like browser-use-reference)
	// This sees elements in virtualized lists and iframes without scrolling
	// Fallback to querySelectorAll if CDP fails or is not available

	// Get CDP session for the page (like browser-use-reference)
	context := page.Context()
	cdpSession, err := context.NewCDPSession(page)
	if err == nil && cdpSession != nil {
		defer cdpSession.Detach()

		// Try to get accessibility tree via CDP (like browser-use-reference)
		result, cdpErr := cdpSession.Send("Accessibility.getFullAXTree", map[string]interface{}{})
		if cdpErr == nil && result != nil {
			// Parse accessibility tree and convert to Elements
			elems, parseErr := parseAccessibilityTree(result, limit)
			if parseErr == nil && len(elems) > 0 {
				// CDP worked, return elements
				// Log CDP success for debugging
				if resultMap, ok := result.(map[string]interface{}); ok {
					if nodes, ok := resultMap["nodes"].([]interface{}); ok {
						// Log CDP stats
						fmt.Printf("[CDP] Successfully parsed %d elements from %d CDP nodes\n", len(elems), len(nodes))
					}
				}
				return elems, nil
			}
			// If parsing failed, log and fall through to querySelectorAll
			if parseErr != nil {
				fmt.Printf("[CDP] Parse error: %v\n", parseErr)
			} else {
				fmt.Printf("[CDP] Parsed 0 elements (falling back to querySelectorAll)\n")
			}
		} else {
			// CDP failed
			if cdpErr != nil {
				fmt.Printf("[CDP] Error: %v\n", cdpErr)
			} else {
				fmt.Printf("[CDP] No result returned\n")
			}
		}
	} else {
		// CDP session creation failed
		if err != nil {
			fmt.Printf("[CDP] Failed to create session: %v (falling back to querySelectorAll)\n", err)
		}
	}

	// Fallback: Use querySelectorAll (fast but doesn't see virtualized lists without scrolling)
	script := `(limit) => {
		// Helper to check if element is scrollable (from browser-use pattern)
		function isScrollable(el) {
			if (!el) return false;
			const style = window.getComputedStyle(el);
			const overflowY = style.overflowY || style.overflow;
			const overflowX = style.overflowX || style.overflow;
			const allowsScroll = (overflowY === 'auto' || overflowY === 'scroll' || overflowY === 'overlay' ||
			                     overflowX === 'auto' || overflowX === 'scroll' || overflowX === 'overlay');
			return allowsScroll && (el.scrollHeight > el.clientHeight || el.scrollWidth > el.clientWidth);
		}
		
		// Helper to get scroll info text (from browser-use pattern)
		function getScrollInfo(el) {
			if (!isScrollable(el)) return "";
			const scrollTop = el.scrollTop || 0;
			const scrollHeight = el.scrollHeight || 0;
			const clientHeight = el.clientHeight || 0;
			if (scrollHeight <= clientHeight) return "";
			
			const contentAbove = Math.max(0, scrollTop);
			const contentBelow = Math.max(0, scrollHeight - clientHeight - scrollTop);
			const maxScrollTop = scrollHeight - clientHeight;
			const vPct = maxScrollTop > 0 ? Math.round((scrollTop / maxScrollTop) * 100) : 0;
			
			const pagesAbove = (contentAbove / clientHeight).toFixed(1);
			const pagesBelow = (contentBelow / clientHeight).toFixed(1);
			
			if (pagesAbove > 0 || pagesBelow > 0) {
				return pagesAbove + "↑ " + pagesBelow + "↓ " + vPct + "%";
			}
			return "";
		}
		
		// Helper to collect from shadow DOM
		function collectFromShadow(root, pick, limit) {
			if (!root || pick.length >= limit) return;
			try {
				// More aggressive selector: include common interactive patterns AND scrollable containers
				const nodes = root.querySelectorAll("a,button,input,select,textarea,[role],[tabindex],[data-testid],[data-qa],[data-qa-type],[onclick],div,section,main,article,aside");
				for (const el of nodes) {
					if (pick.length >= limit) break;
					const rect = el.getBoundingClientRect();
					if (rect.width === 0 && rect.height === 0) continue; // skip invisible
					
					// Check if element is scrollable (browser-use pattern)
					const scrollInfo = getScrollInfo(el);
					const isScrollableEl = scrollInfo !== "";
					
					// Skip non-interactive, non-scrollable elements
					const hasRole = el.getAttribute("role");
					const hasTabIndex = el.hasAttribute("tabindex");
					const isInteractive = el.tagName === "A" || el.tagName === "BUTTON" || el.tagName === "INPUT" || 
					                      el.tagName === "SELECT" || el.tagName === "TEXTAREA" || hasRole || hasTabIndex;
					if (!isInteractive && !isScrollableEl) continue;
					
					const bbox = [Math.round(rect.x), Math.round(rect.y), Math.round(rect.width), Math.round(rect.height)].join(",");
					const role = el.getAttribute("role") || el.tagName.toLowerCase();
					const attrs = ["name","aria-label","placeholder","type","value","role","tabindex","data-testid","data-qa","data-qa-type","title"].map(a => (a + ":" + (el.getAttribute(a) || ""))).join("|");
					// Get text content
					let text = (el.innerText || el.textContent || el.value || "").trim();
					text = text.slice(0, 120);
					
					// For scrollable containers, add scroll info to text
					if (isScrollableEl && !text) {
						text = "scrollable container";
					}
					
					if (!text && !attrs.trim() && !isScrollableEl) continue;
					
					// Build selector
					let sel = "";
					if (el.id) {
						sel = "#" + el.id;
					} else if (el.getAttribute("name")) {
						sel = "[name=\"" + el.getAttribute("name") + "\"]";
					} else {
						const testId = el.getAttribute("data-testid");
						const label = el.getAttribute("aria-label") || "";
						const name = el.getAttribute("name") || "";
						// Use only first line of text, remove newlines
						const textPart = text.split("\n")[0].slice(0,30).trim();
						// Sanitize: remove newlines, quotes, brackets, limit length
						let safe = (label || name || textPart).replace(/"/g, "").replace(/\[/g, "").replace(/\]/g, "").replace(/\n/g, " ").replace(/\r/g, " ").trim();
						if (safe.length > 40) safe = safe.slice(0, 40);
						
						if (testId && safe) {
							sel = "[data-testid=\"" + testId + "\"][aria-label*=\"" + safe + "\"]";
						} else if (testId) {
							const tag = el.tagName.toLowerCase();
							const siblings = Array.from(el.parentElement ? el.parentElement.children : []);
							const sameTestId = siblings.filter(c => c.getAttribute("data-testid") === testId);
							if (sameTestId.length > 1) {
								const idx = sameTestId.indexOf(el) + 1;
								sel = "[data-testid=\"" + testId + "\"]:nth-of-type(" + idx + ")";
							} else {
								sel = "[data-testid=\"" + testId + "\"]";
							}
						} else if (role && safe) {
							sel = "[role=\"" + role + "\"][aria-label*=\"" + safe + "\"]";
						} else if (role) {
							sel = "[role=\"" + role + "\"]";
						} else {
							const tag = el.tagName.toLowerCase();
							const siblings = Array.from(el.parentElement ? el.parentElement.children : []);
							const idx = siblings.filter(c => c.tagName === el.tagName).indexOf(el) + 1;
							if (idx > 0) sel = tag + ":nth-of-type(" + idx + ")";
						}
					}
					pick.push({role, text, attr: attrs, bbox, selector: sel, scrollInfo: scrollInfo});
					
					// Recurse into shadow DOM
					if (el.shadowRoot) {
						collectFromShadow(el.shadowRoot, pick, limit);
					}
				}
			} catch (e) {
				// ignore shadow DOM errors
			}
		}
		
		const pick = [];
		
		// Collect from main document
		collectFromShadow(document, pick, limit);
		
		// Collect from all iframes (more aggressive)
		const iframes = document.querySelectorAll("iframe");
		for (const iframe of iframes) {
			if (pick.length >= limit) break;
			try {
				// Try multiple ways to access iframe
				let iframeDoc = iframe.contentDocument;
				if (!iframeDoc) {
					iframeDoc = iframe.contentWindow?.document;
				}
				if (iframeDoc) {
					collectFromShadow(iframeDoc, pick, limit);
				}
			} catch (e) {
				// cross-origin iframe, skip
			}
		}
		
		return pick;
	}`
	// Collect from main frame
	val, err := page.Evaluate(script, limit)
	if err != nil {
		return nil, err
	}
	bytes, err := json.Marshal(val)
	if err != nil {
		return nil, err
	}
	var elems []Element
	if err := json.Unmarshal(bytes, &elems); err != nil {
		return nil, err
	}

	// Also collect from all iframes using Playwright API (more aggressive)
	frames := page.Frames()
	for _, frame := range frames {
		if len(elems) >= limit {
			break
		}
		// Skip main frame (already collected)
		if frame == page.MainFrame() {
			continue
		}
		// Try to collect from iframe
		iframeVal, iframeErr := frame.Evaluate(script, limit-len(elems))
		if iframeErr != nil {
			// Cross-origin iframe or error, skip
			continue
		}
		iframeBytes, err := json.Marshal(iframeVal)
		if err != nil {
			continue
		}
		var iframeElems []Element
		if err := json.Unmarshal(iframeBytes, &iframeElems); err != nil {
			continue
		}
		elems = append(elems, iframeElems...)
	}

	// Limit final result
	if len(elems) > limit {
		elems = elems[:limit]
	}

	return elems, nil
}

// parseAccessibilityTree parses CDP Accessibility.getFullAXTree response and converts to Elements
// This is like browser-use-reference approach - sees elements in virtualized lists and iframes
func parseAccessibilityTree(cdpResult interface{}, limit int) ([]Element, error) {
	// CDP returns accessibility tree with nodes
	// Each node has: role, name, value, description, boundingBox, etc.
	// We need to extract actionable elements (buttons, links, inputs, etc.)

	resultMap, ok := cdpResult.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid CDP result format")
	}

	nodes, ok := resultMap["nodes"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("no nodes in accessibility tree")
	}

	// Debug: log total nodes from CDP
	fmt.Printf("[CDP] Processing %d nodes from accessibility tree\n", len(nodes))

	// Step 1: Build node map and parent-child relationships
	// Map: nodeId -> node data
	nodeMap := make(map[string]map[string]interface{})
	// Map: nodeId -> parentId
	parentMap := make(map[string]string)
	// Map: nodeId -> childIds
	childMap := make(map[string][]string)

	// First pass: build node map and extract relationships
	for _, nodeInterface := range nodes {
		node, ok := nodeInterface.(map[string]interface{})
		if !ok {
			continue
		}

		// Get nodeId
		var nodeId string
		if id, ok := node["nodeId"]; ok {
			if idStr, ok := id.(string); ok {
				nodeId = idStr
			} else if idFloat, ok := id.(float64); ok {
				nodeId = fmt.Sprintf("%.0f", idFloat)
			}
		}
		if nodeId == "" {
			continue
		}

		nodeMap[nodeId] = node

		// Get childIds
		if childIds, ok := node["childIds"]; ok {
			if childIdsArr, ok := childIds.([]interface{}); ok {
				childIdStrs := make([]string, 0, len(childIdsArr))
				for _, childId := range childIdsArr {
					if childIdStr, ok := childId.(string); ok {
						childIdStrs = append(childIdStrs, childIdStr)
						parentMap[childIdStr] = nodeId
					} else if childIdFloat, ok := childId.(float64); ok {
						childIdStr := fmt.Sprintf("%.0f", childIdFloat)
						childIdStrs = append(childIdStrs, childIdStr)
						parentMap[childIdStr] = nodeId
					}
				}
				childMap[nodeId] = childIdStrs
			}
		}
	}

	// Step 2: Calculate depth for each node (BFS from root)
	depthMap := make(map[string]int)
	// Find root nodes (nodes without parents)
	var queue []string
	for nodeId := range nodeMap {
		if _, hasParent := parentMap[nodeId]; !hasParent {
			depthMap[nodeId] = 0
			queue = append(queue, nodeId)
		}
	}
	// BFS to calculate depths
	for len(queue) > 0 {
		currentId := queue[0]
		queue = queue[1:]
		currentDepth := depthMap[currentId]
		for _, childId := range childMap[currentId] {
			if _, visited := depthMap[childId]; !visited {
				depthMap[childId] = currentDepth + 1
				queue = append(queue, childId)
			}
		}
	}

	var elems []Element
	actionableRoles := map[string]bool{
		"button": true, "link": true, "textbox": true, "checkbox": true,
		"radio": true, "radiogroup": true, "combobox": true, "listitem": true, "menuitem": true,
		"tab": true, "option": true, "article": true, "row": true,
		"list": true, "listbox": true, "treeitem": true, "cell": true,
	}

	// Roles to skip (not actionable)
	skipRoles := map[string]bool{
		"text": true, "statictext": true, "inlineTextBox": true,
		"lineBreak": true, "paragraph": true,
	}

	processedCount := 0
	skippedCount := 0
	actionableCount := 0
	noBboxCount := 0
	noTextCount := 0

	// Step 3: Process nodes and build elements with hierarchy info
	for _, nodeInterface := range nodes {
		if len(elems) >= limit {
			break
		}

		node, ok := nodeInterface.(map[string]interface{})
		if !ok {
			continue
		}

		processedCount++

		// Get role - CDP structure: role is an object with "type" field
		roleValue, ok := node["role"]
		if !ok {
			skippedCount++
			continue
		}

		var roleType string
		// CDP returns role as object: {type: "role", value: "link"} or {type: "internalRole", value: "RootWebArea"}
		// The ACTUAL role is in the "value" field, not "type"!
		if roleMap, ok := roleValue.(map[string]interface{}); ok {
			// CRITICAL FIX: Role is in "value" field, not "type"!
			if rt, ok := roleMap["value"].(string); ok && rt != "" {
				roleType = rt
			} else if rt, ok := roleMap["type"].(string); ok && rt != "role" && rt != "internalRole" {
				// Fallback: sometimes role might be directly in type (but this is rare)
				roleType = rt
			}
		} else if rt, ok := roleValue.(string); ok {
			// Fallback: sometimes role might be string directly
			roleType = rt
		}

		// Skip if no valid role or if it's a role we want to skip
		if roleType == "" || skipRoles[roleType] {
			skippedCount++
			continue
		}

		// Track if this is an actionable role
		if actionableRoles[roleType] {
			actionableCount++
		}

		// Get name - CDP structure: name is an object with "value" field
		nameValue := ""
		if name, ok := node["name"]; ok {
			if nameMap, ok := name.(map[string]interface{}); ok {
				if nv, ok := nameMap["value"].(string); ok {
					nameValue = nv
				}
			} else if nv, ok := name.(string); ok {
				// Fallback: sometimes name might be string directly
				nameValue = nv
			}
		}

		// Get bounding box
		// CDP boundingBox can be null for virtualized elements (outside viewport)
		// This is OK - we still want to include actionable elements even without bbox
		var bboxStr string
		if bbox, ok := node["boundingBox"].(map[string]interface{}); ok && bbox != nil {
			if x, ok := bbox["x"].(float64); ok {
				y, _ := bbox["y"].(float64)
				width, _ := bbox["width"].(float64)
				height, _ := bbox["height"].(float64)
				// Only set bbox if all values are valid (not 0,0,0,0)
				if x != 0 || y != 0 || width != 0 || height != 0 {
					bboxStr = fmt.Sprintf("%.0f,%.0f,%.0f,%.0f", x, y, width, height)
				}
			}
		}

		// Get value if available
		valueStr := ""
		if value, ok := node["value"].(map[string]interface{}); ok {
			if vv, ok := value["value"].(string); ok {
				valueStr = vv
			}
		} else if vv, ok := node["value"].(string); ok {
			valueStr = vv
		}

		text := nameValue
		if valueStr != "" && text == "" {
			text = valueStr
		}
		if len(text) > 120 {
			text = text[:120]
		}

		// Build attributes
		attrs := []string{}
		if nameValue != "" {
			attrs = append(attrs, "name:"+nameValue)
		}
		if valueStr != "" {
			attrs = append(attrs, "value:"+valueStr)
		}
		attrStr := strings.Join(attrs, "|")

		// Build selector - need to match elements by role and text/name
		// CDP doesn't give us DOM selectors, so we build role-based selector with text matching
		// For textbox, also try input[type] selectors for better matching
		sel := ""
		if roleType != "" {
			// For textbox, try to build better selector using input type
			if roleType == "textbox" {
				// Try to get input type from properties or attributes
				inputType := "text" // default
				if props, ok := node["properties"].([]interface{}); ok {
					for _, prop := range props {
						if propMap, ok := prop.(map[string]interface{}); ok {
							if propName, ok := propMap["name"].(string); ok && propName == "inputType" {
								if propValue, ok := propMap["value"].(map[string]interface{}); ok {
									if typeVal, ok := propValue["value"].(string); ok {
										inputType = typeVal
									}
								}
							}
						}
					}
				}
				// Build selector: prefer input[type] over [role="textbox"] for better matching
				if nameValue != "" && len(nameValue) < 50 {
					safeName := strings.ReplaceAll(nameValue, "\"", "'")
					safeName = strings.ReplaceAll(safeName, "\n", " ")
					if len(safeName) > 40 {
						safeName = safeName[:40]
					}
					sel = fmt.Sprintf("input[type=\"%s\"][aria-label*=\"%s\"], [role=\"textbox\"][aria-label*=\"%s\"]", inputType, safeName, safeName)
				} else {
					sel = fmt.Sprintf("input[type=\"%s\"], [role=\"textbox\"]", inputType)
				}
			} else if nameValue != "" && len(nameValue) < 50 {
				// Use role with aria-label matching name
				safeName := strings.ReplaceAll(nameValue, "\"", "'")
				safeName = strings.ReplaceAll(safeName, "\n", " ")
				if len(safeName) > 40 {
					safeName = safeName[:40]
				}
				sel = fmt.Sprintf("[role=\"%s\"][aria-label*=\"%s\"]", roleType, safeName)
			} else {
				sel = fmt.Sprintf("[role=\"%s\"]", roleType)
			}
		}

		// Track statistics
		if bboxStr == "" {
			noBboxCount++
		}
		if text == "" {
			noTextCount++
		}

		// Get nodeId and parentId for hierarchy
		var nodeId string
		var parentId string
		var depth int
		if id, ok := node["nodeId"]; ok {
			if idStr, ok := id.(string); ok {
				nodeId = idStr
			} else if idFloat, ok := id.(float64); ok {
				nodeId = fmt.Sprintf("%.0f", idFloat)
			}
		}
		if nodeId != "" {
			if pid, ok := parentMap[nodeId]; ok {
				parentId = pid
			}
			if d, ok := depthMap[nodeId]; ok {
				depth = d
			}
		}

		// Check if this is an actionable role
		isActionableRole := actionableRoles[roleType]
		hasText := text != ""
		hasBbox := bboxStr != ""

		// CRITICAL FIX: Include actionable roles even without bbox/text
		// CDP sees virtualized elements that may not have bbox (outside viewport)
		// or text (containers). We must include them to see content in virtualized lists.
		if isActionableRole {
			// Always include actionable roles - CDP sees virtualized content
			elems = append(elems, Element{
				Role:     roleType,
				Text:     text,
				Attr:     attrStr,
				BBox:     bboxStr,
				Sel:      sel,
				Depth:    depth,
				NodeId:   nodeId,
				ParentId: parentId,
			})
		} else if hasText || hasBbox {
			// Include non-actionable elements only if they have text or bbox
			elems = append(elems, Element{
				Role:     roleType,
				Text:     text,
				Attr:     attrStr,
				BBox:     bboxStr,
				Sel:      sel,
				Depth:    depth,
				NodeId:   nodeId,
				ParentId: parentId,
			})
		} else {
			// Skip elements with no actionable role, no text, and no bbox
			skippedCount++
		}
	}

	// Debug: log parsing stats
	fmt.Printf("[CDP] Parsed %d actionable elements (processed: %d, skipped: %d, actionable roles: %d, noBbox: %d, noText: %d)\n",
		len(elems), processedCount, skippedCount, actionableCount, noBboxCount, noTextCount)

	return elems, nil
}

// WithDeadline shortens context to avoid long snapshot waits.
func WithDeadline(ctx context.Context, dur time.Duration) (context.Context, context.CancelFunc) {
	if dur <= 0 {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, dur)
}

// filterAndRankElements filters and ranks elements by relevance
func filterAndRankElements(elems []Element, maxCount int) []Element {
	if len(elems) <= maxCount {
		return elems
	}

	// Score elements by relevance
	type scoredElement struct {
		element Element
		score   int
	}

	scored := make([]scoredElement, 0, len(elems))
	for _, el := range elems {
		score := scoreElement(el)
		// Filter out completely irrelevant elements (score 0)
		if score > 0 {
			scored = append(scored, scoredElement{element: el, score: score})
		}
	}

	// Sort by score (highest first)
	for i := 0; i < len(scored)-1; i++ {
		for j := i + 1; j < len(scored); j++ {
			if scored[i].score < scored[j].score {
				scored[i], scored[j] = scored[j], scored[i]
			}
		}
	}

	// Take top N
	result := make([]Element, 0, maxCount)
	for i := 0; i < len(scored) && i < maxCount; i++ {
		result = append(result, scored[i].element)
	}

	return result
}

// scoreElement calculates relevance score for an element
func scoreElement(el Element) int {
	score := 0
	textLower := strings.ToLower(el.Text)
	attrLower := strings.ToLower(el.Attr)

	// High score: interactive elements with meaningful text
	if el.Role != "" && el.Role != "generic" && el.Role != "presentation" {
		score += 5
	}

	// Text content
	if len(el.Text) > 0 {
		score += 3
		if len(el.Text) > 10 && len(el.Text) < 200 {
			score += 2 // Good length
		}
	}

	// Common interactive indicators (not site-specific)
	if strings.Contains(textLower, "@") {
		score += 3 // Content with @ symbol (could be email, mention, etc.)
	}
	if strings.Contains(attrLower, "data-testid") {
		score += 3
	}
	if strings.Contains(attrLower, "aria-label") {
		score += 2
	}

	// Penalty: empty or very short text
	if len(el.Text) == 0 && el.Role == "" {
		score -= 5
	}

	// Penalty: very long text (likely not interactive)
	if len(el.Text) > 500 {
		score -= 3
	}

	return score
}
