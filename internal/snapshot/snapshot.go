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
	Role string `json:"role"`
	Text string `json:"text"`
	Attr string `json:"attr"`
	BBox string `json:"bbox"`
	Sel  string `json:"selector"`
}

// Summary is a compact view of current page.
type Summary struct {
	URL      string
	Title    string
	Visible  string
	Elements []Element
}

// ToMap returns summary as a JSON-friendly map.
func (s Summary) ToMap() map[string]any {
	return map[string]any{
		"url":      s.URL,
		"title":    s.Title,
		"visible":  s.Visible,
		"elements": s.Elements,
	}
}

func Collect(ctx context.Context, ctrl browser.Controller) (Summary, error) {
	page := ctrl.Page()
	title, _ := page.Title()
	url := page.URL()

	text, _ := page.InnerText("body")
	if len(text) > 1200 {
		text = text[:1200]
	}

	elems, _ := collectInteractive(ctx, page, 300)

	// Snapshot Filtering Layer: filter and rank elements
	filteredElems := filterAndRankElements(elems, 150) // Top 150 most relevant

	return Summary{
		URL:      url,
		Title:    title,
		Visible:  strings.TrimSpace(text),
		Elements: filteredElems,
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
	script := `(limit) => {
		// Helper to collect from shadow DOM
		function collectFromShadow(root, pick, limit) {
			if (!root || pick.length >= limit) return;
			try {
				// More aggressive selector: include email-specific patterns
				const nodes = root.querySelectorAll("a,button,input,select,textarea,[role],[tabindex],[data-testid],[data-qa],[data-qa-type],[onclick],[data-uid],[data-subject],[data-sender]");
				for (const el of nodes) {
					if (pick.length >= limit) break;
					const rect = el.getBoundingClientRect();
					if (rect.width === 0 && rect.height === 0) continue; // skip invisible
					const bbox = [Math.round(rect.x), Math.round(rect.y), Math.round(rect.width), Math.round(rect.height)].join(",");
					const role = el.getAttribute("role") || el.tagName.toLowerCase();
					const attrs = ["name","aria-label","placeholder","type","value","role","tabindex","data-testid","data-qa","data-qa-type","title"].map(a => (a + ":" + (el.getAttribute(a) || ""))).join("|");
					// Get more text: include subject, sender, date if available
					let text = (el.innerText || el.textContent || el.value || "").trim();
					// Try to get email-specific info
					const subject = el.closest("[data-subject]")?.getAttribute("data-subject");
					const sender = el.closest("[data-sender]")?.getAttribute("data-sender");
					if (subject) text = (subject + " " + text).trim();
					if (sender) text = (sender + " " + text).trim();
					text = text.slice(0, 120); // increased from 80
					if (!text && !attrs.trim()) continue;
					
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
					pick.push({role, text, attr: attrs, bbox, selector: sel});
					
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
					// Also try to find email-specific containers in iframe
					const emailContainers = iframeDoc.querySelectorAll("[data-testid*='message'], [data-testid*='mail'], [role='row'][aria-label*='@']");
					for (const container of emailContainers) {
						if (pick.length >= limit) break;
						try {
							const rect = container.getBoundingClientRect();
							if (rect.width > 0 && rect.height > 0) {
								const bbox = [Math.round(rect.x), Math.round(rect.y), Math.round(rect.width), Math.round(rect.height)].join(",");
								const role = container.getAttribute("role") || container.tagName.toLowerCase();
								const text = (container.innerText || container.textContent || "").trim().slice(0, 120);
								const attrs = ["name","aria-label","data-testid","data-uid"].map(a => (a + ":" + (container.getAttribute(a) || ""))).join("|");
								let sel = "";
								if (container.id) {
									sel = "#" + container.id;
								} else if (container.getAttribute("data-testid")) {
									sel = "[data-testid=\"" + container.getAttribute("data-testid") + "\"]";
								}
								if (text || attrs) {
									pick.push({role, text, attr: attrs, bbox, selector: sel});
								}
							}
						} catch (e) {
							// skip individual element errors
						}
					}
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

	// Email-specific indicators
	if strings.Contains(textLower, "@") {
		score += 10
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
