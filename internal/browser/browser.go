package browser

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/playwright-community/playwright-go"
)

const (
	defaultNavTimeout   = 30 * time.Second
	defaultActionTime   = 10 * time.Second
	headlessEnv         = "AGENT_HEADLESS"
	defaultScrollAmount = 600
)

// Controller exposes minimal browser actions to the agent.
type Controller interface {
	Close(ctx context.Context) error
	Navigate(ctx context.Context, url string) error
	GoBack(ctx context.Context) error
	ClickText(ctx context.Context, text string, exact bool) error
	ClickRole(ctx context.Context, role, name string, exact bool) error
	Click(ctx context.Context, selector string) error
	ClickByCoordinates(ctx context.Context, x, y float64) error
	ClickByTextFuzzy(ctx context.Context, text string) error
	Fill(ctx context.Context, selector, text string) error
	Read(ctx context.Context, selector string) (string, error)
	Scroll(ctx context.Context, direction string, distance int) (int, error)
	ScrollToElement(ctx context.Context, selector string) error
	WaitFor(ctx context.Context, selector string, timeout time.Duration) error
	WaitForLazyListItems(ctx context.Context, timeout time.Duration) error
	WaitForStableDOM(ctx context.Context, timeout time.Duration) error
	SaveState(ctx context.Context, path string) error
	Hover(ctx context.Context, selector string) error // Hover over element to reveal hidden elements
	Page() playwright.Page
}

// Launcher owns playwright lifecycle.
type Launcher struct {
	pw       *playwright.Playwright
	browser  playwright.Browser
	headless bool
}

func NewLauncher(ctx context.Context) (*Launcher, error) {
	if err := ensureDeps(); err != nil {
		return nil, err
	}
	pw, err := playwright.Run()
	if err != nil {
		return nil, fmt.Errorf("start playwright: %w", err)
	}
	headless := parseBoolEnv(headlessEnv, false)
	browser, err := pw.Chromium.Launch(playwright.BrowserTypeLaunchOptions{
		Headless: playwright.Bool(headless),
		Args: []string{
			"--disable-dev-shm-usage",
			"--no-sandbox",
		},
	})
	if err != nil {
		_ = pw.Stop()
		return nil, fmt.Errorf("launch chromium: %w", err)
	}
	return &Launcher{pw: pw, browser: browser, headless: headless}, nil
}

func (l *Launcher) NewController(ctx context.Context, storagePath string) (Controller, error) {
	opts := playwright.BrowserNewContextOptions{
		IgnoreHttpsErrors: playwright.Bool(true),
	}
	hasStorageState := false
	if strings.TrimSpace(storagePath) != "" {
		// Check if storage state file exists
		if _, err := os.Stat(storagePath); err == nil {
			opts.StorageStatePath = playwright.String(storagePath)
			hasStorageState = true
		}
	}
	context, err := l.browser.NewContext(opts)
	if err != nil {
		return nil, fmt.Errorf("new context: %w", err)
	}
	page, err := context.NewPage()
	if err != nil {
		_ = context.Close()
		return nil, fmt.Errorf("new page: %w", err)
	}
	page.SetDefaultTimeout(float64(defaultNavTimeout.Milliseconds()))

	// If storage state was loaded, page might be on about:blank
	// This is normal - agent will navigate to the site and cookies will be applied
	ctrl := &controller{context: context, page: page, hasStorageState: hasStorageState}
	return ctrl, nil
}

func (l *Launcher) Close() error {
	if l.browser != nil {
		_ = l.browser.Close()
	}
	if l.pw != nil {
		return l.pw.Stop()
	}
	return nil
}

type controller struct {
	context         playwright.BrowserContext
	page            playwright.Page
	hasStorageState bool // Track if storage state was loaded
}

func (c *controller) Page() playwright.Page {
	return c.page
}

func (c *controller) Close(ctx context.Context) error {
	_ = ctx
	if c.page != nil {
		_ = c.page.Close()
	}
	if c.context != nil {
		return c.context.Close()
	}
	return nil
}

func (c *controller) Navigate(ctx context.Context, url string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	// When navigating with storage state, cookies from storage state are automatically applied
	// by Playwright when navigating to the domain
	_, err := c.page.Goto(url, playwright.PageGotoOptions{
		WaitUntil: playwright.WaitUntilStateLoad,
		Timeout:   playwright.Float(float64(defaultNavTimeout.Milliseconds())),
	})
	return wrap(err)
}

func (c *controller) GoBack(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	_, err := c.page.GoBack()
	return wrap(err)
}

func (c *controller) ClickText(ctx context.Context, text string, exact bool) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	loc := c.page.GetByText(text, playwright.PageGetByTextOptions{
		Exact: playwright.Bool(exact),
	})
	first := loc.First()
	if err := first.WaitFor(playwright.LocatorWaitForOptions{State: playwright.WaitForSelectorStateVisible}); err != nil {
		return wrap(err)
	}
	return wrap(first.Click())
}

func (c *controller) ClickRole(ctx context.Context, role, name string, exact bool) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	aria := playwright.AriaRole(strings.ToLower(strings.TrimSpace(role)))
	loc := c.page.GetByRole(aria, playwright.PageGetByRoleOptions{
		Name:  name,
		Exact: playwright.Bool(exact),
	})
	first := loc.First()
	// Use 15s timeout - balance between reliability and speed
	if err := first.WaitFor(playwright.LocatorWaitForOptions{
		State:   playwright.WaitForSelectorStateVisible,
		Timeout: playwright.Float(15000), // 15s timeout
	}); err != nil {
		return wrap(err)
	}
	return wrap(first.Click())
}

func (c *controller) Click(ctx context.Context, selector string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	loc := c.page.Locator(selector)
	// Use First() to avoid strict mode violation when multiple elements match
	first := loc.First()
	if err := first.WaitFor(playwright.LocatorWaitForOptions{State: playwright.WaitForSelectorStateVisible}); err != nil {
		return wrap(err)
	}
	// Scroll element into view before clicking
	if err := first.ScrollIntoViewIfNeeded(); err != nil {
		// If scroll fails, try click anyway
	}
	// Use Click with HasText option if possible to be more specific, but fallback to First()
	return wrap(first.Click())
}

// ClickByCoordinates clicks at specific coordinates (fallback when selector fails)
func (c *controller) ClickByCoordinates(ctx context.Context, x, y float64) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	err := c.page.Mouse().Click(x, y)
	return wrap(err)
}

// ClickByTextFuzzy finds element by partial text match and clicks it
func (c *controller) ClickByTextFuzzy(ctx context.Context, text string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	// Try exact match first
	loc := c.page.GetByText(text, playwright.PageGetByTextOptions{
		Exact: playwright.Bool(false), // Fuzzy match
	})
	first := loc.First()
	if err := first.WaitFor(playwright.LocatorWaitForOptions{
		State: playwright.WaitForSelectorStateVisible,
	}); err != nil {
		return wrap(err)
	}
	if err := first.ScrollIntoViewIfNeeded(); err != nil {
		// Continue anyway
	}
	return wrap(first.Click())
}

// ScrollToElement scrolls element into view before interaction
func (c *controller) ScrollToElement(ctx context.Context, selector string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	loc := c.page.Locator(selector)
	first := loc.First()
	return wrap(first.ScrollIntoViewIfNeeded())
}

// Hover hovers over element to reveal hidden elements (useful for dynamic content)
func (c *controller) Hover(ctx context.Context, selector string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	loc := c.page.Locator(selector)
	first := loc.First()
	if err := first.WaitFor(playwright.LocatorWaitForOptions{State: playwright.WaitForSelectorStateVisible}); err != nil {
		return wrap(err)
	}
	return wrap(first.Hover())
}

// WaitForLazyListItems waits for lazy-loaded list items to appear (universal solution)
func (c *controller) WaitForLazyListItems(ctx context.Context, timeout time.Duration) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if timeout <= 0 {
		timeout = 10 * time.Second
	}

	// Wait for common list item patterns (universal, not site-specific)
	patterns := []string{
		"[role='row']",
		"[role='listitem']",
		"[role='option']",
		"li[data-*]",
		"div[data-*][role]",
	}

	deadline := time.Now().Add(timeout)
	for _, pattern := range patterns {
		if time.Now().After(deadline) {
			break
		}
		loc := c.page.Locator(pattern)
		first := loc.First()
		if err := first.WaitFor(playwright.LocatorWaitForOptions{
			State:   playwright.WaitForSelectorStateVisible,
			Timeout: playwright.Float(timeout.Seconds() * 1000 / float64(len(patterns))),
		}); err == nil {
			// Found at least one list item
			return nil
		}
	}

	// Also check all frames
	frames := c.page.Frames()
	for _, frame := range frames {
		if time.Now().After(deadline) {
			break
		}
		for _, pattern := range patterns {
			loc := frame.Locator(pattern)
			first := loc.First()
			if err := first.WaitFor(playwright.LocatorWaitForOptions{
				State:   playwright.WaitForSelectorStateVisible,
				Timeout: playwright.Float(2000),
			}); err == nil {
				return nil
			}
		}
	}

	// Fallback: search by text content for list-like structures
	fallbackScript := `(limit) => {
		const out = [];
		function scan(root) {
			const nodes = root.querySelectorAll("div, li, span, a, [role='option'], [role='listitem'], [role='row']");
			for (const n of nodes) {
				try {
					const t = (n.innerText || n.textContent || "").trim();
					if (t && t.length > 10 && t.length < 500) {
						out.push(t.slice(0,200));
						if (out.length >= limit) return;
					}
				} catch(e){}
			}
		}
		scan(document);
		const iframes = document.querySelectorAll("iframe");
		for (const iframe of iframes) {
			try {
				const doc = iframe.contentDocument || (iframe.contentWindow && iframe.contentWindow.document);
				if (doc) scan(doc);
			} catch(e){}
			if (out.length >= limit) break;
		}
		return out;
	}`

	val, err := c.page.Evaluate(fallbackScript, 3)
	if err == nil {
		if arr, ok := val.([]interface{}); ok && len(arr) > 0 {
			// Found list-like content
			return nil
		}
	}

	return fmt.Errorf("no list items found after %v", timeout)
}

func (c *controller) Fill(ctx context.Context, selector, text string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	loc := c.page.Locator(selector)
	if err := loc.WaitFor(playwright.LocatorWaitForOptions{State: playwright.WaitForSelectorStateVisible}); err != nil {
		return wrap(err)
	}
	if err := loc.Fill(text); err != nil {
		return wrap(err)
	}
	return nil
}

func (c *controller) Read(ctx context.Context, selector string) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}

	// Improved: try all frames (including iframes) for better iframe support
	frames := c.page.Frames()

	if strings.TrimSpace(selector) == "" {
		// Try main frame first
		val, err := c.page.InnerText("body")
		if err == nil && strings.TrimSpace(val) != "" {
			return val, nil
		}
		// Try all frames
		for _, frame := range frames {
			if frame == c.page.MainFrame() {
				continue
			}
			iframeVal, iframeErr := frame.InnerText("body")
			if iframeErr == nil && strings.TrimSpace(iframeVal) != "" {
				return iframeVal, nil
			}
		}
		return val, nil // Return main frame result even if empty
	}

	// Try main frame first
	loc := c.page.Locator(selector)
	if err := loc.WaitFor(playwright.LocatorWaitForOptions{
		State:   playwright.WaitForSelectorStateVisible,
		Timeout: playwright.Float(5000), // 5s timeout
	}); err == nil {
		val, err := loc.InnerText()
		if err == nil && strings.TrimSpace(val) != "" {
			return val, nil
		}
	}

	// Try all frames if main frame failed
	for _, frame := range frames {
		if frame == c.page.MainFrame() {
			continue
		}
		iframeLoc := frame.Locator(selector)
		if err := iframeLoc.WaitFor(playwright.LocatorWaitForOptions{
			State:   playwright.WaitForSelectorStateVisible,
			Timeout: playwright.Float(3000), // Shorter timeout for iframes
		}); err == nil {
			val, err := iframeLoc.InnerText()
			if err == nil && strings.TrimSpace(val) != "" {
				return val, nil
			}
		}
	}

	// Return error if nothing found
	return "", fmt.Errorf("selector not found in any frame: %s", selector)
}

func (c *controller) Scroll(ctx context.Context, direction string, distance int) (int, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}

	// Get viewport height for more accurate scrolling (like browser-use)
	viewportHeight := defaultScrollAmount
	viewportScript := `() => {
		return Math.max(
			window.innerHeight || 0,
			document.documentElement.clientHeight || 0,
			document.body.clientHeight || 0,
			600 // Fallback
		);
	}`
	if vh, err := c.page.Evaluate(viewportScript); err == nil {
		if vhNum, ok := vh.(float64); ok && vhNum > 0 {
			viewportHeight = int(vhNum)
		}
	}

	// Use viewport height as default if distance is 0 or not provided
	if distance == 0 {
		distance = viewportHeight
	}

	// Improved scroll: find scrollable container (ChatGPT recommendation)
	// Prefer focused element's ancestor, then common containers
	// This is critical for SPAs that use internal scroll containers
	script := `(dir, dist) => {
		function isScrollable(el) {
			if (!el) return false;
			const s = window.getComputedStyle(el);
			return (s.overflowY === 'auto' || s.overflowY === 'scroll' || s.overflow === 'auto' || s.overflow === 'scroll') 
				   && el.scrollHeight > el.clientHeight;
		}

		const distance = Number(dist) || 600;
		const candidates = [];

		// Heuristic: prefer focused element's ancestor (ChatGPT recommendation)
		const active = document.activeElement;
		if (active) {
			let p = active;
			while (p) {
				if (isScrollable(p)) { 
					candidates.push(p); 
					break; 
				}
				p = p.parentElement;
			}
		}

		// Search for common scroll containers
		const nodes = document.querySelectorAll('div,section,main,[role="main"],aside');
		for (const n of nodes) {
			if (isScrollable(n)) candidates.push(n);
		}

		// Also include document.scrollingElement
		if (document.scrollingElement) candidates.push(document.scrollingElement);

		// Remove duplicates
		const seen = new Set();
		const final = [];
		for (const el of candidates) {
			if (!el) continue;
			const id = el.tagName + '|' + (el.id||'') + '|' + (el.className||'');
			if (!seen.has(id)) { 
				seen.add(id); 
				final.push(el); 
			}
		}

		// Fix: handle undefined/null direction (critical bug fix)
		if (!dir || typeof dir !== 'string') {
			dir = 'down'; // Default to down if direction is missing
		}
		const dirLower = dir.toLowerCase();
		let move = distance;
		if (dirLower === 'up' || dirLower === 'page_up' || dirLower === 'top') {
			move = -distance;
			if (dirLower === 'page_up') move *= 2;
		} else if (dirLower === 'page_down') {
			move = distance * 2;
		} else if (dirLower === 'top') {
			if (final.length > 0) {
				final[0].scrollTop = 0;
				return {scrolled: true, scrollTop: final[0].scrollTop};
			}
			window.scrollTo(0, 0);
			return {scrolled: false};
		} else if (dirLower === 'bottom') {
			if (final.length > 0) {
				final[0].scrollTop = final[0].scrollHeight;
				return {scrolled: true, scrollTop: final[0].scrollTop};
			}
			window.scrollTo(0, document.body.scrollHeight);
			return {scrolled: false};
		}

		// Try scrolling first candidate; if none, fallback to window.scrollBy
		if (final.length > 0) {
			final[0].scrollBy({top: move, left: 0, behavior: 'auto'});
			return {scrolled: true, scrollTop: final[0].scrollTop};
		}
		
		window.scrollBy(0, move);
		return {scrolled: false, distance: distance};
	}`

	_, err := c.page.Evaluate(script, direction, distance)
	if err != nil {
		return 0, wrap(err)
	}

	// Return the actual distance used
	return distance, nil
}

func (c *controller) WaitFor(ctx context.Context, selector string, timeout time.Duration) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if timeout <= 0 {
		timeout = defaultActionTime
	}
	loc := c.page.Locator(selector)
	return wrap(loc.WaitFor(playwright.LocatorWaitForOptions{
		Timeout: playwright.Float(timeout.Seconds() * 1000),
		State:   playwright.WaitForSelectorStateVisible,
	}))
}

// WaitForStableDOM waits for DOM to stabilize (no mutations for a period)
// This replaces fixed sleep() calls with event-driven waiting
func (c *controller) WaitForStableDOM(ctx context.Context, timeout time.Duration) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if timeout <= 0 {
		timeout = 2 * time.Second
	}

	// Use Playwright's wait for load state + network idle
	// This is more efficient than fixed sleep
	if err := c.page.WaitForLoadState(playwright.PageWaitForLoadStateOptions{
		State:   playwright.LoadStateNetworkidle,
		Timeout: playwright.Float(float64(timeout.Milliseconds())),
	}); err != nil {
		// If network idle fails, try DOMContentLoaded as fallback
		_ = c.page.WaitForLoadState(playwright.PageWaitForLoadStateOptions{
			State:   playwright.LoadStateDomcontentloaded,
			Timeout: playwright.Float(1000),
		})
	}

	// Additional: wait for no DOM mutations for 300ms (like MutationObserver)
	script := `
		() => {
			return new Promise((resolve) => {
				let timeoutId;
				const observer = new MutationObserver(() => {
					clearTimeout(timeoutId);
					timeoutId = setTimeout(() => {
						observer.disconnect();
						resolve();
					}, 300);
				});
				observer.observe(document.body, {
					childList: true,
					subtree: true,
					attributes: true,
					attributeOldValue: false
				});
				timeoutId = setTimeout(() => {
					observer.disconnect();
					resolve();
				}, 300);
			});
		}
	`
	_, err := c.page.Evaluate(script)
	return wrap(err)
}

func (c *controller) SaveState(ctx context.Context, path string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	state, err := c.context.StorageState()
	if err != nil {
		return wrap(err)
	}
	data, err := json.Marshal(state)
	if err != nil {
		return fmt.Errorf("marshal storage: %w", err)
	}
	return os.WriteFile(path, data, 0o600)
}

func wrap(err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("playwright: %w", err)
}

func parseBoolEnv(name string, def bool) bool {
	val := strings.TrimSpace(os.Getenv(name))
	if val == "" {
		return def
	}
	switch strings.ToLower(val) {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return def
	}
}

func ensureDeps() error {
	// Browsers usually preinstalled in this workspace. Hook for future checks.
	return nil
}
