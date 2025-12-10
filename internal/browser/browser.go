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
	ClickText(ctx context.Context, text string, exact bool) error
	ClickRole(ctx context.Context, role, name string, exact bool) error
	Click(ctx context.Context, selector string) error
	ClickByCoordinates(ctx context.Context, x, y float64) error
	ClickByTextFuzzy(ctx context.Context, text string) error
	Fill(ctx context.Context, selector, text string) error
	Read(ctx context.Context, selector string) (string, error)
	Scroll(ctx context.Context, direction string, distance int) error
	ScrollToElement(ctx context.Context, selector string) error
	WaitFor(ctx context.Context, selector string, timeout time.Duration) error
	WaitForEmailElements(ctx context.Context, timeout time.Duration) error
	SaveState(ctx context.Context, path string) error
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
	if strings.TrimSpace(storagePath) != "" {
		opts.StorageStatePath = playwright.String(storagePath)
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
	return &controller{context: context, page: page}, nil
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
	context playwright.BrowserContext
	page    playwright.Page
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
	_, err := c.page.Goto(url, playwright.PageGotoOptions{
		WaitUntil: playwright.WaitUntilStateLoad,
		Timeout:   playwright.Float(float64(defaultNavTimeout.Milliseconds())),
	})
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
	if err := first.WaitFor(playwright.LocatorWaitForOptions{State: playwright.WaitForSelectorStateVisible}); err != nil {
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
		State:   playwright.WaitForSelectorStateVisible,
		Timeout: playwright.Float(5000), // Shorter timeout for fuzzy
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

// WaitForEmailElements waits for email-like elements to appear (non-trivial solution)
func (c *controller) WaitForEmailElements(ctx context.Context, timeout time.Duration) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if timeout <= 0 {
		timeout = 10 * time.Second
	}

	// Wait for common email patterns
	patterns := []string{
		"[data-testid*='message']",
		"[data-testid*='mail']",
		"[data-testid*='item'][role='row']",
		"[role='row'][aria-label*='@']",
		"[data-uid]", // Yandex Mail specific
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
			// Found at least one email element
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

	return fmt.Errorf("no email elements found after %v", timeout)
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
	if strings.TrimSpace(selector) == "" {
		val, err := c.page.InnerText("body")
		if err != nil {
			return "", wrap(err)
		}
		return val, nil
	}
	loc := c.page.Locator(selector)
	if err := loc.WaitFor(playwright.LocatorWaitForOptions{State: playwright.WaitForSelectorStateVisible}); err != nil {
		return "", wrap(err)
	}
	val, err := loc.InnerText()
	return val, wrap(err)
}

func (c *controller) Scroll(ctx context.Context, direction string, distance int) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if distance == 0 {
		distance = defaultScrollAmount
	}
	move := distance
	switch strings.ToLower(direction) {
	case "up", "north":
		move = -distance
	case "top":
		_, err := c.page.Evaluate("window.scrollTo(0,0);")
		return wrap(err)
	case "bottom":
		_, err := c.page.Evaluate("window.scrollTo(0, document.body.scrollHeight);")
		return wrap(err)
	case "page_down":
		move = distance * 2
	case "page_up":
		move = -distance * 2
	}
	script := fmt.Sprintf("window.scrollBy(0,%d);", move)
	_, err := c.page.Evaluate(script)
	return wrap(err)
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
