package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/joho/godotenv"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/polzovatel/ai-agent-for-browser-fast/internal/agent"
	"github.com/polzovatel/ai-agent-for-browser-fast/internal/browser"
	"github.com/polzovatel/ai-agent-for-browser-fast/internal/llm"
	"github.com/polzovatel/ai-agent-for-browser-fast/internal/snapshot"
	"github.com/polzovatel/ai-agent-for-browser-fast/internal/tools"
)

type cliOptions struct {
	task        string
	storage     string
	saveState   string
	maxSteps    int
	temperature float64
}

func main() {
	_ = godotenv.Load()
	opts := parseFlags()
	if opts.task == "" {
		task, cancelled, err := promptTask()
		if err != nil {
			log.Fatal().Err(err).Msg("prompt task failed")
		}
		if cancelled {
			fmt.Println("Отменено.")
			return
		}
		opts.task = task
	}

	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	llmClient, err := llm.NewAnthropicWithLogger(log.With().Str("comp", "llm").Logger())
	if err != nil {
		log.Fatal().Err(err).Msg("llm init")
	}

	launcher, err := browser.NewLauncher(ctx)
	if err != nil {
		log.Fatal().Err(err).Msg("browser init")
	}
	defer launcher.Close()

	ctrl, err := launcher.NewController(ctx, opts.storage)
	if err != nil {
		log.Fatal().Err(err).Msg("browser controller")
	}
	defer ctrl.Close(ctx)

	toolbox := tools.New(ctrl, terminalPrompt())
	planner := agent.NewPlanner(llmClient)

	// Create specialized sub-agents
	emailAgent := agent.NewEmailAgent(llmClient)

	// Create orchestrator with sub-agents
	orch := agent.NewOrchestrator(
		agent.Config{MaxSteps: opts.maxSteps},
		planner,
		toolbox,
		log.With().Str("comp", "orch").Logger(),
		emailAgent, // Register EmailAgent
	)

	fmt.Println("Начинаю задачу...")
	task := agent.Task{Description: opts.task}
	err = orch.Run(ctx, task, func(c context.Context) (snapshot.Summary, error) {
		return snapshot.Collect(c, ctrl)
	})
	if err != nil {
		log.Error().Err(err).Msg("run finished with error")
	} else if opts.saveState != "" {
		if err := ctrl.SaveState(ctx, opts.saveState); err != nil {
			log.Error().Err(err).Msg("save state")
		} else {
			log.Info().Str("path", opts.saveState).Msg("storage saved")
		}
	}
}

func parseFlags() cliOptions {
	task := flag.String("task", "", "Task description")
	storage := flag.String("storage", "", "Path to Playwright storage state")
	save := flag.String("save-state", "", "Path to save updated storage state")
	maxSteps := flag.Int("max-steps", 40, "Max agent steps")
	temp := flag.Float64("temperature", 0.1, "LLM temperature")
	flag.Parse()
	return cliOptions{
		task:        strings.TrimSpace(*task),
		storage:     strings.TrimSpace(*storage),
		saveState:   strings.TrimSpace(*save),
		maxSteps:    *maxSteps,
		temperature: *temp,
	}
}

func promptTask() (string, bool, error) {
	reader := bufio.NewReader(os.Stdin)
	fmt.Print("Введите задачу (оставьте пустым, чтобы отменить): ")
	line, err := reader.ReadString('\n')
	if err != nil {
		return "", false, err
	}
	line = strings.TrimSpace(line)
	if line == "" {
		return "", true, nil
	}

	// Validate and sanitize input
	const maxTaskLength = 2000
	if len(line) > maxTaskLength {
		fmt.Printf("Задача слишком длинная (макс. %d символов), обрезана\n", maxTaskLength)
		line = line[:maxTaskLength]
	}

	// Basic sanitization: remove control characters except newlines/tabs
	var sanitized strings.Builder
	for _, r := range line {
		if r >= 32 || r == '\n' || r == '\r' || r == '\t' {
			sanitized.WriteRune(r)
		}
	}

	return sanitized.String(), false, nil
}

func terminalPrompt() tools.PromptFunc {
	reader := bufio.NewReader(os.Stdin)
	return func(ctx context.Context, message string) (string, error) {
		fmt.Printf("\n=== Требуется ввод ===\n%s\n> ", message)
		text, err := reader.ReadString('\n')
		if err != nil {
			return "", err
		}
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		default:
		}
		return strings.TrimSpace(text), nil
	}
}
