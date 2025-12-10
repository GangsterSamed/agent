# Fast Browser Agent (Go + Anthropic)

Минимальный быстрый агент для видимого Playwright-браузера:

- короткий контекст (снапшот страницы + список интерактивных элементов);
- LLM только выбирает действие в формате JSON, без длинного reasoning;
- toolbox без хардкода селекторов: navigate/click_text/click_role/fill/read/scroll/wait/request_user_input/save_state;
- поддержка storage state (persistent session), headless выключен по умолчанию через `AGENT_HEADLESS=false`.

## Запуск

```bash
cd /Users/polzovatel/go/src/backendMicroservice/agents/Ai-agent-for-browser-fast
go mod tidy
go run ./cmd/agent -task "Открой hh.ru и покажи новые вакансии"
# или интерактивно без -task
```

Флаги:
- `-storage path` — путь к Playwright storage state (cookies).
- `-save-state path` — сохранить обновлённый state после успешного прогона.
- `-max-steps 60` — лимит шагов.
- `-temperature 0.1` — температура LLM.

Переменные окружения:
- `ANTHROPIC_API_KEY` (обязательно)
- `ANTHROPIC_MODEL` (опционально, по умолчанию claude-sonnet-4-5-20250929)
- `AGENT_HEADLESS=false` чтобы браузер был видимым.
```

