# Personal Research Agent ✨

Your stylish, local-first AI research buddy. Built on modern LangChain, designed for LM Studio, crafted for clarity and learning.

## Highlights

- 🤖 **Local LLM by default**: Works out-of-the-box with LM Studio (OpenAI-compatible API).
- 🧠 **Memory**: Conversation memory + vector search (Chroma + HF embeddings).
- 🔍 **Web Search**: DuckDuckGo instant answers + mock fallback.
- 📄 **Docs**: Process PDF/DOCX/MD/HTML/TXT; chunk and analyze.
- 📝 **Summarization**: General, bullet-points, key-insights, executive, focused.
- 💬 **Elegant CLI**: Clean interactive mode plus one-shot commands.
- 🛡️ **Resilient UX**: Health check + friendly errors when LM Studio isn’t ready.

---

## TL;DR Run Guide 🚀

```bash
pip install -e .

# Optional: quick smoke test (works even if LM Studio is down)
python examples/quick_test.py

# Interactive CLI
python -m personal_research_agent.cli interactive

# One-shot research
research-agent research "What changed in Python 3.13?"
```

LM Studio setup:
- Open LM Studio → load Qwen 2.5 Coder 7B Instruct → Start Server (default `http://localhost:1234/v1`).

---

## Tools & Tech 🧰

Used in this project (ready today):
- LangChain (core, prompts, tools)
- `langchain_chroma` (vector store)
- `langchain_huggingface` (embeddings)
- Typer + Rich (CLI)

Optional add-ons (not required, nice upgrades later):
- LangGraph (stateful planner/reflection workflows)
- LiteLLM (switch providers seamlessly: OpenAI/Anthropic/local)
- FastAPI (server), OpenTelemetry (tracing)

---

## Interactive CLI 💬

Start interactive mode:

```bash
python -m personal_research_agent.cli interactive
# or
research-agent interactive
```

Inside interactive mode:
- `/help` – show commands
- `/settings` – current config + session info
- `/history` – recent conversation
- `/stats` – session metrics (tasks, avg latency)
- `/clear` – clear session memory
- `/quit` – exit

One-shot command:

```bash
research-agent research "Latest LLM evaluation techniques"
```

Output formats:

```bash
research-agent research "Compare RAG vs fine-tuning" -f markdown -o result.md
```

---

## Installation 📦

```bash
git clone <repository-url>
cd PersonalResearchAgent
pip install -e .
```

> Tip: If you see friendly “can’t reach local model” messages, just start LM Studio and re-run.

---

## Minimal Config ⚙️

Defaults are sensible. To override, create a `.env` in the repo root:

```env
# LLM
LLM__MODEL_NAME=qwen2.5-coder-7b-instruct
LLM__API_BASE=http://localhost:1234/v1
LLM__TEMPERATURE=0.7
LLM__MAX_TOKENS=4096

# Research
RESEARCH__ENABLE_WEB_SEARCH=true
# Memory
MEMORY__ENABLE_MEMORY=true
```

Show or change from CLI:

```bash
research-agent config --show
research-agent config --model "qwen2.5-coder-7b-instruct" --temperature 0.8
```

---

## Architecture 🧩

```
User (CLI)
  → PersonalResearchAgent (prompt + optional tools)
      → Memory (conversation + vector store)
      → Tools (web search, doc processing, summarizer)
      → LLM (LM Studio OpenAI-compatible)
  ← Response (also persisted to memory)
```

Key files:
- `core/agent.py` – the orchestrator (prompt + optional web search injection + friendly errors)
- `core/memory.py` – conversation turns + vector memory (Chroma + HF)
- `tools/` – `web_search.py`, `document_processor.py`, `summarizer.py`
- `cli.py` – interactive and one-shot commands

---

## Python API 🐍

```python
import asyncio
from personal_research_agent import PersonalResearchAgent

async def main():
    agent = PersonalResearchAgent(user_id="me")
    print(await agent.chat("Hello! Can you summarize the latest on LLM safety?"))

    result = await agent.research("Compare RAG vs fine-tuning with pros/cons")
    print(result["result"]["output"])  # formatted text

asyncio.run(main())
```

---

## Troubleshooting 🛠️

- “Can’t reach local model” / 503:
  - Start LM Studio, load model, Start Server. Default: `http://localhost:1234/v1`.
- Import issues:
  - `pip install -e .` again; ensure Python ≥ 3.9.
- Memory too big / slow:
  - Reduce `LLM__MAX_TOKENS`, adjust chunk sizes, or clear session (`/clear`).

---

## Why this is modern & minimal ✅

- Uses `langchain_chroma` and `langchain_huggingface` (current packages).
- Pydantic v2 tools with `PrivateAttr` + lazy init → clean & fast imports.
- Health check + friendly errors → excellent CLI UX while iterating locally.

---

## Credits & License

- Built with [LangChain](https://langchain.com) and [LM Studio](https://lmstudio.ai)
- Default model: Qwen 2.5 Coder Instruct (local)
- CLI: [Typer](https://typer.tiangolo.com) + [Rich](https://rich.readthedocs.io)

Licensed under MIT.

**Happy researching!** 🔬✨
