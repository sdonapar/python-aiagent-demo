# python-aiagent-demo

A tiny AI agent built **without any agent framework** (no LangChain, no
CrewAI, no AutoGen — just the Google Gemini SDK for calling the model). It runs
an interactive loop you can chat with, and it can use four homemade tools.

## Features

- **Interactive REPL loop** — type a message, get a reply, type the next one.
  Every message is appended to the conversation history, so the model stays in
  context across turns.
- **4 tools, written by hand**:
  - `read_file(path)` — read a file
  - `write_file(path, content)` — write a file
  - `list_files(path)` — list a directory
  - `run_command(command)` — run a shell command, **only after the user
    approves it** (the agent asks `y`/`n` every single time)
- **Human-in-the-loop permissions** — the SDK's automatic function calling is
  disabled, so *no* tool can execute without going through the agent loop and
  the permission prompt.

## Requirements

- Python 3.12+
- `uv` (or just `pip install google-genai python-dotenv`)
- A Google AI Studio API key

## Setup

1. Get a key at <https://aistudio.google.com/apikey>.
2. Put it in `.env` (see [`.env.example`](.env.example)):
   ```dotenv
   GOOGLE_API_KEY=AIza...
   ```
3. Install dependencies:
   ```bash
   uv sync
   ```

## Run

```bash
uv run python main.py
```

Example session:

```
you> list the files in this directory

[agent] calling tool: list_files({"path": "."})
README.md
demo_agent/
main.py
...

assistant> Here's what's in the current directory...

you> run 'ls -la' for me

[permission] The agent wants to run:
  ls -la
[permission] Approve? [y]es / [n]o: y

[exit code: 0]
total 32
...
```

Type `exit`, `quit`, or `bye` to leave.

## How it works

```
loop.py   REPL + permission prompt + tool-call dispatch (your own while-loop)
llm.py    thin wrapper around google-genai (Gemini chat + tool declarations)
tools.py  the four tools, plus their JSON function specs
```

The loop keeps one growing `contents` list (roles `user` / `model` /
`functionResponse`), sends it to Gemini, and if the model picks a tool it
executes the tool and hands the result back to the model — repeating until the
model produces a normal text reply. Nothing about this loop comes from an agent
framework.