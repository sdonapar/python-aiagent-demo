"""The agent's four tools, implemented by hand (no agent frameworks).

Every tool is a plain wrapper around stdlib functionality:

* read_file   - read a file from disk
* write_file  - write text (or bytes) to a file
* list_files  - list the files in a directory
* run_command - run a shell command, ONLY after the user explicitly allows it

The only thing these functions need from the outside is the result of
``ask_permission``, which is injected so the tool set stays pure and the
permission policy stays in one place.
"""

from __future__ import annotations

import base64
import os
import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Rationale for 5 MiB: a generous limit for normal text/artifacts while
# avoiding accidental write_file/read_file of huge binaries into memory.
MAX_FILE_BYTES = 5 * 1024 * 1024


def read_file(path: str) -> str:
    """Read and return the contents of ``path`` (UTF-8 text by default)."""
    file = Path(path)
    if not file.is_file():
        return f"ERROR: no such file: {path}"
    data = file.read_bytes()
    if len(data) > MAX_FILE_BYTES:
        return f"ERROR: file is too large ({len(data)} bytes, limit {MAX_FILE_BYTES})"
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return (
            f"ERROR: file is not UTF-8 text ({len(data)} bytes). "
            "Write a script with run_command if you need to process a binary."
        )


def write_file(path: str, content: str) -> str:
    """Write ``content`` to ``path``.

    By default the text is written as UTF-8; if the file path ends in a known
    binary extension (.png, .jpg, .jpeg, .gif, .webp, .pdf, .zip, .bin) the
    content is decoded from base64 first.
    """
    file = Path(path)
    if file.exists() and not file.is_file():
        return f"ERROR: path is not a regular file: {path}"
    if len(content.encode("utf-8")) > MAX_FILE_BYTES:
        return f"ERROR: content is too large (limit {MAX_FILE_BYTES} bytes)"
    data = content.encode("utf-8")
    if file.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip", ".bin"}:
        data = base64.b64decode(content)
    try:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(data)
    except OSError as exc:
        return f"ERROR: could not write {path}: {exc}"
    return f"OK: wrote {len(data)} bytes to {path}"


def list_files(path: str = ".") -> str:
    """List the entries under ``path`` and return them one per line."""
    root = Path(path)
    if not root.is_dir():
        return f"ERROR: no such directory: {path}"
    entries = sorted(
        (child.name + ("/" if child.is_dir() else "")) for child in root.iterdir()
    )
    if not entries:
        return f"(empty directory: {path})"
    return "\n".join(entries)


def run_command(command: str, ask_permission: Callable[[str], bool]) -> str:
    """Run ``command`` in a shell, but only with the user's explicit consent."""
    if not command.strip():
        return "ERROR: empty command"
    if not ask_permission(command):
        return "ERROR: command was not approved by the user"
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 60 seconds"
    except OSError as exc:
        return f"ERROR: failed to run command: {exc}"
    output = result.stdout or ""
    if result.stderr:
        output += ("\n" if output else "") + f"[stderr]\n{result.stderr}"
    status = result.returncode
    return f"[exit code: {status}]\n{output}" if output else f"[exit code: {status}]"


def _tool_specs() -> dict[str, dict[str, Any]]:
    """Gemini function-declaration specs for the four tools.

    Each spec's keys match ``types.FunctionDeclaration`` directly, so the LLM
    wrapper can build declarations with ``FunctionDeclaration(**spec)``.
    """
    return {
        "read_file": {
            "name": "read_file",
            "description": "Read a file from disk and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path of the file to read.",
                    }
                },
                "required": ["path"],
            },
        },
        "write_file": {
            "name": "write_file",
            "description": (
                "Write content to a file. Text is written as UTF-8. For binary "
                "extensions (.png, .jpg, .pdf, .zip, ...) provide base64."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path of the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full content of the file (base64 for binary).",
                    },
                },
                "required": ["path", "content"],
            },
        },
        "list_files": {
            "name": "list_files",
            "description": "List the entries in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list (defaults to '.').",
                    }
                },
                "required": [],
            },
        },
        "run_command": {
            "name": "run_command",
            "description": (
                "Run a shell command. IMPORTANT: the user will ALWAYS be asked "
                "to approve the command before it is executed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run.",
                    }
                },
                "required": ["command"],
            },
        },
    }


TOOLS: list[dict[str, Any]] = list(_tool_specs().values())

NAME_TO_TOOL = {tool["name"]: tool for tool in TOOLS}


def execute_tool(name: str, args: dict[str, Any], ask_permission: Callable[[str], bool]) -> str:
    """Dispatch a parsed tool call to its implementation and return the result."""
    if name not in NAME_TO_TOOL:
        return f"ERROR: unknown tool '{name}'"
    try:
        if name == "read_file":
            return read_file(args["path"])
        if name == "write_file":
            return write_file(args["path"], args["content"])
        if name == "list_files":
            return list_files(args.get("path", "."))
        if name == "run_command":
            return run_command(args["command"], ask_permission)
    except KeyError as exc:
        return f"ERROR: tool '{name}' is missing argument {exc}"
    except Exception as exc:  # last-resort safety net; never crash the agent loop
        return f"ERROR: tool '{name}' failed: {exc}"
    raise AssertionError(f"unreachable: {name}")