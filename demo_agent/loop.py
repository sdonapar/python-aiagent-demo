"""The agent's main loop: a REPL that keeps a persistent message history.

No agent framework — just a while-loop that asks the user for input, appends
the message to the conversation, and lets the model decide whether it wants to
talk or call one of the tools. Every tool dispatch also runs the permission
handler, and ``run_command`` only executes when the user approves it.
"""

from __future__ import annotations

import time

from .llm import LLM
from .tools import NAME_TO_TOOL, execute_tool

EXIT_COMMANDS = {"exit", "quit", "bye"}


class Agent:
    def __init__(self, model: str | None = None, loop_max: int = 20):
        self.llm = LLM(model=model)
        self.loop_max = loop_max

    def ask_permission(self, command: str) -> bool:
        """Ask the user to approve a shell command before it is executed."""
        print(f"\n[permission] The agent wants to run:\n  {command}")
        while True:
            reply = input("[permission] Approve? [y]es / [n]o: ").strip().lower()
            if reply in {"y", "yes", "approve", "allow"}:
                return True
            if reply in {"n", "no", "deny", "reject"}:
                return False
            print("[permission] Please answer 'y' or 'n'.")

    def chat(self, user_message: str) -> str:
        """Send one user message, and follow up until the model stops or hits the cap."""
        response = self.llm.generate(user_message)

        for _ in range(self.loop_max):
            if self.llm.is_blocked(response):
                return f"(blocked by the model: {self.llm.finish_reason_name(response)})"

            function_calls = self.llm.is_function_call(response)
            if not function_calls:
                # Normal text answer (finish reason is STOP here, which is fine).
                return self.llm.extract_text(response) or "(empty response)"

            results = []
            for call in function_calls:
                name, args = call.name, call.args
                print(f"\n[agent] calling tool: {name}({self._pretty_args(args)})")
                assert name in NAME_TO_TOOL, f"model requested unknown tool '{name}'"
                results.append(execute_tool(name, dict(args), self.ask_permission))

            response = self.llm.send_function_results(function_calls, results)

        return "(tool loop exceeded the step limit)"

    @staticmethod
    def _pretty_args(args: dict) -> str:
        """Render tool arguments compactly for the console."""
        try:
            import json

            return json.dumps(args)
        except (TypeError, ValueError):
            return str(args)


def run(model: str | None = None) -> None:
    """Start the interactive agent REPL."""
    try:
        agent = Agent(model=model)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return

    print("\nAgent ready. Type a message and press Enter (or type 'exit').")
    print("Ask me to read/write files, list folders, or run commands.\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            print("(type a message, or 'exit' to leave)")
            continue
        if user_input.lower() in EXIT_COMMANDS:
            break

        start = time.monotonic()
        reply = agent.chat(user_input)
        elapsed = time.monotonic() - start
        print(f"\nassistant> {reply}\n({elapsed:.1f}s)\n")