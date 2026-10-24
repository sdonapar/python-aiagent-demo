"""Thin wrapper around the Google Gen AI SDK (Gemini).

Only the Google SDK is used. There is no AI-agent framework here: the agent
loop in ``loop.py`` talks to the model directly with plain messages and its
own homemade tool loop.
"""

from __future__ import annotations

import os

from google.genai import Client, types

from .tools import TOOLS


class LLM:
    """A minimal chat wrapper over the Gemini API."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Put it in the .env file "
                "(see .env.example) or export it in your shell before running."
            )
        self.client = Client(api_key=key)
        self.parts: list[types.Content] = []

    def _generation_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=[types.FunctionDeclaration(**tool) for tool in TOOLS])],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.AUTO)
            ),
            # Disable the SDK's automatic function calling: the public API only
            # takes a fake callable, so tool execution MUST go through our own
            # permission-aware loop and can never be auto-executed here.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def generate(self, prompt: str | None = None) -> types.GenerateContentResponse:
        """Call the model with accumulated history (plus ``prompt`` if given)."""
        if prompt:
            self.parts.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
        response = self.client.models.generate_content(
            model=self.model,
            contents=self.parts,
            config=self._generation_config(),
        )
        self._append_content(response.candidates[0].content)
        return response

    def send_function_results(
        self,
        function_calls: list[types.FunctionCall],
        results: list[str],
    ) -> types.GenerateContentResponse:
        """Feed the tool results back to the model (user-turn function responses)."""
        response_parts = [
            types.Part(
                function_response=types.FunctionResponse(
                    name=call.name,
                    id=call.id,
                    response={"result": result},
                )
            )
            for call, result in zip(function_calls, results, strict=False)
        ]
        self.parts.append(types.Content(role="user", parts=response_parts))
        response = self.client.models.generate_content(
            model=self.model,
            contents=self.parts,
            config=self._generation_config(),
        )
        self._append_content(response.candidates[0].content)
        return response

    def _append_content(self, content: types.Content) -> None:
        """Merge consecutive turns with the same role so history stays valid."""
        if content and content.parts:
            if self.parts and self.parts[-1].role == content.role:
                self.parts[-1].parts.extend(content.parts)
            else:
                self.parts.append(content)

    @staticmethod
    def is_function_call(response: types.GenerateContentResponse) -> list[types.FunctionCall]:
        """Return the FunctionCalls in the response, or [] if none."""
        assert response.candidates, "empty response candidate"
        parts = response.candidates[0].content.parts
        return [p.function_call for p in parts if p.function_call]

    @staticmethod
    def extract_text(response: types.GenerateContentResponse) -> str:
        """Concatenate plain text parts in the response."""
        out: list[str] = []
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.text:
                    out.append(part.text)
        return "\n".join(out).strip()

    @staticmethod
    def is_finish_reason(
        response: types.GenerateContentResponse, reason: types.FinishReason
    ) -> bool:
        return (
            response.candidates
            and response.candidates[0].finish_reason == reason
        )

    @staticmethod
    def is_blocked(response: types.GenerateContentResponse) -> bool:
        return LLM.is_finish_reason(response, types.FinishReason.SAFETY) or LLM.is_finish_reason(
            response, types.FinishReason.BLOCKLIST
        )

    @staticmethod
    def finish_reason_name(response: types.GenerateContentResponse) -> str:
        if response.candidates:
            return str(response.candidates[0].finish_reason)
        return "no candidates"