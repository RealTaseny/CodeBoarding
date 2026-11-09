from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from openai import BaseModel


class DeepSeekThinkingChat(ChatOpenAI):
    """Preserve DeepSeek reasoning across Chat Completions tool calls."""

    def _create_chat_result(self, response: dict | BaseModel, generation_info: dict | None = None) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        choices = response["choices"] if isinstance(response, dict) else response.model_dump()["choices"]
        for generation, choice in zip(result.generations, choices, strict=True):
            reasoning = choice["message"].get("reasoning_content")
            if reasoning is not None:
                generation.message.additional_kwargs["reasoning_content"] = reasoning
        return result

    def _get_request_payload(self, input_: LanguageModelInput, *, stop: list[str] | None = None, **kwargs: Any) -> dict:
        messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(messages, stop=stop, **kwargs)
        # CodeBoarding's prompt names the final tool. DeepSeek thinking rejects
        # a forced tool choice even when the agent expects a structured answer.
        tool_choice = payload.get("tool_choice")
        if tool_choice == "required" or isinstance(tool_choice, dict):
            payload["tool_choice"] = "auto"
        if payload.get("tools"):
            for message, serialized in zip(messages, payload["messages"], strict=True):
                if isinstance(message, AIMessage) and "reasoning_content" in message.additional_kwargs:
                    serialized["reasoning_content"] = message.additional_kwargs["reasoning_content"]
        return payload
