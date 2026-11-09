import json

import httpx
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel

from agents.llm_config import LLM_ENV_VARS, LLM_PROVIDERS, initialize_agent_llm


class Result(BaseModel):
    value: str


def read_sample() -> str:
    """Return the fixture value."""
    return "sample"


def test_deepseek_thinking_round_trips_reasoning_with_structured_tools(monkeypatch):
    for name in LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        assert request.url == "https://api.deepseek.com/chat/completions"
        assert body["model"] == "deepseek-flash"
        assert body["thinking"] == {"type": "enabled"}
        assert body["reasoning_effort"] == "high"
        assert body["tool_choice"] == "auto"
        turn = len(requests)
        if turn == 1:
            tool, arguments = "read_sample", "{}"
        else:
            assert body["messages"][-2]["reasoning_content"] == "fixture reasoning 1"
            assert body["messages"][-1]["content"] == "sample"
            tool, arguments = "Result", '{"value":"sample"}'
        return httpx.Response(
            200,
            json={
                "id": f"response-{turn}",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek-flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": f"fixture reasoning {turn}",
                            "tool_calls": [
                                {
                                    "id": f"call-{turn}",
                                    "type": "function",
                                    "function": {"name": tool, "arguments": arguments},
                                }
                            ],
                        },
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setitem(LLM_PROVIDERS["deepseek"].extra_args, "http_client", client)
        model = initialize_agent_llm()
        agent = create_agent(model, tools=[read_sample], response_format=ToolStrategy(Result))
        state = agent.invoke({"messages": [{"role": "user", "content": "Read and report the sample."}]})

    assert state["structured_response"] == Result(value="sample")
    assert len(requests) == 2
