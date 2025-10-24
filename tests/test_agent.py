import asyncio
import pytest

from personal_research_agent.core.agent import PersonalResearchAgent


@pytest.mark.asyncio
async def test_agent_initialize():
    agent = PersonalResearchAgent()
    assert agent.session_id
    info = agent.get_session_info()
    assert info["session_id"] == agent.session_id


@pytest.mark.asyncio
async def test_agent_chat_handles_llm_down(monkeypatch):
    agent = PersonalResearchAgent()

    async def fake_invoke(_):
        raise RuntimeError("LLM down")

    # Patch the underlying chain to avoid hitting LM Studio
    agent.agent_chain.ainvoke = fake_invoke

    resp = await agent.chat("hello")
    assert "error" in resp.lower()

