import os

from personal_research_agent.config import get_settings


def test_settings_defaults():
    settings = get_settings()
    assert settings.llm.model_name == "qwen2.5-coder-7b-instruct"
    assert settings.llm.api_base == "http://localhost:1234/v1"
    assert settings.research.enable_web_search is True
    assert settings.vector_db.collection_name.startswith("research_knowledge")

