import pytest
import asyncio

from personal_research_agent.tools.web_search import WebSearchTool
from personal_research_agent.tools.document_processor import DocumentProcessor
from personal_research_agent.tools.summarizer import SummarizerTool


def test_web_search_init():
    tool = WebSearchTool()
    # Lazy init - nothing to assert eagerly, just ensure constructable
    assert tool.name == "web_search"


@pytest.mark.asyncio
async def test_summarizer_short_text():
    tool = SummarizerTool()
    out = await tool._arun("short", max_length=50)
    assert "too short" in out.lower()


@pytest.mark.asyncio
async def test_document_processor_detect_text():
    tool = DocumentProcessor()
    out = await tool._arun("This is plain text content.", source_type="auto")
    assert "Document Processing Results" in out

