"""
Personal Research Agent - AI-powered research assistant using LangChain and local LLMs.

This package provides a comprehensive research assistant that can:
- Search and analyze web content
- Process and summarize documents
- Generate structured research reports
- Work with local LLMs via LM Studio
"""

__version__ = "0.1.0"
__author__ = "Chao"
__email__ = "chao@example.com"

from .core.agent import PersonalResearchAgent
from .agents import ResearchAgent  # Backward compatibility alias
from .tools.web_search import WebSearchTool
from .tools.document_processor import DocumentProcessor
from .tools.summarizer import SummarizerTool

__all__ = [
    "PersonalResearchAgent",
    "ResearchAgent",
    "WebSearchTool", 
    "DocumentProcessor",
    "SummarizerTool",
]
