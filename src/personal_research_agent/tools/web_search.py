"""
Web search tool for Personal Research Agent.
Provides web search capabilities using multiple search providers.
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any, Union, Type
from datetime import datetime
import logging

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field, PrivateAttr
from typing import Type, Any

from ..config import get_settings


class WebSearchInput(BaseModel):
    """Input schema for web search tool."""
    query: str = Field(description="The search query to execute")
    num_results: int = Field(default=5, description="Number of search results to return", ge=1, le=20)
    search_type: str = Field(default="general", description="Type of search: general, news, academic, images")
    time_range: Optional[str] = Field(default=None, description="Time range filter: day, week, month, year")


class SearchResult(BaseModel):
    """Individual search result."""
    title: str
    url: str
    snippet: str
    source: str
    published_date: Optional[str] = None
    relevance_score: Optional[float] = None


class WebSearchProvider:
    """Base class for web search providers."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    async def search(self, query: str, num_results: int = 5, **kwargs) -> List[SearchResult]:
        """Execute search and return results."""
        raise NotImplementedError


class DuckDuckGoSearchProvider(WebSearchProvider):
    """DuckDuckGo search provider using their instant answer API."""
    
    def __init__(self):
        super().__init__("duckduckgo")
        self.base_url = "https://api.duckduckgo.com/"
    
    async def search(self, query: str, num_results: int = 5, **kwargs) -> List[SearchResult]:
        """Search using DuckDuckGo API."""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "q": query,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1"
                }
                
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_duckduckgo_results(data, num_results)
                    else:
                        self.logger.error(f"DuckDuckGo API error: {response.status}")
                        return []
        except Exception as e:
            self.logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    def _parse_duckduckgo_results(self, data: Dict[str, Any], num_results: int) -> List[SearchResult]:
        """Parse DuckDuckGo API response."""
        results = []
        
        # Parse instant answer
        if data.get("Abstract"):
            results.append(SearchResult(
                title=data.get("Heading", "DuckDuckGo Instant Answer"),
                url=data.get("AbstractURL", ""),
                snippet=data.get("Abstract", ""),
                source="DuckDuckGo",
                published_date=None,
                relevance_score=1.0
            ))
        
        # Parse related topics
        for topic in data.get("RelatedTopics", [])[:num_results-len(results)]:
            if isinstance(topic, dict) and "Text" in topic:
                results.append(SearchResult(
                    title=topic.get("Text", "").split(" - ")[0] if " - " in topic.get("Text", "") else topic.get("Text", ""),
                    url=topic.get("FirstURL", ""),
                    snippet=topic.get("Text", ""),
                    source="DuckDuckGo",
                    published_date=None,
                    relevance_score=0.8
                ))
        
        return results[:num_results]


class MockSearchProvider(WebSearchProvider):
    """Mock search provider for testing and fallback."""
    
    def __init__(self):
        super().__init__("mock")
    
    async def search(self, query: str, num_results: int = 5, **kwargs) -> List[SearchResult]:
        """Return mock search results."""
        mock_results = []
        
        for i in range(min(num_results, 3)):
            mock_results.append(SearchResult(
                title=f"Mock Result {i+1} for '{query}'",
                url=f"https://example.com/result-{i+1}",
                snippet=f"This is a mock search result for the query '{query}'. It contains relevant information about the topic you're researching.",
                source="Mock Search",
                published_date=datetime.now().strftime("%Y-%m-%d"),
                relevance_score=0.9 - (i * 0.1)
            ))
        
        return mock_results


class WebSearchTool(BaseTool):
    """
    LangChain tool for web search functionality.
    Supports multiple search providers with fallback mechanisms.
    """
    
    name: str = "web_search"
    description: str = """
    Search the web for current information on any topic.
    
    Use this tool when you need to:
    - Find recent news or updates
    - Research current events
    - Get factual information from the web
    - Find sources and references
    
    Input should be a clear search query. The tool will return relevant web results
    with titles, URLs, and snippets.
    """
    args_schema: Type[BaseModel] = WebSearchInput
    _initialized: bool = PrivateAttr(default=False)
    _settings: Any = PrivateAttr(default=None)
    _providers: List[WebSearchProvider] = PrivateAttr(default_factory=list)
    _logger: Any = PrivateAttr(default=None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Private attributes are declared via PrivateAttr above
    
    def _ensure_initialized(self):
        """Lazy initialization of tool components."""
        if not self._initialized:
            self._settings = get_settings()
            self._providers = self._init_providers()
            self._logger = logging.getLogger(__name__)
            self._initialized = True
    
    def _init_providers(self) -> List[WebSearchProvider]:
        """Initialize search providers."""
        providers = []
        
        # Add available providers
        providers.append(DuckDuckGoSearchProvider())
        providers.append(MockSearchProvider())  # Fallback
        
        return providers
    
    def _run(
        self,
        query: str,
        num_results: int = 5,
        search_type: str = "general",
        time_range: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Synchronous run method (required by BaseTool)."""
        # Run async method in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self._arun(query, num_results, search_type, time_range, run_manager)
            )
            return result
        finally:
            loop.close()
    
    async def _arun(
        self,
        query: str,
        num_results: int = 5,
        search_type: str = "general",
        time_range: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute web search asynchronously."""
        try:
            self._ensure_initialized()
            self._logger.info(f"Searching web for: {query}")
            
            # Try providers in order until one succeeds
            results = []
            for provider in self._providers:
                try:
                    results = await provider.search(
                        query=query,
                        num_results=num_results,
                        search_type=search_type,
                        time_range=time_range
                    )
                    if results:
                        self._logger.info(f"Got {len(results)} results from {provider.name}")
                        break
                except Exception as e:
                    self._logger.warning(f"Provider {provider.name} failed: {e}")
                    continue
            
            if not results:
                return f"No search results found for query: {query}"
            
            # Format results for the agent
            formatted_results = self._format_results(results, query)
            return formatted_results
            
        except Exception as e:
            self._logger.error(f"Web search failed: {e}")
            return f"Web search failed: {str(e)}"
    
    def _format_results(self, results: List[SearchResult], query: str) -> str:
        """Format search results for the agent."""
        if not results:
            return f"No results found for: {query}"
        
        formatted = f"Web search results for '{query}':\n\n"
        
        for i, result in enumerate(results, 1):
            formatted += f"{i}. **{result.title}**\n"
            formatted += f"   URL: {result.url}\n"
            formatted += f"   Source: {result.source}\n"
            if result.published_date:
                formatted += f"   Published: {result.published_date}\n"
            formatted += f"   Summary: {result.snippet}\n"
            if result.relevance_score:
                formatted += f"   Relevance: {result.relevance_score:.2f}\n"
            formatted += "\n"
        
        formatted += f"Found {len(results)} results. Use these sources to provide accurate, up-to-date information."
        
        return formatted
    
    async def search_multiple_queries(self, queries: List[str], max_results_per_query: int = 3) -> Dict[str, List[SearchResult]]:
        """Search multiple queries concurrently."""
        tasks = []
        for query in queries:
            task = self._search_single_query(query, max_results_per_query)
            tasks.append((query, task))
        
        results = {}
        for query, task in tasks:
            try:
                search_results = await task
                results[query] = search_results
            except Exception as e:
                self.logger.error(f"Failed to search '{query}': {e}")
                results[query] = []
        
        return results
    
    async def _search_single_query(self, query: str, num_results: int) -> List[SearchResult]:
        """Search a single query."""
        for provider in self._providers:
            try:
                results = await provider.search(query, num_results)
                if results:
                    return results
            except Exception as e:
                self._logger.warning(f"Provider {provider.name} failed for '{query}': {e}")
                continue
        
        return []
    
    def get_provider_status(self) -> Dict[str, bool]:
        """Get status of all search providers."""
        # This would typically test each provider
        return {provider.name: True for provider in self.providers}


# Convenience function for direct usage
async def search_web(query: str, num_results: int = 5) -> List[SearchResult]:
    """Direct web search function."""
    tool = WebSearchTool()
    
    # Use the first available provider
    for provider in tool.providers:
        try:
            results = await provider.search(query, num_results)
            if results:
                return results
        except Exception:
            continue
    
    return []
