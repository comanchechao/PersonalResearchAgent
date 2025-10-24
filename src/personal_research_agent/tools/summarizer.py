"""
Summarization tool for Personal Research Agent.
Provides text summarization and analysis capabilities using the local LLM.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union, Type
from datetime import datetime
import logging

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, PrivateAttr

from ..config import get_settings


class SummarizerInput(BaseModel):
    """Input schema for summarizer tool."""
    text: str = Field(description="Text content to summarize")
    summary_type: str = Field(default="general", description="Type of summary: general, bullet_points, key_insights, executive")
    max_length: int = Field(default=200, description="Maximum length of summary in words", ge=50, le=1000)
    focus_areas: List[str] = Field(default_factory=list, description="Specific areas to focus on in the summary")


class SummaryResult(BaseModel):
    """Summary result with metadata."""
    summary: str
    summary_type: str
    original_length: int
    summary_length: int
    compression_ratio: float
    key_topics: List[str]
    confidence_score: float
    processing_time: float


class SummarizerTool(BaseTool):
    """
    LangChain tool for text summarization and analysis.
    Uses the local LLM to generate high-quality summaries.
    """
    
    name: str = "summarizer"
    description: str = """
    Summarize and analyze text content using advanced AI.
    
    Use this tool to:
    - Create concise summaries of long documents
    - Extract key insights and main points
    - Generate executive summaries
    - Create bullet-point summaries
    - Focus on specific aspects of the content
    - Analyze and distill complex information
    
    Input should include the text to summarize and the desired summary type.
    The tool will return a structured summary with metadata.
    """
    args_schema: Type[BaseModel] = SummarizerInput
    _initialized: bool = PrivateAttr(default=False)
    _settings: Any = PrivateAttr(default=None)
    _llm: Any = PrivateAttr(default=None)
    _prompts: dict = PrivateAttr(default_factory=dict)
    _text_splitter: Any = PrivateAttr(default=None)
    _logger: Any = PrivateAttr(default=None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _ensure_initialized(self):
        """Lazy initialization of tool components."""
        if not self._initialized:
            self._settings = get_settings()
            self._init_llm()
            self._init_prompts()
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,  # Smaller chunks for summarization
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            self._logger = logging.getLogger(__name__)
            self._initialized = True
    
    def _init_llm(self) -> None:
        """Initialize the language model for summarization."""
        llm_kwargs = self._settings.get_llm_kwargs()
        
        # Use a lower temperature for more focused summaries
        self._llm = ChatOpenAI(
            model=llm_kwargs["model"],
            temperature=0.3,  # Lower temperature for more focused output
            max_tokens=llm_kwargs["max_tokens"],
            timeout=llm_kwargs["timeout"],
            base_url=llm_kwargs.get("base_url", "http://localhost:1234/v1"),
            api_key=llm_kwargs.get("api_key", "lm-studio"),
        )
    
    def _init_prompts(self) -> None:
        """Initialize summarization prompts."""
        self._prompts = {
            "general": PromptTemplate(
                input_variables=["text", "max_length"],
                template="""
Please provide a comprehensive summary of the following text in approximately {max_length} words.
Focus on the main ideas, key points, and important details while maintaining clarity and coherence.

Text to summarize:
{text}

Summary:
"""
            ),
            
            "bullet_points": PromptTemplate(
                input_variables=["text", "max_length"],
                template="""
Please create a bullet-point summary of the following text with approximately {max_length} words total.
Organize the information into clear, concise bullet points that capture the main ideas.

Text to summarize:
{text}

Bullet-point summary:
"""
            ),
            
            "key_insights": PromptTemplate(
                input_variables=["text", "max_length"],
                template="""
Please extract the key insights and important takeaways from the following text in approximately {max_length} words.
Focus on the most significant findings, conclusions, and actionable information.

Text to summarize:
{text}

Key insights:
"""
            ),
            
            "executive": PromptTemplate(
                input_variables=["text", "max_length"],
                template="""
Please create an executive summary of the following text in approximately {max_length} words.
Focus on the most critical information that a decision-maker would need to know.
Include key findings, recommendations, and implications.

Text to summarize:
{text}

Executive summary:
"""
            ),
            
            "focused": PromptTemplate(
                input_variables=["text", "max_length", "focus_areas"],
                template="""
Please create a focused summary of the following text in approximately {max_length} words.
Pay special attention to these areas: {focus_areas}

Text to summarize:
{text}

Focused summary:
"""
            )
        }
    
    def _run(
        self,
        text: str,
        summary_type: str = "general",
        max_length: int = 200,
        focus_areas: List[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Synchronous run method (required by BaseTool)."""
        # Run async method in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self._arun(text, summary_type, max_length, focus_areas or [], run_manager)
            )
            return result
        finally:
            loop.close()
    
    async def _arun(
        self,
        text: str,
        summary_type: str = "general",
        max_length: int = 200,
        focus_areas: List[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Summarize text asynchronously."""
        start_time = datetime.now()
        focus_areas = focus_areas or []
        
        try:
            self._ensure_initialized()
            self._logger.info(f"Summarizing {len(text)} characters with type: {summary_type}")
            
            # Handle empty or very short text
            if not text or len(text.strip()) < 50:
                return "Text is too short to summarize meaningfully."
            
            # Split text if it's too long
            if len(text) > 8000:  # Approximate token limit consideration
                summary_result = await self._summarize_long_text(text, summary_type, max_length, focus_areas)
            else:
                summary_result = await self._summarize_single_text(text, summary_type, max_length, focus_areas)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            summary_result.processing_time = processing_time
            
            # Format result for the agent
            result = self._format_result(summary_result)
            
            self._logger.info(f"Summarization completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self._logger.error(f"Summarization failed: {e}")
            return f"Summarization failed: {str(e)}"
    
    async def _summarize_single_text(self, text: str, summary_type: str, max_length: int, focus_areas: List[str]) -> SummaryResult:
        """Summarize a single text chunk."""
        # Select appropriate prompt
        if focus_areas and summary_type == "general":
            prompt_template = self._prompts["focused"]
            prompt_input = {
                "text": text,
                "max_length": max_length,
                "focus_areas": ", ".join(focus_areas)
            }
        else:
            prompt_template = self._prompts.get(summary_type, self._prompts["general"])
            prompt_input = {
                "text": text,
                "max_length": max_length
            }
        
        # Generate summary
        prompt = prompt_template.format(**prompt_input)
        response = await self._llm.ainvoke(prompt)
        summary = response.content.strip()
        
        # Extract key topics (simple keyword extraction)
        key_topics = self._extract_key_topics(text)
        
        # Calculate metrics
        original_length = len(text.split())
        summary_length = len(summary.split())
        compression_ratio = summary_length / original_length if original_length > 0 else 0
        
        # Simple confidence score based on compression ratio and summary length
        confidence_score = min(1.0, max(0.1, 1.0 - abs(compression_ratio - 0.2)))
        
        return SummaryResult(
            summary=summary,
            summary_type=summary_type,
            original_length=original_length,
            summary_length=summary_length,
            compression_ratio=compression_ratio,
            key_topics=key_topics,
            confidence_score=confidence_score,
            processing_time=0.0  # Will be set by caller
        )
    
    async def _summarize_long_text(self, text: str, summary_type: str, max_length: int, focus_areas: List[str]) -> SummaryResult:
        """Summarize long text by chunking and combining summaries."""
        # Split text into chunks
        chunks = self._text_splitter.split_text(text)
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            try:
                # Use smaller max_length for individual chunks
                chunk_max_length = max(50, max_length // len(chunks))
                chunk_result = await self._summarize_single_text(chunk, summary_type, chunk_max_length, focus_areas)
                chunk_summaries.append(chunk_result.summary)
            except Exception as e:
                self._logger.warning(f"Failed to summarize chunk {i}: {e}")
                continue
        
        if not chunk_summaries:
            raise ValueError("Failed to summarize any chunks")
        
        # Combine chunk summaries
        combined_summary_text = " ".join(chunk_summaries)
        
        # Create final summary from combined chunks
        final_result = await self._summarize_single_text(combined_summary_text, summary_type, max_length, focus_areas)
        
        # Update metrics to reflect original text
        final_result.original_length = len(text.split())
        final_result.compression_ratio = final_result.summary_length / final_result.original_length
        
        return final_result
    
    def _extract_key_topics(self, text: str, max_topics: int = 5) -> List[str]:
        """Extract key topics from text using simple frequency analysis."""
        if not text:
            return []
        
        # Simple approach: find most frequent meaningful words
        import re
        from collections import Counter
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        # Extract words (simple tokenization)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out stop words and count frequencies
        meaningful_words = [word for word in words if word not in stop_words]
        word_counts = Counter(meaningful_words)
        
        # Return most common words as topics
        return [word for word, count in word_counts.most_common(max_topics)]
    
    def _format_result(self, result: SummaryResult) -> str:
        """Format the summary result for the agent."""
        formatted = f"Text Summarization Results:\n\n"
        
        # Summary
        formatted += f"**Summary ({result.summary_type}):**\n{result.summary}\n\n"
        
        # Metrics
        formatted += f"**Metrics:**\n"
        formatted += f"- Original length: {result.original_length:,} words\n"
        formatted += f"- Summary length: {result.summary_length:,} words\n"
        formatted += f"- Compression ratio: {result.compression_ratio:.2%}\n"
        formatted += f"- Processing time: {result.processing_time:.2f}s\n"
        formatted += f"- Confidence score: {result.confidence_score:.2f}\n\n"
        
        # Key topics
        if result.key_topics:
            formatted += f"**Key Topics:** {', '.join(result.key_topics)}\n\n"
        
        formatted += "The summary captures the main points while significantly reducing the text length."
        
        return formatted
    
    async def summarize_multiple_texts(self, texts: List[str], summary_type: str = "general", max_length: int = 200) -> List[SummaryResult]:
        """Summarize multiple texts concurrently."""
        tasks = []
        for text in texts:
            task = self._summarize_single_text(text, summary_type, max_length, [])
            tasks.append(task)
        
        results = []
        for task in tasks:
            try:
                result = await task
                results.append(result)
            except Exception as e:
                self._logger.error(f"Failed to summarize text: {e}")
                # Create error result
                error_result = SummaryResult(
                    summary=f"Summarization failed: {str(e)}",
                    summary_type=summary_type,
                    original_length=0,
                    summary_length=0,
                    compression_ratio=0.0,
                    key_topics=[],
                    confidence_score=0.0,
                    processing_time=0.0
                )
                results.append(error_result)
        
        return results
    
    def get_supported_summary_types(self) -> List[str]:
        """Get list of supported summary types."""
        return list(self._prompts.keys())


# Convenience function for direct usage
async def summarize_text(text: str, summary_type: str = "general", max_length: int = 200) -> SummaryResult:
    """Direct text summarization function."""
    tool = SummarizerTool()
    return await tool._summarize_single_text(text, summary_type, max_length, [])
