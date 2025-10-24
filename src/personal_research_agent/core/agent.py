"""
Main Personal Research Agent implementation using LangChain and LangGraph.
Orchestrates research tasks, tool usage, and memory management.
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
import logging

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from .state import AgentState, TaskStatus, ResearchTask
from .memory import AgentMemory
from ..config import get_settings
from ..tools.web_search import WebSearchTool
from ..tools.document_processor import DocumentProcessor
from ..tools.summarizer import SummarizerTool


class ResearchAgentCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for the research agent."""
    
    def __init__(self, agent_state: AgentState):
        self.agent_state = agent_state
        self.start_time = None
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Called when LLM starts running."""
        self.start_time = datetime.now()
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when LLM ends running."""
        if self.start_time:
            response_time = (datetime.now() - self.start_time).total_seconds()
            self.agent_state.update_metrics(success=True, response_time=response_time)
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Called when LLM errors."""
        if self.start_time:
            response_time = (datetime.now() - self.start_time).total_seconds()
            self.agent_state.update_metrics(success=False, response_time=response_time)
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """Called when tool starts running."""
        tool_name = serialized.get("name", "unknown_tool")
        self.agent_state.record_tool_usage(tool_name)
        self.agent_state.add_message(
            role="system",
            content=f"Using tool: {tool_name} with input: {input_str}",
            metadata={"action_type": "tool_call", "tool": tool_name}
        )


class PersonalResearchAgent:
    """
    Main Personal Research Agent that orchestrates research tasks using LangChain.
    Integrates with local LLMs via LM Studio and manages memory and tools.
    """
    
    def __init__(self, session_id: Optional[str] = None, user_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.user_id = user_id
        self.settings = get_settings()
        
        # Initialize state and memory
        self.state = AgentState(session_id=self.session_id, user_id=user_id)
        self.memory = AgentMemory(session_id=self.session_id, user_id=user_id)
        
        # Initialize LLM
        self._init_llm()
        
        # Initialize tools
        self._init_tools()
        
        # Initialize agent
        self._init_agent()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _init_llm(self) -> None:
        """Initialize the language model."""
        llm_kwargs = self.settings.get_llm_kwargs()
        
        # For local LLMs via LM Studio, we use ChatOpenAI with custom base_url
        self.llm = ChatOpenAI(
            model=llm_kwargs["model"],
            temperature=llm_kwargs["temperature"],
            max_tokens=llm_kwargs["max_tokens"],
            timeout=llm_kwargs["timeout"],
            base_url=llm_kwargs.get("base_url", "http://localhost:1234/v1"),
            api_key=llm_kwargs.get("api_key", "lm-studio"),  # LM Studio doesn't require real API key
            streaming=True
        )

        # Health check: try a tiny prompt with retry/backoff
        async def _check():
            try:
                _ = await self.llm.ainvoke("ping")
                return True
            except Exception:
                return False

        async def _retry_check(retries: int = 2, delay: float = 0.8) -> bool:
            for i in range(retries + 1):
                ok = await _check()
                if ok:
                    return True
                if i < retries:
                    await asyncio.sleep(delay)
            return False

        # Launch background health check; don't block construction
        asyncio.get_event_loop().create_task(_retry_check())
    
    def _init_tools(self) -> None:
        """Initialize research tools."""
        self.tools = [
            WebSearchTool(),
            DocumentProcessor(),
            SummarizerTool()
        ]
    
    def _init_agent(self) -> None:
        """Initialize the LangChain agent with a simplified approach."""
        # Create system prompt
        system_prompt = """You are a Personal Research Assistant powered by advanced AI. Your role is to help users conduct thorough research on any topic they're interested in.

Your capabilities include:
- Searching the web for current information
- Processing and analyzing documents
- Summarizing complex information
- Maintaining context across conversations
- Learning from user preferences

Guidelines:
1. Always be thorough and accurate in your research
2. Cite sources when providing information
3. Ask clarifying questions when the user's request is ambiguous
4. Provide structured, well-organized responses
5. Remember previous conversations and build upon them
6. Adapt your communication style to the user's preferences

When conducting research:
- Use multiple sources to verify information
- Look for recent and authoritative sources
- Provide both summary and detailed information as appropriate
- Highlight any limitations or uncertainties in the information

Remember: You have access to web search, document processing, and summarization tools. Use them effectively to provide comprehensive research assistance."""

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Create a simple chain for now
        self.agent_chain = self.prompt | self.llm
    
    async def research(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Conduct research on a given query.
        
        Args:
            query: The research question or topic
            context: Optional context to guide the research
            
        Returns:
            Dictionary containing research results and metadata
        """
        try:
            # Create research task
            task = self.state.create_task(query, metadata=context)
            self.state.update_task_status(task.id, TaskStatus.IN_PROGRESS)
            
            # Add user message to state
            self.state.add_message("human", query, metadata=context)
            
            # Execute research
            result = await self._execute_research(query, task.id)
            
            # Update task status
            self.state.update_task_status(task.id, TaskStatus.COMPLETED)
            self.state.add_task_result(task.id, result)
            
            # Store in memory
            await self.memory.add_conversation_turn(query, result.get("output", ""), context)
            await self.memory.store_research_result(query, result)
            
            return {
                "task_id": task.id,
                "query": query,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id
            }
            
        except Exception as e:
            self.logger.error(f"Research failed: {e}")
            
            # Update task status
            if 'task' in locals():
                self.state.update_task_status(task.id, TaskStatus.FAILED, str(e))
            
            return {
                "error": str(e),
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id
            }
    
    async def _execute_research(self, query: str, task_id: str) -> Dict[str, Any]:
        """Execute the actual research using the agent."""
        try:
            # Get relevant context from memory
            memory_context = await self.memory.search_memory(query, limit=3)
            
            # Prepare input with context
            chat_history = await self.memory.get_conversation_history(limit=5)
            
            # Format chat history for the prompt
            formatted_history = []
            for turn in chat_history:
                formatted_history.append(HumanMessage(content=turn["human_message"]))
                formatted_history.append(AIMessage(content=turn["ai_message"]))
            
            # Add memory context if available
            enhanced_query = query
            if memory_context:
                context_text = "\n".join([
                    f"Previous research: {ctx['content'][:200]}..."
                    for ctx in memory_context
                ])
                enhanced_query = f"{query}\n\nRelevant context from previous research:\n{context_text}"
            
            # Determine if we need to use tools
            needs_web_search = any(keyword in query.lower() for keyword in 
                                 ["latest", "recent", "current", "news", "today", "2024", "2025"])
            needs_summarization = "summarize" in query.lower() or "summary" in query.lower()
            
            # Use tools if needed
            tool_results = []
            if needs_web_search:
                try:
                    web_tool = WebSearchTool()
                    web_result = await web_tool._arun(enhanced_query)
                    tool_results.append(f"Web search results:\n{web_result}")
                except Exception as e:
                    self.logger.warning(f"Web search failed: {e}")
            
            # Combine query with tool results
            if tool_results:
                enhanced_query = f"{enhanced_query}\n\nAdditional information:\n" + "\n\n".join(tool_results)
            
            # Execute the chain
            try:
                result = await self.agent_chain.ainvoke({
                    "input": enhanced_query,
                    "chat_history": formatted_history
                })
                return {"output": result.content}
            except Exception as e:
                if "503" in str(e) or "connection" in str(e).lower():
                    return {"output": (
                        "I'm having trouble reaching the local model. "
                        "Please ensure LM Studio is running, the model is loaded, "
                        "and the server is started on http://localhost:1234/v1."
                    )}
                raise
            
        except Exception as e:
            self.logger.error(f"Agent execution failed: {e}")
            raise
    
    async def chat(self, message: str) -> str:
        """
        Have a conversational interaction with the agent.
        
        Args:
            message: User's message
            
        Returns:
            Agent's response
        """
        try:
            # Add user message to state
            self.state.add_message("human", message)
            
            # Get chat history
            chat_history = await self.memory.get_conversation_history(limit=10)
            
            # Format chat history for the prompt
            formatted_history = []
            for turn in chat_history:
                formatted_history.append(HumanMessage(content=turn["human_message"]))
                formatted_history.append(AIMessage(content=turn["ai_message"]))
            
            # Execute the chain
            try:
                result = await self.agent_chain.ainvoke({
                    "input": message,
                    "chat_history": formatted_history
                })
                response = result.content
            except Exception as e:
                # Friendly error for common LM Studio issues
                if "503" in str(e) or "connection" in str(e).lower():
                    response = (
                        "I'm having trouble reaching the local model. "
                        "Please ensure LM Studio is running, the model is loaded, "
                        "and the server is started on http://localhost:1234/v1."
                    )
                else:
                    raise
            
            # Add AI response to state and memory
            self.state.add_message("assistant", response)
            await self.memory.add_conversation_turn(message, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Chat failed: {e}")
            error_response = f"I encountered an error: {str(e)}"
            self.state.add_message("assistant", error_response)
            return error_response
    
    async def get_research_summary(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Get a summary of research activities."""
        if task_id:
            # Get specific task summary
            task = next((t for t in self.state.tasks if t.id == task_id), None)
            if task:
                return {
                    "task_id": task.id,
                    "query": task.query,
                    "status": task.status,
                    "results_count": len(task.results),
                    "created_at": task.created_at.isoformat(),
                    "updated_at": task.updated_at.isoformat()
                }
            else:
                return {"error": "Task not found"}
        else:
            # Get overall summary
            return self.state.get_research_summary()
    
    async def set_user_preference(self, key: str, value: Any) -> None:
        """Set a user preference."""
        await self.memory.store_user_preference(key, value)
        self.state.user_preferences[key] = value
    
    async def get_user_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference."""
        return await self.memory.get_user_preference(key, default)
    
    async def search_history(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search through research history."""
        return await self.memory.search_memory(query, limit=limit)
    
    async def clear_session(self) -> None:
        """Clear the current session."""
        await self.memory.clear_session_memory()
        self.state = AgentState(session_id=self.session_id, user_id=self.user_id)
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current session."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.state.created_at.isoformat(),
            "total_queries": self.state.total_queries,
            "successful_queries": self.state.successful_queries,
            "tools_used": self.state.tools_used,
            "active_task": self.state.get_active_task().query if self.state.get_active_task() else None
        }
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return await self.memory.get_memory_stats()
    
    def __repr__(self) -> str:
        return f"PersonalResearchAgent(session_id='{self.session_id}', user_id='{self.user_id}')"
