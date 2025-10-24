"""
Agent state management for Personal Research Agent.
Defines the state structure used throughout the research workflow.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    """Status of research tasks."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResearchTask(BaseModel):
    """Individual research task."""
    id: str
    query: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None


class AgentState(BaseModel):
    """
    Central state management for the Personal Research Agent.
    Tracks conversation history, research tasks, and agent memory.
    """
    
    # Session information
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Conversation state
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    current_query: Optional[str] = None
    
    # Research tasks
    tasks: List[ResearchTask] = Field(default_factory=list)
    active_task_id: Optional[str] = None
    
    # Research results and knowledge
    research_results: List[Dict[str, Any]] = Field(default_factory=list)
    knowledge_base: Dict[str, Any] = Field(default_factory=dict)
    
    # Agent memory and context
    short_term_memory: List[Dict[str, Any]] = Field(default_factory=list)
    long_term_memory: Dict[str, Any] = Field(default_factory=dict)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    
    # Tool usage tracking
    tools_used: List[str] = Field(default_factory=list)
    tool_results: Dict[str, Any] = Field(default_factory=dict)
    
    # Performance metrics
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    average_response_time: float = 0.0
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to the conversation history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def create_task(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> ResearchTask:
        """Create a new research task."""
        task_id = f"task_{len(self.tasks) + 1}_{datetime.now().timestamp()}"
        task = ResearchTask(
            id=task_id,
            query=query,
            metadata=metadata or {}
        )
        self.tasks.append(task)
        self.active_task_id = task_id
        self.updated_at = datetime.now()
        return task
    
    def update_task_status(self, task_id: str, status: TaskStatus, error_message: Optional[str] = None) -> None:
        """Update the status of a research task."""
        for task in self.tasks:
            if task.id == task_id:
                task.status = status
                task.updated_at = datetime.now()
                if error_message:
                    task.error_message = error_message
                break
        self.updated_at = datetime.now()
    
    def add_task_result(self, task_id: str, result: Dict[str, Any]) -> None:
        """Add a result to a research task."""
        for task in self.tasks:
            if task.id == task_id:
                task.results.append(result)
                task.updated_at = datetime.now()
                break
        self.updated_at = datetime.now()
    
    def get_active_task(self) -> Optional[ResearchTask]:
        """Get the currently active research task."""
        if not self.active_task_id:
            return None
        
        for task in self.tasks:
            if task.id == self.active_task_id:
                return task
        return None
    
    def add_to_knowledge_base(self, key: str, value: Any) -> None:
        """Add information to the knowledge base."""
        self.knowledge_base[key] = value
        self.updated_at = datetime.now()
    
    def update_memory(self, memory_type: str, content: Dict[str, Any]) -> None:
        """Update agent memory."""
        if memory_type == "short_term":
            self.short_term_memory.append({
                **content,
                "timestamp": datetime.now().isoformat()
            })
            # Keep only last 50 short-term memories
            if len(self.short_term_memory) > 50:
                self.short_term_memory = self.short_term_memory[-50:]
        elif memory_type == "long_term":
            self.long_term_memory.update(content)
        
        self.updated_at = datetime.now()
    
    def record_tool_usage(self, tool_name: str, result: Any = None) -> None:
        """Record tool usage for analytics."""
        if tool_name not in self.tools_used:
            self.tools_used.append(tool_name)
        
        if result is not None:
            self.tool_results[f"{tool_name}_{datetime.now().timestamp()}"] = result
        
        self.updated_at = datetime.now()
    
    def update_metrics(self, success: bool, response_time: float) -> None:
        """Update performance metrics."""
        self.total_queries += 1
        if success:
            self.successful_queries += 1
        else:
            self.failed_queries += 1
        
        # Update average response time
        if self.total_queries == 1:
            self.average_response_time = response_time
        else:
            self.average_response_time = (
                (self.average_response_time * (self.total_queries - 1) + response_time) 
                / self.total_queries
            )
        
        self.updated_at = datetime.now()
    
    def get_conversation_context(self, max_messages: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation context."""
        return self.messages[-max_messages:] if self.messages else []
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Get a summary of research activities."""
        completed_tasks = [task for task in self.tasks if task.status == TaskStatus.COMPLETED]
        failed_tasks = [task for task in self.tasks if task.status == TaskStatus.FAILED]
        
        return {
            "total_tasks": len(self.tasks),
            "completed_tasks": len(completed_tasks),
            "failed_tasks": len(failed_tasks),
            "success_rate": len(completed_tasks) / len(self.tasks) if self.tasks else 0,
            "total_results": len(self.research_results),
            "knowledge_base_size": len(self.knowledge_base),
            "tools_used": list(set(self.tools_used)),
            "session_duration": (datetime.now() - self.created_at).total_seconds(),
            "average_response_time": self.average_response_time
        }
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
