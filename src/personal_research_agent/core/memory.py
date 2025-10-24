"""
Memory management for Personal Research Agent.
Handles short-term and long-term memory, user preferences, and learning.
"""

from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import json
import hashlib
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.vectorstores import VectorStore
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ..config import get_settings


class MemoryProvider(ABC):
    """Abstract base class for memory providers."""
    
    @abstractmethod
    async def store(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store a value with optional TTL."""
        pass
    
    @abstractmethod
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value by key."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a value by key."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        pass


class FileMemoryProvider(MemoryProvider):
    """File-based memory provider for local storage."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for a key."""
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.storage_path / f"{safe_key}.json"
    
    async def store(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store a value with optional TTL."""
        try:
            data = {
                "value": value,
                "stored_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(seconds=ttl)).isoformat() if ttl else None
            }
            
            file_path = self._get_file_path(key)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"Error storing memory: {e}")
            return False
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value by key."""
        try:
            file_path = self._get_file_path(key)
            if not file_path.exists():
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check expiration
            if data.get("expires_at"):
                expires_at = datetime.fromisoformat(data["expires_at"])
                if datetime.now() > expires_at:
                    await self.delete(key)
                    return None
            
            return data["value"]
        except Exception as e:
            print(f"Error retrieving memory: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete a value by key."""
        try:
            file_path = self._get_file_path(key)
            if file_path.exists():
                file_path.unlink()
            return True
        except Exception as e:
            print(f"Error deleting memory: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        file_path = self._get_file_path(key)
        return file_path.exists()


class AgentMemory:
    """
    Comprehensive memory management for the Personal Research Agent.
    Handles conversation memory, knowledge storage, and user preferences.
    """
    
    def __init__(self, session_id: str, user_id: Optional[str] = None):
        self.session_id = session_id
        self.user_id = user_id
        self.settings = get_settings()
        
        # Initialize memory providers
        self._init_memory_providers()
        
        # Initialize conversation memory
        self._init_conversation_memory()
        
        # Initialize vector store for semantic memory
        self._init_vector_memory()
        
        # Memory caches
        self._short_term_cache: Dict[str, Any] = {}
        self._user_preferences_cache: Dict[str, Any] = {}
    
    def _init_memory_providers(self) -> None:
        """Initialize memory providers."""
        memory_dir = self.settings.data_dir / "memory"
        self.memory_provider = FileMemoryProvider(memory_dir)
    
    def _init_conversation_memory(self) -> None:
        """Initialize conversation memory."""
        # Simple in-memory conversation storage
        self.conversation_messages: List[BaseMessage] = []
        self.conversation_summary: List[str] = []
    
    def _init_vector_memory(self) -> None:
        """Initialize vector store for semantic memory."""
        try:
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.settings.vector_db.embedding_model,
                model_kwargs={'device': 'cpu'}
            )
            
            # Initialize vector store
            vector_db_path = self.settings.data_dir / "vector_db" / self.session_id
            self.vector_store = Chroma(
                collection_name=f"{self.settings.vector_db.collection_name}_{self.session_id}",
                embedding_function=self.embeddings,
                persist_directory=str(vector_db_path)
            )
        except Exception as e:
            print(f"Warning: Could not initialize vector memory: {e}")
            self.vector_store = None
    
    async def add_conversation_turn(self, human_message: str, ai_message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a conversation turn to memory."""
        # Add to in-memory conversation storage
        self.conversation_messages.append(HumanMessage(content=human_message))
        self.conversation_messages.append(AIMessage(content=ai_message))
        
        # Store in persistent memory
        turn_data = {
            "human_message": human_message,
            "ai_message": ai_message,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        key = f"conversation:{self.session_id}:{datetime.now().timestamp()}"
        await self.memory_provider.store(key, turn_data, ttl=self.settings.memory.short_term_memory_hours * 3600)
        
        # Add to vector store for semantic search
        if self.vector_store:
            try:
                combined_text = f"Human: {human_message}\nAI: {ai_message}"
                self.vector_store.add_texts(
                    texts=[combined_text],
                    metadatas=[{
                        "type": "conversation",
                        "session_id": self.session_id,
                        "timestamp": datetime.now().isoformat(),
                        **(metadata or {})
                    }]
                )
            except Exception as e:
                print(f"Warning: Could not add to vector store: {e}")
    
    async def store_research_result(self, query: str, result: Dict[str, Any]) -> None:
        """Store a research result in memory."""
        key = f"research:{self.session_id}:{hashlib.md5(query.encode()).hexdigest()}"
        
        research_data = {
            "query": query,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id
        }
        
        # Store in persistent memory
        await self.memory_provider.store(key, research_data, ttl=self.settings.memory.long_term_memory_days * 24 * 3600)
        
        # Add to vector store
        if self.vector_store:
            try:
                # Create searchable text from result
                searchable_text = f"Query: {query}\n"
                if isinstance(result, dict):
                    if "summary" in result:
                        searchable_text += f"Summary: {result['summary']}\n"
                    if "content" in result:
                        searchable_text += f"Content: {str(result['content'])[:1000]}\n"
                
                self.vector_store.add_texts(
                    texts=[searchable_text],
                    metadatas=[{
                        "type": "research_result",
                        "query": query,
                        "session_id": self.session_id,
                        "timestamp": datetime.now().isoformat()
                    }]
                )
            except Exception as e:
                print(f"Warning: Could not add research result to vector store: {e}")
    
    async def store_user_preference(self, key: str, value: Any) -> None:
        """Store a user preference."""
        if not self.user_id:
            return
        
        pref_key = f"user_pref:{self.user_id}:{key}"
        await self.memory_provider.store(pref_key, value)
        
        # Update cache
        self._user_preferences_cache[key] = value
    
    async def get_user_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference."""
        if not self.user_id:
            return default
        
        # Check cache first
        if key in self._user_preferences_cache:
            return self._user_preferences_cache[key]
        
        # Retrieve from storage
        pref_key = f"user_pref:{self.user_id}:{key}"
        value = await self.memory_provider.retrieve(pref_key)
        
        if value is not None:
            self._user_preferences_cache[key] = value
            return value
        
        return default
    
    async def search_memory(self, query: str, memory_type: str = "all", limit: int = 5) -> List[Dict[str, Any]]:
        """Search memory using semantic similarity."""
        if not self.vector_store:
            return []
        
        try:
            # Prepare filter
            filter_dict = {"session_id": self.session_id}
            if memory_type != "all":
                filter_dict["type"] = memory_type
            
            # Perform similarity search
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=limit,
                filter=filter_dict
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score
                })
            
            return formatted_results
        except Exception as e:
            print(f"Error searching memory: {e}")
            return []
    
    async def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history."""
        # Get from in-memory conversation storage
        messages = self.conversation_messages[-limit*2:]  # *2 because each turn has 2 messages
        
        history = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                human_msg = messages[i]
                ai_msg = messages[i + 1]
                history.append({
                    "human_message": human_msg.content if hasattr(human_msg, 'content') else str(human_msg),
                    "ai_message": ai_msg.content if hasattr(ai_msg, 'content') else str(ai_msg),
                    "timestamp": datetime.now().isoformat()
                })
        
        return history[-limit:]  # Return last 'limit' turns
    
    async def get_research_history(self, query: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get research history, optionally filtered by query similarity."""
        if query and self.vector_store:
            # Use semantic search
            results = await self.search_memory(query, memory_type="research_result", limit=limit)
            return [result["metadata"] for result in results]
        else:
            # Return recent research results
            # This would require iterating through stored keys, simplified for now
            return []
    
    async def clear_session_memory(self) -> None:
        """Clear memory for the current session."""
        # Clear conversation messages
        self.conversation_messages.clear()
        
        # Clear caches
        self._short_term_cache.clear()
        
        # Note: We don't clear user preferences as they persist across sessions
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        stats = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "conversation_turns": len(self.conversation_messages) // 2,
            "short_term_cache_size": len(self._short_term_cache),
            "user_preferences_cached": len(self._user_preferences_cache),
            "vector_store_available": self.vector_store is not None
        }
        
        if self.vector_store:
            try:
                # Get collection stats if available
                collection = self.vector_store._collection
                stats["vector_store_documents"] = collection.count()
            except:
                stats["vector_store_documents"] = "unknown"
        
        return stats
    
    def get_langchain_memory(self) -> List[BaseMessage]:
        """Get the LangChain memory object for use in chains."""
        return self.conversation_messages
