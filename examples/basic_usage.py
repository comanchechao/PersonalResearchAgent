"""
Basic usage examples for Personal Research Agent.
Demonstrates core functionality and API usage.
"""

import asyncio
import logging
from pathlib import Path

from personal_research_agent import PersonalResearchAgent


async def basic_research_example():
    """Basic research query example."""
    print("=== Basic Research Example ===")
    
    # Initialize the agent
    agent = PersonalResearchAgent(user_id="example_user")
    
    # Simple research query
    query = "What are the latest developments in artificial intelligence?"
    print(f"Query: {query}")
    
    result = await agent.research(query)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Result: {result['result']['output']}")
        print(f"Task ID: {result['task_id']}")


async def conversation_example():
    """Conversational interaction example."""
    print("\n=== Conversation Example ===")
    
    agent = PersonalResearchAgent(user_id="example_user")
    
    # Series of related questions
    questions = [
        "What is machine learning?",
        "How does it differ from traditional programming?",
        "What are some practical applications?",
        "What are the current limitations?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        response = await agent.chat(question)
        print(f"A: {response[:200]}...")  # Truncate for display


async def document_processing_example():
    """Document processing example."""
    print("\n=== Document Processing Example ===")
    
    agent = PersonalResearchAgent()
    
    # Create a sample document
    sample_doc = """
    Artificial Intelligence and Machine Learning: A Comprehensive Overview
    
    Artificial Intelligence (AI) has emerged as one of the most transformative technologies
    of the 21st century. Machine Learning (ML), a subset of AI, enables computers to learn
    and improve from experience without being explicitly programmed.
    
    Key Applications:
    1. Natural Language Processing
    2. Computer Vision
    3. Autonomous Vehicles
    4. Healthcare Diagnostics
    5. Financial Trading
    
    Current Challenges:
    - Data Privacy and Security
    - Algorithmic Bias
    - Interpretability and Explainability
    - Computational Requirements
    
    The future of AI looks promising with continued advancements in deep learning,
    neural networks, and quantum computing integration.
    """
    
    # Process the document
    query = f"Please analyze and summarize this document: {sample_doc}"
    result = await agent.research(query)
    
    if "error" not in result:
        print("Document Analysis:")
        print(result['result']['output'])


async def memory_and_preferences_example():
    """Memory and user preferences example."""
    print("\n=== Memory and Preferences Example ===")
    
    agent = PersonalResearchAgent(user_id="example_user")
    
    # Set user preferences
    await agent.set_user_preference("research_style", "detailed")
    await agent.set_user_preference("preferred_sources", ["academic", "news"])
    await agent.set_user_preference("summary_length", "medium")
    
    # Get preferences
    style = await agent.get_user_preference("research_style")
    sources = await agent.get_user_preference("preferred_sources")
    
    print(f"Research Style: {style}")
    print(f"Preferred Sources: {sources}")
    
    # Research with context from memory
    await agent.chat("I'm interested in renewable energy research")
    await agent.chat("What are the latest solar panel technologies?")
    
    # Search history
    history = await agent.search_history("solar energy", limit=3)
    print(f"Found {len(history)} related items in history")


async def batch_research_example():
    """Batch research processing example."""
    print("\n=== Batch Research Example ===")
    
    agent = PersonalResearchAgent()
    
    # Multiple research queries
    queries = [
        "Current trends in renewable energy",
        "Impact of climate change on agriculture",
        "Advances in quantum computing",
        "Future of space exploration"
    ]
    
    results = []
    for query in queries:
        print(f"Researching: {query}")
        result = await agent.research(query)
        results.append({
            "query": query,
            "success": "error" not in result,
            "task_id": result.get("task_id")
        })
    
    # Summary
    successful = sum(1 for r in results if r["success"])
    print(f"\nBatch Results: {successful}/{len(queries)} successful")


async def session_management_example():
    """Session management and statistics example."""
    print("\n=== Session Management Example ===")
    
    agent = PersonalResearchAgent(user_id="example_user")
    
    # Perform some research
    await agent.research("What is blockchain technology?")
    await agent.chat("How does it work?")
    await agent.research("What are its applications?")
    
    # Get session info
    session_info = agent.get_session_info()
    print(f"Session ID: {session_info['session_id']}")
    print(f"Total Queries: {session_info['total_queries']}")
    print(f"Tools Used: {session_info['tools_used']}")
    
    # Get research summary
    summary = await agent.get_research_summary()
    print(f"Research Summary: {summary}")
    
    # Get memory statistics
    memory_stats = await agent.get_memory_stats()
    print(f"Memory Stats: {memory_stats}")


async def error_handling_example():
    """Error handling and recovery example."""
    print("\n=== Error Handling Example ===")
    
    agent = PersonalResearchAgent()
    
    # Test with various scenarios
    test_cases = [
        "",  # Empty query
        "x" * 10000,  # Very long query
        "What is the meaning of life?",  # Philosophical query
        "Process this non-existent file: /fake/path/file.pdf"  # Invalid file
    ]
    
    for i, query in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {query[:50]}...")
        try:
            result = await agent.research(query)
            if "error" in result:
                print(f"Handled error: {result['error']}")
            else:
                print("Success: Query processed normally")
        except Exception as e:
            print(f"Exception caught: {str(e)}")


async def main():
    """Run all examples."""
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    print("Personal Research Agent - Usage Examples")
    print("=" * 50)
    
    try:
        await basic_research_example()
        await conversation_example()
        await document_processing_example()
        await memory_and_preferences_example()
        await batch_research_example()
        await session_management_example()
        await error_handling_example()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nTo run the interactive CLI:")
        print("python -m personal_research_agent.cli interactive")
        
    except Exception as e:
        print(f"\nExample failed: {str(e)}")
        print("\nMake sure:")
        print("1. LM Studio is running on localhost:1234")
        print("2. Qwen 2.5 Coder model is loaded")
        print("3. All dependencies are installed")


if __name__ == "__main__":
    asyncio.run(main())
