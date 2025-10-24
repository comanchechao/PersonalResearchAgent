"""
Quick test script to verify Personal Research Agent setup.
Run this to ensure everything is working correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from personal_research_agent import PersonalResearchAgent
from personal_research_agent.config import get_settings


async def test_agent_initialization():
    """Test agent initialization."""
    print("Testing agent initialization...")
    try:
        agent = PersonalResearchAgent()
        print(f"[OK] Agent initialized successfully")
        print(f"  Session ID: {agent.session_id}")
        return agent
    except Exception as e:
        print(f"[ERROR] Agent initialization failed: {e}")
        return None


async def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")
    try:
        settings = get_settings()
        print(f"[OK] Configuration loaded")
        print(f"  LLM Model: {settings.llm.model_name}")
        print(f"  API Base: {settings.llm.api_base}")
        print(f"  Temperature: {settings.llm.temperature}")
        return True
    except Exception as e:
        print(f"[ERROR] Configuration failed: {e}")
        return False


async def test_simple_chat():
    """Test simple chat functionality."""
    print("\nTesting simple chat...")
    try:
        agent = PersonalResearchAgent()
        response = await agent.chat("Hello, can you help me with research?")
        print(f"[OK] Chat test successful")
        print(f"  Response length: {len(response)} characters")
        print(f"  Response preview: {response[:100]}...")
        return True
    except Exception as e:
        print(f"[ERROR] Chat test failed: {e}")
        print("  Make sure LM Studio is running on localhost:1234")
        return False


async def test_research_functionality():
    """Test basic research functionality."""
    print("\nTesting research functionality...")
    try:
        agent = PersonalResearchAgent()
        result = await agent.research("What is artificial intelligence?")
        
        if "error" in result:
            print(f"[ERROR] Research failed: {result['error']}")
            return False
        else:
            print(f"[OK] Research test successful")
            print(f"  Task ID: {result['task_id']}")
            print(f"  Response preview: {result['result']['output'][:100]}...")
            return True
    except Exception as e:
        print(f"[ERROR] Research test failed: {e}")
        return False


async def test_tools():
    """Test individual tools."""
    print("\nTesting tools...")
    
    # Test web search tool
    try:
        from personal_research_agent.tools import WebSearchTool
        search_tool = WebSearchTool()
        print("[OK] WebSearchTool imported successfully")
    except Exception as e:
        print(f"[ERROR] WebSearchTool failed: {e}")
    
    # Test document processor
    try:
        from personal_research_agent.tools import DocumentProcessor
        doc_tool = DocumentProcessor()
        print("[OK] DocumentProcessor imported successfully")
    except Exception as e:
        print(f"[ERROR] DocumentProcessor failed: {e}")
    
    # Test summarizer
    try:
        from personal_research_agent.tools import SummarizerTool
        summary_tool = SummarizerTool()
        print("[OK] SummarizerTool imported successfully")
    except Exception as e:
        print(f"[ERROR] SummarizerTool failed: {e}")


async def main():
    """Run all tests."""
    print("Personal Research Agent - Quick Test")
    print("=" * 40)
    
    # Test configuration first
    config_ok = await test_configuration()
    if not config_ok:
        print("\n[FAIL] Configuration test failed. Cannot continue.")
        return
    
    # Test agent initialization
    agent = await test_agent_initialization()
    if not agent:
        print("\n[FAIL] Agent initialization failed. Cannot continue.")
        return
    
    # Test tools
    await test_tools()
    
    # Test chat (requires LM Studio)
    chat_ok = await test_simple_chat()
    
    # Test research (requires LM Studio)
    research_ok = await test_research_functionality()
    
    print("\n" + "=" * 40)
    print("Test Summary:")
    print(f"Configuration: [OK]")
    print(f"Agent Init: [OK]")
    print(f"Tools: [OK]")
    print(f"Chat: {'[OK]' if chat_ok else '[ERROR]'}")
    print(f"Research: {'[OK]' if research_ok else '[ERROR]'}")
    
    if chat_ok and research_ok:
        print("\n[SUCCESS] All tests passed! Your setup is working correctly.")
        print("\nNext steps:")
        print("1. Run interactive mode: python -m personal_research_agent.cli interactive")
        print("2. Try the examples: python examples/basic_usage.py")
    else:
        print("\n[WARNING] Some tests failed. Check that:")
        print("1. LM Studio is running on localhost:1234")
        print("2. Qwen 2.5 Coder model is loaded in LM Studio")
        print("3. All dependencies are installed: pip install -e .")


if __name__ == "__main__":
    asyncio.run(main())
