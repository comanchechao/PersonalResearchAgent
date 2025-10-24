# Personal Research Agent

An AI-powered personal research assistant built with LangChain and designed to work with local LLMs via LM Studio. This agent helps you conduct thorough research, process documents, and maintain context across conversations.

## Features

- 🤖 **Local LLM Integration**: Works with LM Studio and your Qwen 2.5 Coder Instruct model
- 🔍 **Web Search**: Intelligent web search with multiple providers
- 📄 **Document Processing**: Handle PDFs, Word docs, web pages, and more
- 📝 **Smart Summarization**: AI-powered text summarization and analysis
- 🧠 **Memory & Context**: Remembers conversations and learns from interactions
- 👤 **Personalization**: Adapts to user preferences and research style
- 💬 **Interactive CLI**: Rich command-line interface for easy interaction
- 🔧 **Extensible**: Built with LangChain for easy customization

## Prerequisites

1. **LM Studio**: Download and install [LM Studio](https://lmstudio.ai/)
2. **Qwen 2.5 Coder Model**: Load the Qwen 2.5 Coder 7B Instruct model in LM Studio
3. **Python 3.9+**: Ensure you have Python 3.9 or higher installed

## Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd PersonalResearchAgent
   ```

2. **Install the package**:

   ```bash
   pip install -e .
   ```

3. **Set up LM Studio**:

   - Start LM Studio
   - Load the Qwen 2.5 Coder 7B Instruct model
   - Start the local server (usually on `localhost:1234`)

4. **Test the installation**:
   ```bash
   python examples/quick_test.py
   ```

## Quick Start

### Interactive Mode

Start an interactive research session:

```bash
python -m personal_research_agent.cli interactive
```

Or use the installed command:

```bash
research-agent interactive
```

### Single Query

Execute a single research query:

```bash
research-agent research "What are the latest developments in AI?"
```

### Python API

```python
import asyncio
from personal_research_agent import PersonalResearchAgent

async def main():
    # Initialize the agent
    agent = PersonalResearchAgent(user_id="your_user_id")

    # Conduct research
    result = await agent.research("What is quantum computing?")
    print(result['result']['output'])

    # Have a conversation
    response = await agent.chat("How does it differ from classical computing?")
    print(response)

asyncio.run(main())
```

## Configuration

The agent uses a comprehensive configuration system. You can modify settings via:

### Environment Variables

Create a `.env` file in your project root:

```env
# LLM Configuration
LLM__MODEL_NAME=qwen2.5-coder-7b-instruct
LLM__API_BASE=http://localhost:1234/v1
LLM__TEMPERATURE=0.7
LLM__MAX_TOKENS=4096

# Research Configuration
RESEARCH__MAX_SEARCH_RESULTS=10
RESEARCH__ENABLE_WEB_SEARCH=true
RESEARCH__ENABLE_DOCUMENT_PROCESSING=true

# Memory Configuration
MEMORY__ENABLE_MEMORY=true
MEMORY__ENABLE_PERSONALIZATION=true
```

### CLI Configuration

```bash
# View current configuration
research-agent config --show

# Update configuration
research-agent config --model "qwen2.5-coder-7b-instruct" --temperature 0.8
```

### Python Configuration

```python
from personal_research_agent.config import update_settings

update_settings(
    llm={"temperature": 0.8, "max_tokens": 2048},
    research={"max_search_results": 15}
)
```

## Usage Examples

### Basic Research

```python
import asyncio
from personal_research_agent import PersonalResearchAgent

async def research_example():
    agent = PersonalResearchAgent()

    # Research a topic
    result = await agent.research("Latest trends in renewable energy")
    print(result['result']['output'])

asyncio.run(research_example())
```

### Document Processing

```python
async def document_example():
    agent = PersonalResearchAgent()

    # Process a document
    result = await agent.research("Summarize this document: /path/to/document.pdf")
    print(result['result']['output'])
```

### Conversational Research

```python
async def conversation_example():
    agent = PersonalResearchAgent(user_id="researcher_1")

    # Series of related questions
    await agent.chat("I'm researching climate change impacts")
    response1 = await agent.chat("What are the main effects on agriculture?")
    response2 = await agent.chat("How about on coastal cities?")
    response3 = await agent.chat("What mitigation strategies are most effective?")
```

### User Preferences

```python
async def preferences_example():
    agent = PersonalResearchAgent(user_id="researcher_1")

    # Set preferences
    await agent.set_user_preference("research_style", "academic")
    await agent.set_user_preference("summary_length", "detailed")

    # The agent will adapt its responses based on these preferences
    result = await agent.research("Machine learning applications")
```

## CLI Commands

### Interactive Mode

```bash
research-agent interactive [--user-id USER] [--session-id SESSION] [--verbose]
```

### Single Query

```bash
research-agent research "Your query here" [--output file.txt] [--format json|text|markdown]
```

### Configuration

```bash
research-agent config [--show] [--model MODEL] [--temperature TEMP] [--api-base URL]
```

### Interactive Commands

When in interactive mode, you can use these commands:

- `/help` - Show available commands
- `/settings` - View current settings
- `/history` - View research history
- `/clear` - Clear current session
- `/stats` - Show session statistics
- `/quit` - Exit the application

## Architecture

The Personal Research Agent is built with a modular architecture:

```
src/personal_research_agent/
├── core/
│   ├── agent.py          # Main agent orchestration
│   ├── memory.py         # Memory and context management
│   └── state.py          # Agent state management
├── tools/
│   ├── web_search.py     # Web search capabilities
│   ├── document_processor.py  # Document processing
│   └── summarizer.py     # Text summarization
├── config.py             # Configuration management
└── cli.py               # Command-line interface
```

### Key Components

1. **PersonalResearchAgent**: Main orchestrator that coordinates tools and manages conversations
2. **AgentMemory**: Handles short-term and long-term memory, user preferences
3. **AgentState**: Manages conversation state and research tasks
4. **Tools**: Specialized tools for web search, document processing, and summarization

## Customization

### Adding New Tools

```python
from langchain.tools import BaseTool
from personal_research_agent.core.agent import PersonalResearchAgent

class CustomTool(BaseTool):
    name = "custom_tool"
    description = "Description of what this tool does"

    def _run(self, query: str) -> str:
        # Your tool implementation
        return "Tool result"

# Add to agent
agent = PersonalResearchAgent()
agent.tools.append(CustomTool())
```

### Custom Memory Providers

```python
from personal_research_agent.core.memory import MemoryProvider

class CustomMemoryProvider(MemoryProvider):
    async def store(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        # Your storage implementation
        pass

    async def retrieve(self, key: str) -> Optional[Any]:
        # Your retrieval implementation
        pass
```

## Troubleshooting

### Common Issues

1. **"Connection refused" errors**:

   - Ensure LM Studio is running
   - Check that the server is started on localhost:1234
   - Verify the model is loaded

2. **"Model not found" errors**:

   - Confirm Qwen 2.5 Coder model is loaded in LM Studio
   - Check the model name in configuration matches exactly

3. **Memory errors**:

   - Reduce `max_tokens` in configuration
   - Clear session data: `research-agent interactive` then `/clear`

4. **Import errors**:
   - Reinstall the package: `pip install -e .`
   - Check Python version (3.9+ required)

### Debug Mode

Enable verbose logging:

```bash
research-agent interactive --verbose
```

Or in Python:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Tips

1. **Optimize LLM Settings**:

   - Lower temperature (0.3-0.7) for focused responses
   - Adjust max_tokens based on your needs
   - Use appropriate chunk sizes for long documents

2. **Memory Management**:

   - Clear sessions periodically
   - Limit conversation history length
   - Use appropriate TTL for cached data

3. **Tool Usage**:
   - Limit web search results for faster processing
   - Process documents in chunks for large files
   - Use focused summarization for specific needs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [LangChain](https://langchain.com/) for LLM orchestration
- Uses [LM Studio](https://lmstudio.ai/) for local LLM hosting
- Powered by Qwen 2.5 Coder Instruct model
- CLI built with [Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/)

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Run the quick test: `python examples/quick_test.py`
3. Enable verbose logging for detailed error information
4. Open an issue on GitHub with detailed error logs

---

**Happy Researching! 🔬✨**
