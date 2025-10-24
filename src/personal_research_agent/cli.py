"""
Command-line interface for Personal Research Agent.
Provides interactive and batch modes for research tasks.
"""

import asyncio
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
import logging

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
import click

from .core.agent import PersonalResearchAgent
from .config import get_settings, update_settings


# Initialize Rich console
console = Console()
app = typer.Typer(help="Personal Research Agent - AI-powered research assistant")


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(get_settings().logs_dir / "research_agent.log"),
            logging.StreamHandler() if verbose else logging.NullHandler()
        ]
    )


@app.command()
def interactive(
    user_id: Optional[str] = typer.Option(None, "--user-id", "-u", help="User ID for personalization"),
    session_id: Optional[str] = typer.Option(None, "--session-id", "-s", help="Session ID to resume"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Start interactive research session."""
    setup_logging(verbose)
    
    console.print(Panel.fit(
        "[bold blue]Personal Research Agent[/bold blue]\n"
        "AI-powered research assistant using LangChain and local LLMs\n\n"
        "Commands:\n"
        "• Type your research questions naturally\n"
        "• '/help' - Show available commands\n"
        "• '/settings' - View/modify settings\n"
        "• '/history' - View research history\n"
        "• '/clear' - Clear current session\n"
        "• '/quit' or Ctrl+C - Exit",
        title="Welcome"
    ))
    
    asyncio.run(_interactive_session(user_id, session_id))


async def _interactive_session(user_id: Optional[str], session_id: Optional[str]) -> None:
    """Run interactive research session."""
    try:
        # Initialize agent
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Initializing research agent...", total=None)
            agent = PersonalResearchAgent(session_id=session_id, user_id=user_id)
            progress.update(task, description="Agent ready!")
        
        console.print(f"[green]✓[/green] Research agent initialized (Session: {agent.session_id[:8]}...)")
        
        if user_id:
            console.print(f"[blue]ℹ[/blue] Personalized for user: {user_id}")
        
        # Main interaction loop
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold cyan]Research Query[/bold cyan]", default="").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if await _handle_command(user_input, agent):
                        break  # Exit if quit command
                    continue
                
                # Process research query
                await _process_research_query(agent, user_input)
                
            except KeyboardInterrupt:
                if Confirm.ask("\n[yellow]Are you sure you want to exit?[/yellow]"):
                    break
            except Exception as e:
                console.print(f"[red]Error:[/red] {str(e)}")
                if Confirm.ask("Continue session?", default=True):
                    continue
                else:
                    break
    
    except Exception as e:
        console.print(f"[red]Failed to initialize agent:[/red] {str(e)}")
        console.print("[yellow]Make sure LM Studio is running on localhost:1234[/yellow]")
        sys.exit(1)
    
    console.print("\n[green]Thank you for using Personal Research Agent![/green]")


async def _handle_command(command: str, agent: PersonalResearchAgent) -> bool:
    """Handle CLI commands. Returns True if should exit."""
    command = command.lower().strip()
    
    if command in ['/quit', '/exit', '/q']:
        return True
    
    elif command == '/help':
        _show_help()
    
    elif command == '/settings':
        await _show_settings(agent)
    
    elif command == '/history':
        await _show_history(agent)
    
    elif command == '/clear':
        if Confirm.ask("Clear current session?"):
            await agent.clear_session()
            console.print("[green]✓[/green] Session cleared")
    
    elif command == '/stats':
        await _show_stats(agent)
    
    else:
        console.print(f"[red]Unknown command:[/red] {command}")
        console.print("Type '/help' for available commands")
    
    return False


def _show_help() -> None:
    """Show help information."""
    help_text = """
[bold]Available Commands:[/bold]

[cyan]/help[/cyan] - Show this help message
[cyan]/settings[/cyan] - View and modify agent settings
[cyan]/history[/cyan] - View research history
[cyan]/clear[/cyan] - Clear current session
[cyan]/stats[/cyan] - Show session statistics
[cyan]/quit[/cyan] - Exit the application

[bold]Research Tips:[/bold]

• Ask specific questions for better results
• Use natural language - the agent understands context
• Request summaries of long documents
• Ask for comparisons between different sources
• The agent remembers previous conversations in the session

[bold]Examples:[/bold]

• "What are the latest developments in AI research?"
• "Summarize this document: /path/to/document.pdf"
• "Compare the pros and cons of renewable energy sources"
• "Find recent news about climate change policies"
"""
    console.print(Panel(help_text, title="Help"))


async def _show_settings(agent: PersonalResearchAgent) -> None:
    """Show current settings."""
    settings = get_settings()
    session_info = agent.get_session_info()
    
    table = Table(title="Current Settings")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("LLM Model", settings.llm.model_name)
    table.add_row("LLM Provider", settings.llm.provider)
    table.add_row("API Base", settings.llm.api_base or "Not set")
    table.add_row("Temperature", str(settings.llm.temperature))
    table.add_row("Max Tokens", str(settings.llm.max_tokens))
    table.add_row("Session ID", session_info["session_id"])
    table.add_row("User ID", session_info["user_id"] or "Not set")
    table.add_row("Total Queries", str(session_info["total_queries"]))
    
    console.print(table)


async def _show_history(agent: PersonalResearchAgent) -> None:
    """Show research history."""
    try:
        memory_stats = await agent.get_memory_stats()
        
        table = Table(title="Session History")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Conversation Turns", str(memory_stats.get("conversation_turns", 0)))
        table.add_row("Vector Store Documents", str(memory_stats.get("vector_store_documents", "N/A")))
        table.add_row("Cached Preferences", str(memory_stats.get("user_preferences_cached", 0)))
        
        console.print(table)
        
        # Show recent conversation if available
        history = await agent.memory.get_conversation_history(limit=5)
        if history:
            console.print("\n[bold]Recent Conversations:[/bold]")
            for i, turn in enumerate(history[-3:], 1):  # Show last 3 turns
                console.print(f"\n[cyan]Q{i}:[/cyan] {turn['human_message'][:100]}...")
                console.print(f"[green]A{i}:[/green] {turn['ai_message'][:100]}...")
    
    except Exception as e:
        console.print(f"[red]Error retrieving history:[/red] {str(e)}")


async def _show_stats(agent: PersonalResearchAgent) -> None:
    """Show session statistics."""
    try:
        summary = await agent.get_research_summary()
        
        table = Table(title="Session Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Tasks", str(summary.get("total_tasks", 0)))
        table.add_row("Completed Tasks", str(summary.get("completed_tasks", 0)))
        table.add_row("Success Rate", f"{summary.get('success_rate', 0):.1%}")
        table.add_row("Total Results", str(summary.get("total_results", 0)))
        table.add_row("Knowledge Base Size", str(summary.get("knowledge_base_size", 0)))
        table.add_row("Tools Used", ", ".join(summary.get("tools_used", [])))
        table.add_row("Session Duration", f"{summary.get('session_duration', 0):.1f}s")
        table.add_row("Avg Response Time", f"{summary.get('average_response_time', 0):.2f}s")
        
        console.print(table)
    
    except Exception as e:
        console.print(f"[red]Error retrieving stats:[/red] {str(e)}")


async def _process_research_query(agent: PersonalResearchAgent, query: str) -> None:
    """Process a research query and display results."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Researching...", total=None)
        
        try:
            # Execute research
            result = await agent.research(query)
            progress.update(task, description="Research completed!")
            
            # Display results
            if "error" in result:
                console.print(f"[red]Research failed:[/red] {result['error']}")
            else:
                # Show the research result
                agent_response = result["result"].get("output", "No response generated")
                
                console.print(Panel(
                    Markdown(agent_response),
                    title=f"Research Results for: {query[:50]}...",
                    border_style="green"
                ))
                
                # Show metadata
                console.print(f"\n[dim]Task ID: {result['task_id']} | "
                            f"Session: {result['session_id'][:8]}... | "
                            f"Time: {result['timestamp']}[/dim]")
        
        except Exception as e:
            progress.update(task, description="Research failed!")
            console.print(f"[red]Research error:[/red] {str(e)}")


@app.command()
def research(
    query: str = typer.Argument(..., help="Research query to execute"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file for results"),
    format: str = typer.Option("text", "--format", "-f", help="Output format: text, json, markdown"),
    user_id: Optional[str] = typer.Option(None, "--user-id", "-u", help="User ID for personalization"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Execute a single research query."""
    setup_logging(verbose)
    
    async def _research():
        try:
            # Initialize agent
            agent = PersonalResearchAgent(user_id=user_id)
            
            # Execute research
            result = await agent.research(query)
            
            # Format output
            if format == "json":
                output_content = json.dumps(result, indent=2, default=str)
            elif format == "markdown":
                output_content = f"# Research Results\n\n**Query:** {query}\n\n"
                if "error" in result:
                    output_content += f"**Error:** {result['error']}\n"
                else:
                    output_content += result["result"].get("output", "No results")
            else:  # text format
                if "error" in result:
                    output_content = f"Error: {result['error']}"
                else:
                    output_content = result["result"].get("output", "No results")
            
            # Output results
            if output:
                Path(output).write_text(output_content, encoding='utf-8')
                console.print(f"[green]✓[/green] Results saved to: {output}")
            else:
                console.print(output_content)
        
        except Exception as e:
            console.print(f"[red]Research failed:[/red] {str(e)}")
            sys.exit(1)
    
    asyncio.run(_research())


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    model: Optional[str] = typer.Option(None, "--model", help="Set LLM model name"),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="Set LLM temperature"),
    api_base: Optional[str] = typer.Option(None, "--api-base", help="Set API base URL"),
) -> None:
    """Manage configuration settings."""
    settings = get_settings()
    
    if show:
        console.print(Panel(
            f"[cyan]LLM Model:[/cyan] {settings.llm.model_name}\n"
            f"[cyan]Provider:[/cyan] {settings.llm.provider}\n"
            f"[cyan]API Base:[/cyan] {settings.llm.api_base}\n"
            f"[cyan]Temperature:[/cyan] {settings.llm.temperature}\n"
            f"[cyan]Max Tokens:[/cyan] {settings.llm.max_tokens}\n"
            f"[cyan]Data Directory:[/cyan] {settings.data_dir}",
            title="Current Configuration"
        ))
        return
    
    # Update settings
    updates = {}
    if model:
        updates["llm"] = {"model_name": model}
    if temperature is not None:
        updates.setdefault("llm", {})["temperature"] = temperature
    if api_base:
        updates.setdefault("llm", {})["api_base"] = api_base
    
    if updates:
        update_settings(**updates)
        console.print("[green]✓[/green] Configuration updated")
    else:
        console.print("[yellow]No configuration changes specified[/yellow]")


def main() -> None:
    """Main CLI entry point."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
