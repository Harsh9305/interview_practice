import os
import sys
from rich.console import Console
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.panel import Panel

from agent import InterviewAgent, InterviewStage
from llm_client import LLMClient

def main():
    console = Console()

    # Initialize LLM Client
    # Check for API Key in env, otherwise warn or use mock
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print(Panel("No OPENAI_API_KEY found in environment variables.\nUsing Mock LLM mode for demonstration.", title="Warning", style="bold yellow"))
        client = LLMClient(mock=True)
    else:
        client = LLMClient(api_key=api_key)

    agent = InterviewAgent(client)

    console.print(Panel("Welcome to the Interview Practice Partner", style="bold blue"))

    # Start conversation
    response = agent.start()
    console.print(Markdown(response))
    console.print()

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]")

            if user_input.lower() in ['exit', 'quit']:
                console.print("[bold red]Goodbye![/bold red]")
                break

            with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
                response = agent.process_input(user_input)

            console.print(Panel(Markdown(response), title="Interviewer", title_align="left", border_style="blue"))
            console.print()

            if agent.stage == InterviewStage.FINISHED:
                 break

        except KeyboardInterrupt:
            console.print("\n[bold red]Exiting...[/bold red]")
            break
        except Exception as e:
            console.print(f"[bold red]An error occurred: {e}[/bold red]")
            break

if __name__ == "__main__":
    main()
