# Interview Practice Partner

An AI agent designed to help users prepare for job interviews by conducting mock interviews for specific roles.

## Features

*   **Role-Specific Interviews**: Practices for roles like Sales, Engineer, Retail Associate, etc.
*   **Voice & Chat Modes**: Choose between typing or speaking your answers.
*   **Role Selection**: Select from predefined roles or enter a custom one via the sidebar.
*   **Dynamic Follow-up Questions**: Using LLM to ask relevant follow-up questions.
*   **Post-Interview Feedback**: Detailed feedback on communication and technical knowledge.
*   **Web Interface**: A modern chat interface using `Streamlit`.

## Design Decisions

*   **Architecture**: The application is split into `InterviewAgent` (logic), `LLMClient` (AI interface), and `app.py` (Web UI). This separation allows for easier testing and modularity.
*   **State Management**: The agent uses an `Enum` (`InterviewStage`) to track the conversation state (`ROLE_SELECTION`, `INTERVIEW`, `FEEDBACK`). This ensures the agent knows when to switch contexts.
*   **Prompt Engineering**:
    *   **Role Selection**: The agent asks the LLM to extract the role and confirm it ("Role Confirmed: ...") to deterministically transition to the interview stage.
    *   **Interview**: The system prompt is updated with the specific role to enforce the persona.
    *   **Feedback**: A dedicated prompt triggers a comprehensive review of the conversation history.
*   **Mock Mode**: To facilitate testing and usage without an API key, a Mock mode is implemented in `LLMClient`.

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Set your API Keys (optional). The app checks for OpenAI first, then falls back to Gemini if configured, and finally Mock mode.

    Create a `.env` file in the project root or export variables:
    ```bash
    export OPENAI_API_KEY="your-openai-key"
    export GEMINI_API_KEY="your-gemini-key"
    ```

    *Windows (Command Prompt):*
    ```cmd
    set OPENAI_API_KEY=your-openai-key
    set GEMINI_API_KEY=your-gemini-key
    ```

    *Windows (PowerShell):*
    ```powershell
    $env:OPENAI_API_KEY="your-openai-key"
    $env:GEMINI_API_KEY="your-gemini-key"
    ```

    *   **OpenAI**: Used for GPT-4o and Whisper.
    *   **Gemini**: Used as a fallback if OpenAI quota is exceeded or key is missing.

2.  Run the application:
    ```bash
    streamlit run src/app.py
    ```

## Testing

Run unit tests (ensure `src` is in your PYTHONPATH):
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python -m unittest discover tests
```
