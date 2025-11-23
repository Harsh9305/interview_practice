from enum import Enum, auto
from typing import List, Dict, Optional
from llm_client import LLMClient

class InterviewStage(Enum):
    ROLE_SELECTION = auto()
    INTERVIEW = auto()
    FEEDBACK = auto()
    FINISHED = auto()

class InterviewAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.history: List[Dict[str, str]] = []
        self.stage = InterviewStage.ROLE_SELECTION
        self.role: Optional[str] = None
        self.question_count = 0
        self.max_questions = 5  # Default max questions before suggesting feedback

    def start(self) -> str:
        """Initializes the conversation."""
        self.history = [
            {"role": "system", "content": "You are a helpful assistant designed to help users prepare for job interviews. Your goal is to facilitate a mock interview practice session."}
        ]
        greeting = "Hello! I'm your Interview Practice Partner. I can help you prepare for job interviews by conducting mock interviews and providing feedback.\n\nTo get started, please tell me what job role you would like to practice for (e.g., Software Engineer, Sales Associate, Retail Manager)."
        self.history.append({"role": "assistant", "content": greeting})
        return greeting

    def process_input(self, user_input: str) -> str:
        """Processes user input and returns the agent's response."""
        self.history.append({"role": "user", "content": user_input})

        if self.stage == InterviewStage.ROLE_SELECTION:
            return self._handle_role_selection(user_input)
        elif self.stage == InterviewStage.INTERVIEW:
            return self._handle_interview(user_input)
        elif self.stage == InterviewStage.FEEDBACK:
            # If we are already in feedback mode, usually we just end or answer questions about feedback.
            # But for now, let's just continue conversation if they ask something, or close.
             return self._generate_response()
        else:
            return "The interview session has finished. Please restart the application to practice again."

    def _generate_response(self) -> str:
        """Generates a generic response based on history."""
        response = self.llm_client.get_response(self.history)
        self.history.append({"role": "assistant", "content": response})
        return response

    def _handle_role_selection(self, user_input: str) -> str:
        # We rely on the LLM to confirm the role or ask for clarification.
        # However, we want to transition to INTERVIEW stage if a role is clearly identified.
        # Let's ask the LLM to extract the role or confirm it.

        # Update system prompt for role selection context
        messages = self.history.copy()
        messages.append({
            "role": "system",
            "content": "The user has provided input for the role they want to practice. If they specified a valid role, acknowledge it, set the context that you are now the interviewer for that role, and ask the first interview question. If the input is unclear, ask for clarification. Start your response with 'Role Confirmed: [Role Name]' if a role is found, otherwise just ask for clarification."
        })

        response = self.llm_client.get_response(messages)

        if "Role Confirmed:" in response:
            # Extract role and transition
            parts = response.split("Role Confirmed:")
            # Take the part after the key, split by newline to handle multiline responses,
            # and strip whitespace to get just the role name.
            role_part = parts[1].strip()
            if "\n" in role_part:
                self.role = role_part.split("\n")[0].strip()
            else:
                self.role = role_part

            # Clean up the response to show to the user (remove the internal flag if we want, or keep it)
            # Let's keep the response but maybe clean the flag if it looks robotic.
            # Actually, let's instruct LLM to just start.

            self.stage = InterviewStage.INTERVIEW

            # Re-orient the system prompt for the interview
            self.history = [
                {"role": "system", "content": f"You are an expert interviewer for a {self.role} position. Conduct a professional mock interview. Ask one question at a time. Wait for the user's response before asking the next one. Do not overwhelm the user. If the user's answer is brief or lacks detail, ask a follow-up question. If the user goes off-topic, gently bring them back to the interview. You can end the interview if the user asks to stop or if you have asked {self.max_questions} questions."}
            ]
            # We need to generate the first question now.
            # The previous response might have been "Role Confirmed: Engineer. Okay let's start..."
            # Let's just generate a fresh start message.

            start_message = f"Great! I will act as the interviewer for the {self.role} position. Let's begin.\n\nTell me a little bit about yourself and why you are interested in this role."
            self.history.append({"role": "assistant", "content": start_message})
            return start_message

        else:
            # Not confirmed, just return response
            self.history.append({"role": "assistant", "content": response})
            return response

    def _handle_interview(self, user_input: str) -> str:
        # Check for exit commands
        if any(cmd in user_input.lower() for cmd in ["end interview", "stop interview", "give me feedback", "finish"]):
            return self.end_interview()

        self.question_count += 1

        # If we reached max questions, suggest ending
        if self.question_count > self.max_questions:
             # Add a system hint to wrap up
             messages = self.history.copy()
             messages.append({
                 "role": "system",
                 "content": "The user has answered several questions. Suggest concluding the interview and moving to the feedback stage, or ask one final challenging question."
             })
        else:
             messages = self.history

        response = self.llm_client.get_response(messages)
        self.history.append({"role": "assistant", "content": response})
        return response

    def end_interview(self) -> str:
        """Ends the interview and provides feedback."""
        self.stage = InterviewStage.FEEDBACK

        feedback_prompt = "The interview is now over. Please provide detailed feedback on the user's performance based on the conversation history. Assess their communication skills, technical knowledge (if applicable), and relevance to the role. Be constructive and highlight both strengths and areas for improvement. Format the output clearly."

        self.history.append({"role": "system", "content": feedback_prompt})

        response = self.llm_client.get_response(self.history)
        self.history.append({"role": "assistant", "content": response})

        # Transition to FINISHED so the loop in main.py knows to exit (or prompt for exit)
        self.stage = InterviewStage.FINISHED
        return response
