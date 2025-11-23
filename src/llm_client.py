import os
from openai import OpenAI
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from gtts import gTTS
import tempfile

load_dotenv()

class LLMClient:
    def __init__(self, api_key: Optional[str] = None, mock: bool = False):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.mock = mock
        self.client = None

        if not self.mock and self.api_key:
            self.client = OpenAI(api_key=self.api_key)

        if not self.api_key and not self.mock:
            print("Warning: No API Key found. Switch to mock mode or provide key.")
            self.mock = True

    def get_response(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        Get a response from the LLM or a mock response.
        messages: list of dicts with 'role' and 'content'
        """
        if self.mock:
            return self._get_mock_response(messages)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o", # Default to a strong model
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error communicating with LLM: {str(e)}"

    def transcribe_audio(self, audio_file) -> str:
        """
        Transcribes audio file to text using OpenAI Whisper or Mock.
        audio_file: file-like object or path
        """
        if self.mock:
            return "This is a mock transcription of the audio."

        try:
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            return transcription.text
        except Exception as e:
            return f"Error transcribing audio: {str(e)}"

    def text_to_speech(self, text: str) -> str:
        """
        Converts text to speech. Returns path to the audio file.
        Uses OpenAI TTS if available, otherwise gTTS.
        """
        try:
            # Create a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                output_path = fp.name

            if not self.mock and self.client:
                try:
                    response = self.client.audio.speech.create(
                        model="tts-1",
                        voice="alloy",
                        input=text
                    )
                    response.stream_to_file(output_path)
                    return output_path
                except Exception:
                    # Fallback to gTTS if OpenAI fails (e.g. quota or model issue)
                    pass

            # Fallback or Mock
            tts = gTTS(text=text, lang='en')
            tts.save(output_path)
            return output_path

        except Exception as e:
             print(f"Error in text_to_speech: {e}")
             return None

    def _get_mock_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a mock response based on the last user message.
        This is for testing without an API key.
        """
        # When extracting role, the last message is actually the system prompt asking for clarification/confirmation,
        # but we need to look at the user message before that to know the role.
        # Or better, just look at the whole conversation history for keywords.

        # For simplicity in this mock, we concatenate all user messages or look for key terms in the last user input.

        # In Agent._handle_role_selection, it appends a system message at the end asking to confirm role.
        # So messages[-1] is that system message.
        # messages[-2] is the user input (if history order is preserved as expected).

        user_inputs = [m['content'].lower() for m in messages if m['role'] == 'user']
        last_user_input = user_inputs[-1] if user_inputs else ""

        if "sales" in last_user_input:
            return "Role Confirmed: Sales Representative\nExcellent choice. I will be interviewing you for a Sales position. Let's begin. \n\nCan you tell me about a time you had to sell a difficult product?"

        if "engineer" in last_user_input:
             return "Role Confirmed: Software Engineer\nExcellent choice. I will be interviewing you for a Software Engineer position. Let's begin. \n\nCan you describe a challenging technical problem you solved recently?"

        return "That's an interesting point. Could you elaborate on that? (Mock response)"
