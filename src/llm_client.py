import os
try:
    import google.generativeai as genai
except ImportError:
    genai = None
from openai import OpenAI
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from gtts import gTTS
import tempfile
import pathlib

load_dotenv()

class LLMClient:
    def __init__(self, api_key: Optional[str] = None, mock: bool = False):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.mock = mock
        self.client = None
        self.gemini_configured = False

        # Configure OpenAI
        if not self.mock and self.api_key:
            self.client = OpenAI(api_key=self.api_key)

        # Configure Gemini
        if self.gemini_key:
            if genai:
                try:
                    genai.configure(api_key=self.gemini_key)
                    self.gemini_configured = True
                except Exception as e:
                    print(f"Failed to configure Gemini: {e}")
            else:
                 print("Warning: google-generativeai package not found. Install it to use Gemini.")

        # Track last Gemini error for UI feedback
        self.last_gemini_error = None

        if not self.api_key and not self.gemini_configured and not self.mock:
            print("Warning: No API Keys found. Switch to mock mode or provide key.")
            self.mock = True

    def get_response(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        Get a response from the LLM (OpenAI -> Gemini -> Mock).
        messages: list of dicts with 'role' and 'content'
        """
        if self.mock:
            return self._get_mock_response(messages)

        # Try OpenAI
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            except Exception as e:
                # If quota error or other critical error, try Gemini
                print(f"OpenAI Error: {e}. Attempting fallback...")
                pass # Proceed to Gemini fallback logic

        # Try Gemini
        if self.gemini_configured:
            return self._get_gemini_response(messages)

        # Fallback to Mock
        print("All LLMs failed or not configured. Using Mock.")
        return self._get_mock_response(messages)

    def _get_gemini_response(self, messages: List[Dict[str, str]]) -> str:
        self.last_gemini_error = None

        # Try primary model then fallback
        models_to_try = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']

        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)

                # Construct a prompt from messages
                prompt_text = ""
                for msg in messages:
                    role = msg['role']
                    content = msg['content']
                    if role == "system":
                        prompt_text += f"System: {content}\n"
                    elif role == "user":
                        prompt_text += f"User: {content}\n"
                    elif role == "assistant":
                        prompt_text += f"Assistant: {content}\n"

                prompt_text += "Assistant: "

                response = model.generate_content(prompt_text)
                return response.text
            except Exception as e:
                print(f"Gemini Error ({model_name}): {e}")
                error_str = str(e)
                if "API_KEY_INVALID" in error_str or "API key not valid" in error_str:
                    self.last_gemini_error = "Gemini API Key is invalid."
                    # If key is invalid, no model will work, break loop
                    break

                # If 404/not found, loop to next model
                if "404" in error_str or "not found" in error_str:
                    continue

                # For other errors, maybe try next model?
                continue

        # If all failed
        return self._get_mock_response(messages)

    def transcribe_audio(self, audio_file) -> str:
        """
        Transcribes audio file to text using OpenAI Whisper -> Gemini -> Mock.
        audio_file: file-like object or path
        """
        if self.mock:
            return "This is a mock transcription of the audio."

        # Try OpenAI Whisper
        if self.client:
            try:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                return transcription.text
            except Exception as e:
                print(f"OpenAI Whisper Error: {e}. Attempting fallback...")

        # Try Gemini
        if self.gemini_configured:
            result = self._transcribe_audio_gemini(audio_file)
            if result:
                return result

        return None # Return None to indicate failure

    def _transcribe_audio_gemini(self, audio_file) -> Optional[str]:
        self.last_gemini_error = None

        # Models that support audio
        models_to_try = ['gemini-1.5-flash', 'gemini-1.5-pro']

        # Prepare audio data once
        try:
            # audio_file might be a BytesIO or a path
            if hasattr(audio_file, 'read'):
                if hasattr(audio_file, 'seek'):
                     audio_file.seek(0)
                audio_data = audio_file.read()
            else:
                with open(audio_file, 'rb') as f:
                    audio_data = f.read()
        except Exception as e:
            print(f"Error reading audio file: {e}")
            return None

        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)

                response = model.generate_content([
                    "Transcribe the following audio accurately.",
                    {
                        "mime_type": "audio/wav",
                        "data": audio_data
                    }
                ])
                return response.text
            except Exception as e:
                 print(f"Gemini Transcription Error ({model_name}): {e}")
                 error_str = str(e)
                 if "API_KEY_INVALID" in error_str or "API key not valid" in error_str:
                    self.last_gemini_error = "Gemini API Key is invalid."
                    return None

                 if "404" in error_str or "not found" in error_str:
                     continue

                 continue

        return None

    def text_to_speech(self, text: str) -> str:
        """
        Converts text to speech. Returns path to the audio file.
        Uses OpenAI TTS if available, otherwise gTTS.
        Gemini doesn't support TTS directly in this SDK version easily as OpenAI does.
        So we fallback to gTTS directly if OpenAI fails.
        """
        try:
            # Create a temp file
            # On Windows, we must close the file before opening it again for writing by external libs
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                output_path = fp.name

            # Now the file is closed, we can write to it by path

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
        user_inputs = [m['content'].lower() for m in messages if m['role'] == 'user']
        last_user_input = user_inputs[-1] if user_inputs else ""

        if "sales" in last_user_input:
            return "Role Confirmed: Sales Representative\nExcellent choice. I will be interviewing you for a Sales position. Let's begin. \n\nCan you tell me about a time you had to sell a difficult product?"

        if "engineer" in last_user_input:
             return "Role Confirmed: Software Engineer\nExcellent choice. I will be interviewing you for a Software Engineer position. Let's begin. \n\nCan you describe a challenging technical problem you solved recently?"

        return "That's an interesting point. Could you elaborate on that? (Mock response)"
