import unittest
from unittest.mock import MagicMock
from src.llm_client import LLMClient

class TestLLMClientFallback(unittest.TestCase):
    def test_quota_fallback_chat(self):
        client = LLMClient(api_key="fake-key", mock=False)
        # Mock the internal OpenAI client
        client.client = MagicMock()
        # Simulate 429 error
        client.client.chat.completions.create.side_effect = Exception("Error code: 429 - insufficient_quota")

        response = client.get_response([{"role": "user", "content": "I want to be a Software Engineer"}])

        # Should switch to mock response
        self.assertIn("Software Engineer", response)
        # It won't necessarily contain "Mock response" text if it matches a specific role like Engineer
        # But it should be one of the canned responses
        self.assertIn("Excellent choice", response)

    def test_quota_fallback_audio(self):
        client = LLMClient(api_key="fake-key", mock=False)
        client.client = MagicMock()
        client.client.audio.transcriptions.create.side_effect = Exception("Error code: 429 - insufficient_quota")

        response = client.transcribe_audio("fake_audio_file")

        self.assertIn("mock transcription", response)

if __name__ == '__main__':
    unittest.main()
