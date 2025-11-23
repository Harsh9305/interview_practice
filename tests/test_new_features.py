import unittest
from src.agent import InterviewAgent, InterviewStage
from src.llm_client import LLMClient

class TestInterviewAgentNewFeatures(unittest.TestCase):
    def test_start_with_role(self):
        client = LLMClient(mock=True)
        agent = InterviewAgent(client)

        # Start with specific role
        agent.start(role="Data Scientist")

        self.assertEqual(agent.role, "Data Scientist")
        self.assertEqual(agent.stage, InterviewStage.INTERVIEW)
        self.assertTrue(len(agent.history) > 0)
        # Verify system prompt contains role
        self.assertIn("Data Scientist", agent.history[0]['content'])

    def test_start_without_role(self):
        client = LLMClient(mock=True)
        agent = InterviewAgent(client)

        agent.start()
        self.assertEqual(agent.stage, InterviewStage.ROLE_SELECTION)
        self.assertIsNone(agent.role)

if __name__ == '__main__':
    unittest.main()
