import unittest
from src.agent import InterviewAgent, InterviewStage
from src.llm_client import LLMClient

class TestInterviewAgentIntegration(unittest.TestCase):
    def test_mock_flow(self):
        # Use actual LLMClient in mock mode
        client = LLMClient(mock=True)
        agent = InterviewAgent(client)

        greeting = agent.start()
        self.assertIn("Hello", greeting)

        # Test Role Selection
        response = agent.process_input("I want to be a software engineer")
        self.assertIn("Software Engineer", response)
        self.assertEqual(agent.stage, InterviewStage.INTERVIEW)
        self.assertEqual(agent.role, "Software Engineer")

        # Test Interview Question
        response = agent.process_input("I solve problems using Python.")
        self.assertIn("Mock response", response) # Default mock response for generic input

        # Test End Interview - wait, default mock response doesn't know how to end interview
        # because the mock logic in LLMClient is very simple.
        # But let's check if the agent logic handles the request to end even if LLM returns generic text.
        # Actually agent.end_interview() is called if user says "end interview".

        response = agent.process_input("end interview")
        self.assertEqual(agent.stage, InterviewStage.FINISHED)

if __name__ == '__main__':
    unittest.main()
