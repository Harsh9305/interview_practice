import unittest
from unittest.mock import MagicMock
from src.agent import InterviewAgent, InterviewStage
from src.llm_client import LLMClient

class TestInterviewAgent(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock(spec=LLMClient)
        self.agent = InterviewAgent(self.mock_llm)

    def test_initial_state(self):
        self.assertEqual(self.agent.stage, InterviewStage.ROLE_SELECTION)
        self.assertIsNone(self.agent.role)

    def test_start(self):
        greeting = self.agent.start()
        self.assertIn("Hello", greeting)
        self.assertEqual(len(self.agent.history), 2) # system + assistant

    def test_role_selection_success(self):
        self.agent.start()
        # Mock LLM response to simulate role confirmation
        self.mock_llm.get_response.return_value = "Role Confirmed: Software Engineer"

        response = self.agent.process_input("I want to be a software engineer")

        self.assertEqual(self.agent.stage, InterviewStage.INTERVIEW)
        self.assertEqual(self.agent.role, "Software Engineer")
        self.assertIn("Software Engineer", response)

    def test_role_selection_unclear(self):
        self.agent.start()
        # Mock LLM response to simulate unclear input
        self.mock_llm.get_response.return_value = "Could you please clarify which role you are interested in?"

        response = self.agent.process_input("I don't know")

        self.assertEqual(self.agent.stage, InterviewStage.ROLE_SELECTION)
        self.assertIsNone(self.agent.role)
        self.assertEqual(response, "Could you please clarify which role you are interested in?")

    def test_interview_flow(self):
        # Force state to interview
        self.agent.start()
        self.agent.stage = InterviewStage.INTERVIEW
        self.agent.role = "Tester"
        self.agent.history = [{"role": "system", "content": "Interview context"}]

        self.mock_llm.get_response.return_value = "What is your greatest weakness?"

        response = self.agent.process_input("My weakness is chocolate.")

        self.assertEqual(self.agent.question_count, 1)
        self.assertEqual(response, "What is your greatest weakness?")
        self.mock_llm.get_response.assert_called()

    def test_end_interview_trigger(self):
        self.agent.start()
        self.agent.stage = InterviewStage.INTERVIEW
        self.agent.role = "Tester"

        self.mock_llm.get_response.return_value = "Here is your feedback..."

        response = self.agent.process_input("end interview")

        # After end_interview, the stage transitions to FINISHED
        self.assertEqual(self.agent.stage, InterviewStage.FINISHED)
        self.assertIn("Here is your feedback", response)

if __name__ == '__main__':
    unittest.main()
