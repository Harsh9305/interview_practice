import streamlit as st
import os
from llm_client import LLMClient
from agent import InterviewAgent, InterviewStage

def initialize_session_state():
    if "agent" not in st.session_state:
        # Check API Key
        api_key = os.getenv("OPENAI_API_KEY")
        mock_mode = False
        if not api_key:
            st.warning("No OPENAI_API_KEY found. Running in Mock Mode.")
            mock_mode = True

        client = LLMClient(api_key=api_key, mock=mock_mode)
        st.session_state.agent = InterviewAgent(client)

        # Start the agent
        initial_msg = st.session_state.agent.start()
        # The history in agent is [{"role": "system", ...}, {"role": "assistant", ...}]
        # We want to display the assistant's greeting.
        # But agent.history accumulates everything.
        # Streamlit chat history should probably mirror agent history or just be derived from it.
        # Let's just use agent.history for source of truth, but filter out system messages.

def display_chat_history():
    for message in st.session_state.agent.history:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main():
    st.set_page_config(page_title="Interview Practice Partner", page_icon="👔")
    st.title("Interview Practice Partner 👔")

    initialize_session_state()

    agent = st.session_state.agent

    display_chat_history()

    # Chat input
    if prompt := st.chat_input("Type your response here..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from agent
        with st.spinner("Thinking..."):
            response = agent.process_input(prompt)

        # Display assistant message
        with st.chat_message("assistant"):
            st.markdown(response)

        # Check if finished
        if agent.stage == InterviewStage.FINISHED:
             st.success("Interview Finished! Review the feedback above.")
             if st.button("Restart Interview"):
                 del st.session_state.agent
                 st.rerun()

if __name__ == "__main__":
    main()
