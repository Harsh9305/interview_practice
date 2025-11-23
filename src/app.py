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
            st.sidebar.warning("No OPENAI_API_KEY found. Running in Mock Mode.")
            mock_mode = True

        client = LLMClient(api_key=api_key, mock=mock_mode)
        st.session_state.client = client # Store client separately for audio functions
        st.session_state.agent = InterviewAgent(client)
        st.session_state.started = False
        st.session_state.last_audio_response = None

def display_chat_history():
    if "agent" in st.session_state:
        for message in st.session_state.agent.history:
            if message["role"] == "system":
                continue
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

def process_input(prompt, agent, client, mode):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from agent
    with st.spinner("Thinking..."):
        response = agent.process_input(prompt)

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(response)

        # Generate audio if in Voice mode
        if mode == "Voice":
             with st.spinner("Generating audio..."):
                 audio_path = client.text_to_speech(response)
                 if audio_path:
                     st.session_state.last_audio_response = audio_path

    # Check if finished
    if agent.stage == InterviewStage.FINISHED:
         st.success("Interview Finished! Review the feedback above.")
         if st.button("Restart Interview"):
             del st.session_state.agent
             st.session_state.started = False
             st.session_state.last_audio_response = None
             st.rerun()

def main():
    st.set_page_config(page_title="Interview Practice Partner", page_icon="👔")
    st.title("Interview Practice Partner 👔")

    initialize_session_state()

    # Sidebar for Setup
    with st.sidebar:
        st.header("Setup")

        mode = st.radio("Interaction Mode", ["Chat", "Voice"])

        # Role Selection
        roles = ["Software Engineer", "Product Manager", "Sales Representative", "Marketing Specialist", "Data Scientist", "HR Manager", "Other"]
        selected_role = st.selectbox("Select Role", roles)

        if selected_role == "Other":
            custom_role = st.text_input("Enter Role Name")
            if custom_role:
                role = custom_role
            else:
                role = None
        else:
            role = selected_role

        if not st.session_state.started:
            if st.button("Start Interview"):
                if role:
                    st.session_state.agent.start(role=role)
                    st.session_state.started = True
                    st.session_state.last_audio_response = None
                    st.rerun()
                else:
                    st.error("Please select or enter a role.")
        else:
            if st.button("Reset Interview"):
                del st.session_state.agent
                st.session_state.started = False
                st.session_state.last_audio_response = None
                st.rerun()

    agent = st.session_state.agent
    client = st.session_state.client

    if not st.session_state.started:
        st.info("Please select a role and click 'Start Interview' in the sidebar.")
        return

    display_chat_history()

    # Play last audio response if available
    if st.session_state.get("last_audio_response"):
        st.audio(st.session_state.last_audio_response, autoplay=True)

    # Input Handling
    if mode == "Chat":
        if prompt := st.chat_input("Type your response here..."):
            process_input(prompt, agent, client, mode)
            # Force rerun to update UI and play audio if any
            st.rerun()

    elif mode == "Voice":
        # Using st.audio_input (requires Streamlit >= 1.39)
        audio_value = st.audio_input("Record your answer")

        if audio_value:
            current_audio_bytes = audio_value.getvalue()

            if st.session_state.get("processed_audio") != current_audio_bytes:
                with st.spinner("Transcribing..."):
                     transcription = client.transcribe_audio(audio_value)

                if transcription:
                    process_input(transcription, agent, client, mode)
                    st.session_state.processed_audio = current_audio_bytes
                    st.rerun() # Rerun to update history display cleanly and play audio

if __name__ == "__main__":
    main()
