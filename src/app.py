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

        # Play audio if in Voice mode
        if mode == "Voice":
             with st.spinner("Generating audio..."):
                 audio_path = client.text_to_speech(response)
                 if audio_path:
                     st.audio(audio_path, autoplay=True)

    # Check if finished
    if agent.stage == InterviewStage.FINISHED:
         st.success("Interview Finished! Review the feedback above.")
         if st.button("Restart Interview"):
             del st.session_state.agent
             st.session_state.started = False
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
                    st.rerun()
                else:
                    st.error("Please select or enter a role.")
        else:
            if st.button("Reset Interview"):
                del st.session_state.agent
                st.session_state.started = False
                st.rerun()

    agent = st.session_state.agent
    client = st.session_state.client

    if not st.session_state.started:
        st.info("Please select a role and click 'Start Interview' in the sidebar.")
        return

    display_chat_history()

    # Input Handling
    if mode == "Chat":
        if prompt := st.chat_input("Type your response here..."):
            process_input(prompt, agent, client, mode)

    elif mode == "Voice":
        # Using st.audio_input (requires Streamlit >= 1.39)
        audio_value = st.audio_input("Record your answer")

        if audio_value:
            # Check if we already processed this specific audio input
            # st.audio_input value persists until cleared.
            # We need to check if it's new.
            # A simple way is to check if the last message from user matches the transcription?
            # Or store a hash.
            # For now, let's assume Streamlit handles the rerun loop.

            # But wait, st.audio_input triggers rerun.
            # We need to ensure we don't re-process indefinitely.
            # But `process_input` appends to history, so next time we check history?
            # No, we need to transcribe first.

            with st.spinner("Transcribing..."):
                 transcription = client.transcribe_audio(audio_value)

            if transcription:
                # Check if this transcription is already the last user message to avoid loops?
                # Actually, `process_input` will append it.
                # But `audio_value` stays in widget state.
                # We need to know if we just processed it.

                # Let's use a session state flag for 'last_audio_processed'
                if "last_audio_id" not in st.session_state:
                    st.session_state.last_audio_id = None

                # audio_value is a BytesIO, we can't easily ID it unless we hash it or assume change.
                # Streamlit reruns when input changes.
                # If we haven't processed this input yet...

                # Actually, simple fix: check if transcription == last user message?
                # No, user might say same thing twice.

                # Let's rely on the fact that we are in a rerun caused by the input.
                # But we need to not re-run logic on subsequent reruns (e.g. other interactions).

                # Currently, if I speak, it reruns. I process.
                # Then if I click a button or something, it reruns again with same audio_value?
                # Yes.

                # So we must store the audio buffer or something to compare.

                current_audio_bytes = audio_value.getvalue()

                if st.session_state.get("processed_audio") != current_audio_bytes:
                    process_input(transcription, agent, client, mode)
                    st.session_state.processed_audio = current_audio_bytes
                    st.rerun() # Rerun to update history display cleanly

if __name__ == "__main__":
    main()
