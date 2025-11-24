import streamlit as st
import os
import warnings
# Suppress Google API warning about Python 3.10
warnings.filterwarnings("ignore", category=FutureWarning, module="google.api_core")

from llm_client import LLMClient
from agent import InterviewAgent, InterviewStage

def initialize_session_state():
    if "agent" not in st.session_state:
        # Check API Keys
        api_key = os.getenv("OPENAI_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")
        mock_mode = False

        if not api_key:
            if gemini_key:
                 st.sidebar.success("Using Gemini API.")
            else:
                 st.sidebar.warning("No API Keys found. Running in Mock Mode.")
                 mock_mode = True
        elif api_key:
             if gemini_key:
                 st.sidebar.info("Using OpenAI (Fallback to Gemini available).")
             else:
                 st.sidebar.info("Using OpenAI.")

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

        # Check if response indicates fallback
        if "Quota exceeded" in response or "mock transcription" in response:
            st.toast("⚠️ API Quota exceeded. Switched to Mock Mode.", icon="⚠️")

        # Check specifically for Gemini Error
        if hasattr(client, 'last_gemini_error') and client.last_gemini_error:
            st.error(f"⚠️ {client.last_gemini_error} Please check your .env file.")

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
    # We don't render buttons here because this runs inside the processing loop
    # and will be wiped on rerun. We handle the finished state in the main loop.
    pass

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

    # Check if finished to display feedback controls outside the processing loop
    if agent.stage == InterviewStage.FINISHED:
         st.success("Interview Finished! Review the feedback above.")
         if st.button("Restart Interview", key="restart_main"):
             del st.session_state.agent
             st.session_state.started = False
             st.session_state.last_audio_response = None
             st.rerun()
         return # Stop input handling if finished

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
                else:
                    if hasattr(client, 'last_gemini_error') and client.last_gemini_error:
                        st.error(f"Transcription failed: {client.last_gemini_error}")
                    else:
                        st.error("Transcription failed. Please try again or check your API keys.")
                    # Mark as processed to prevent infinite retry loop on the same audio
                    st.session_state.processed_audio = current_audio_bytes

if __name__ == "__main__":
    main()
