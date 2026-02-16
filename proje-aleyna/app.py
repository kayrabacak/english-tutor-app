import streamlit as st
import os
from openai import OpenAI
import google.generativeai as genai
import tempfile

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================

st.set_page_config(
    page_title="AI Fluent | English Tutor",
    page_icon="🇬🇧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Secrets (Hardcoded for now as per user request, but best practice is st.secrets)
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Initialize Clients
client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

# Gemini Model Config
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

system_instruction = """
You are a friendly, patient, and encouraging English tutor named 'Aleyna'. 
Your goal is to help the user practice speaking English.
- Correct grammar mistakes gently inside your response.
- Keep the conversation flowing by asking follow-up questions.
- Speak naturally, like a human friend, not a robot.
- Keep responses concise (3-5 sentences max) so the user can speak more.
"""

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config=generation_config,
    system_instruction=system_instruction,
)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def load_local_css(file_name):
    """Load local CSS file for custom styling."""
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file {file_name} not found. Please ensure 'assets/style.css' exists.")

def speech_to_text(audio_file_path):
    """Convert audio to text using OpenAI Whisper."""
    with open(audio_file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            language="en"
        )
    return transcript.text

def ask_gemini(chat_session, user_text):
    """Get response from Google Gemini."""
    response = chat_session.send_message(user_text)
    return response.text

def text_to_speech(text):
    """Convert text to speech using OpenAI TTS."""
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy", 
        input=text
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        response.stream_to_file(tmp_file.name)
        return tmp_file.name

# ==========================================
# 3. INITIALIZATION
# ==========================================

# Load CSS
load_local_css("assets/style.css")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

# ==========================================
# 4. SIDEBAR UI
# ==========================================

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("Control your learning session.")
    
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_session = model.start_chat(history=[])
        st.rerun()
        
    st.markdown("---")
    st.markdown("### 📘 How to use")
    st.info(
        """
        1. **Tap the microphone** below.
        2. **Speak in English** clearly.
        3. **Listen** to Fluent's response.
        4. Repeat to improve!
        """
    )
    st.markdown("---")
    st.caption("Powered by Gemini 2.5 & OpenAI")

# ==========================================
# 5. MAIN CHAT UI
# ==========================================

# Custom Header
col1, col2 = st.columns([1, 8])
with col1:
    st.markdown("<h1>AI Fluent Partner</h1>", unsafe_allow_html=True)
with col2:
    st.markdown("<h1></h1>", unsafe_allow_html=True)
    st.markdown("*Practice English naturally with your personalized AI tutor.*")

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# Chat History Display
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        # Empty state welcome message
        st.markdown(
            """
            <div style='text-align: center; padding: 50px; opacity: 0.6;'>
                <h3>👋 Welcome!</h3>
                <p>Start speaking to begin your practice session.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "audio" in message:
                st.audio(message["audio"], format="audio/mp3")

# ==========================================
# 6. INPUT AREA (Fixed at bottom)
# ==========================================

st.markdown("---")

# Audio Input
audio_value = st.audio_input("🎤 Tap to speak")

if audio_value:
    # 1. Process User Input
    with st.chat_message("user"):
        with st.spinner("Processing speech..."):
            # Save audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                tmp_audio.write(audio_value.read())
                tmp_audio_path = tmp_audio.name
            
            # Transcribe
            user_text = speech_to_text(tmp_audio_path)
            st.markdown(user_text)
            
    # Add to session
    st.session_state.messages.append({"role": "user", "content": user_text})

    # 2. Process AI Response
    with st.chat_message("assistant"):
        with st.spinner("Fluent is thinking..."):
            # Get Gemini response
            ai_response_text = ask_gemini(st.session_state.chat_session, user_text)
            
            # Generate Audio
            ai_audio_path = text_to_speech(ai_response_text)
            
            # Output
            st.markdown(ai_response_text)
            st.audio(ai_audio_path, format="audio/mp3", autoplay=True)
    
    # Add to session
    st.session_state.messages.append({
        "role": "assistant", 
        "content": ai_response_text, 
        "audio": ai_audio_path
    })

    # Cleanup
    os.remove(tmp_audio_path)