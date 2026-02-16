import streamlit as st
import os
from openai import OpenAI
import google.generativeai as genai
import tempfile

# --- AJAN KOD BAŞLANGIÇ ---
import streamlit as st
import os

try:
    key = st.secrets["OPENAI_API_KEY"]
    st.warning(f"🔑 Anahtar Durumu: Anahtar bulundu! İlk 5 harfi: {key[:5]}... Son 3 harfi: ...{key[-3:]}")
    if key.startswith("sk-"):
        st.success("✅ Format doğru görünüyor (sk- ile başlıyor).")
    else:
        st.error("🚨 HATA: Anahtar 'sk-' ile başlamıyor! Kopyalarken hata olmuş olabilir.")
except Exception as e:
    st.error(f"🚨 Anahtar Okunamadı! Hata: {e}")
# --- AJAN KOD BİTİŞ ---

# ==========================================
# 1. PAGE CONFIG (EN BAŞTA OLMALI)
# ==========================================
# Streamlit kuralı: Bu komut her şeyden (hatta if/else bloklarından) önce gelmeli.
st.set_page_config(
    page_title="AI Fluent Partner",
    page_icon=":uk:",  # Emoji yerine shortcode kullanımı daha güvenlidir
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. SETUP & API KEYS
# ==========================================

# Dosya yollarını hatasız bulmak için "Current Directory" hesabı
current_dir = os.path.dirname(os.path.abspath(__file__))
css_path = os.path.join(current_dir, "assets", "style.css")

# CSS Yükleme Fonksiyonu
def load_css(file_path):
    try:
        with open(file_path, "r") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        # Hata olursa uygulama çökmesin, sadece uyarı versin
        st.warning(f"⚠️ CSS file not found at: {file_path}")

# API Anahtarlarını Al (Secrets Kontrolü)
if "OPENAI_API_KEY" in st.secrets and "GOOGLE_API_KEY" in st.secrets:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("🚨 API Anahtarları bulunamadı! Lütfen Streamlit Cloud ayarlarından 'Secrets' kısmını kontrol et.")
    st.stop()

# İstemcileri Başlat
client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

# Gemini Model Ayarı
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
- Keep responses concise (3-5 sentences max).
"""

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config=generation_config,
    system_instruction=system_instruction,
)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

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
# 4. APP LOGIC
# ==========================================

# CSS'i yükle
load_css(css_path)

# Session State Başlatma
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_session = model.start_chat(history=[])
        st.rerun()
        
    st.markdown("---")
    st.info("Tap the microphone below to start speaking.")

# --- MAIN CHAT UI ---
st.title("AI Fluent Partner")

# Mesaj Geçmişini Göster
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.markdown("<div style='text-align: center; color: #666;'>👋 Start speaking to begin!</div>", unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "audio" in message:
                st.audio(message["audio"], format="audio/mp3")

# --- AUDIO INPUT (BOTTOM) ---
st.markdown("---")
audio_value = st.audio_input("🎤 Tap to speak")

if audio_value:
    # 1. User Logic
    with st.chat_message("user"):
        with st.spinner("Processing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                tmp_audio.write(audio_value.read())
                tmp_audio_path = tmp_audio.name
            
            user_text = speech_to_text(tmp_audio_path)
            st.markdown(user_text)
            st.session_state.messages.append({"role": "user", "content": user_text})
            os.remove(tmp_audio_path) # Clean up input file

    # 2. AI Logic
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ai_response_text = ask_gemini(st.session_state.chat_session, user_text)
            ai_audio_path = text_to_speech(ai_response_text)
            
            st.markdown(ai_response_text)
            st.audio(ai_audio_path, format="audio/mp3", autoplay=True)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": ai_response_text, 
                "audio": ai_audio_path
            })

