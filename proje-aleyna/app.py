import streamlit as st
import os
from openai import OpenAI
import google.generativeai as genai
import tempfile

# ==========================================
# 1. AYARLAR VE KURULUM
# ==========================================

st.set_page_config(
    page_title="AI Fluent | English Tutor",
    F="🇬🇧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS VE DOSYA YOLU AYARLARI (KRİTİK DÜZELTME) ---
# Bu kısım, kodun çalıştığı klasörü otomatik bulur ve CSS yolunu ona göre hesaplar.
current_dir = os.path.dirname(os.path.abspath(__file__))
css_path = os.path.join(current_dir, "assets", "style.css")

# API Anahtarlarını Al (Streamlit Cloud Secrets)
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except FileNotFoundError:
    st.error("API anahtarları bulunamadı! Lütfen .streamlit/secrets.toml dosyasını kontrol edin veya Cloud ayarlarını yapın.")
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
- Keep responses concise (3-5 sentences max) so the user can speak more.
"""

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash", # En stabil model sürümü
    generation_config=generation_config,
    system_instruction=system_instruction,
)

# ==========================================
# 2. YARDIMCI FONKSİYONLAR
# ==========================================

def load_local_css(file_path):
    """CSS dosyasını güvenli bir şekilde yükler."""
    try:
        with open(file_path, "r") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"⚠️ Uyarı: CSS dosyası bulunamadı ({file_path}). Varsayılan tema kullanılıyor.")

def speech_to_text(audio_file_path):
    """Sesi yazıya çevirir (Whisper)."""
    with open(audio_file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            language="en"
        )
    return transcript.text

def ask_gemini(chat_session, user_text):
    """Gemini'den cevap alır."""
    response = chat_session.send_message(user_text)
    return response.text

def text_to_speech(text):
    """Yazıyı sese çevirir (TTS)."""
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy", 
        input=text
    )
    # Streamlit Cloud'da dosya izinleri için güvenli yöntem
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        response.stream_to_file(tmp_file.name)
        return tmp_file.name

# ==========================================
# 3. BAŞLATMA
# ==========================================

# CSS Yükle (Hesaplanmış yol ile)
load_local_css(css_path)

# Oturum Durumlarını Başlat
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

# ==========================================
# 4. YAN MENÜ (SIDEBAR)
# ==========================================

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
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
    st.caption("Powered by Gemini 1.5 & OpenAI")

# ==========================================
# 5. ANA SOHBET EKRANI
# ==========================================

st.markdown("<h1>AI Fluent Partner</h1>", unsafe_allow_html=True)
st.markdown("*Practice English naturally with your personalized AI tutor.*")

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# Sohbet Geçmişini Göster
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        # Boş durum mesajı
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
# 6. GİRİŞ ALANI (EN ALTTA SABİT)
# ==========================================

st.markdown("---")

# Ses Girişi (Audio Input)
audio_value = st.audio_input("🎤 Tap to speak")

if audio_value:
    # 1. Kullanıcı Girişini İşle
    with st.chat_message("user"):
        with st.spinner("Processing speech..."):
            # Sesi kaydet
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                tmp_audio.write(audio_value.read())
                tmp_audio_path = tmp_audio.name
            
            # Yazıya çevir
            user_text = speech_to_text(tmp_audio_path)
            st.markdown(user_text)
            
    # Listeye ekle
    st.session_state.messages.append({"role": "user", "content": user_text})

    # 2. AI Cevabını İşle
    with st.chat_message("assistant"):
        with st.spinner("Fluent is thinking..."):
            # Gemini'ye sor
            ai_response_text = ask_gemini(st.session_state.chat_session, user_text)
            
            # Sese çevir
            ai_audio_path = text_to_speech(ai_response_text)
            
            # Ekrana bas ve sesi çal
            st.markdown(ai_response_text)
            st.audio(ai_audio_path, format="audio/mp3", autoplay=True)
    
    # Listeye ekle
    st.session_state.messages.append({
        "role": "assistant", 
        "content": ai_response_text, 
        "audio": ai_audio_path
    })

    # Temizlik
    if os.path.exists(tmp_audio_path):
        os.remove(tmp_audio_path)
