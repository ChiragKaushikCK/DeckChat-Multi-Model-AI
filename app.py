# app.py - DeckChat Pro with OpenRouter & Dual Models
# A premium AI chatbot experience with Firebase backend

import streamlit as st
import os
import json
import hashlib
import base64
import time
import io
import tempfile
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import pytz

# Audio Processing Imports
import speech_recognition as sr
from gtts import gTTS

IST = pytz.timezone("Asia/Kolkata")

# LangChain imports (Fixed to latest core paths)
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(
    page_title="DeckChat Pro",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------
# Load External CSS
# ----------------------
def load_css():
    """Load custom CSS for better UI"""
    css = """
    <style>
        .stApp { background: white; }
        .main > div {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 20px;
            margin: 10px;
            backdrop-filter: blur(10px);
        }
        .stChatMessage {
            padding: 1.2rem;
            border-radius: 20px;
            margin-bottom: 15px;
            animation: slideIn 0.3s ease-out;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        [data-testid="chat-message-user"] {
            background: linear-gradient(90deg,rgba(230, 230, 230, 1) 0%, rgba(255, 255, 255, 1) 55%, rgba(237, 221, 83, 1) 100%);
            color: #2d3748;
            margin-left: 20%;
        }
        [data-testid="chat-message-assistant"] {
            background: #f8f9fa;
            color: #2d3748;
            margin-right: 20%;
        }
        .user-profile {
            background: #edb268;
            padding: 25px;
            border-radius: 20px;
            color: white;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .stat-card {
            background: rgba(255,255,255,0.2);
            padding: 15px;
            border-radius: 15px;
            margin: 10px 0;
            backdrop-filter: blur(5px);
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ----------------------
# Voice & Audio Helpers
# ----------------------
def speech_to_text(audio_bytes):
    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name
    
    try:
        with sr.AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except Exception as e:
        return f"Audio recognition failed: {str(e)}"
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return base64.b64encode(fp.read()).decode()
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None

# ----------------------
# Firebase Setup
# ----------------------
@st.cache_resource
def init_firebase():
    try:
        if not firebase_admin._apps:
            if 'FIREBASE_CONFIG' in st.secrets:
                cred_dict = dict(st.secrets['FIREBASE_CONFIG'])
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred)
            else:
                st.warning("⚠️ Firebase configuration not found in Secrets.")
                return None
        return firestore.client()
    except Exception as e:
        st.error(f"⚠️ Firebase connection error: {str(e)}")
        return None

# ----------------------
# Authentication & DB
# ----------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def sign_up(email: str, password: str) -> tuple:
    if not db: return False, "Database error"
    try:
        users_ref = db.collection('users')
        if list(users_ref.where('email', '==', email).limit(1).stream()):
            return False, "User already exists"
        
        users_ref.add({
            'email': email,
            'password_hash': hash_password(password),
            'created_at': datetime.now(IST),
            'total_messages': 0, 'total_sessions': 0,
            'preferences': {'model': 'base'}
        })
        return True, "Account created successfully!"
    except Exception as e:
        return False, str(e)

def sign_in(email: str, password: str) -> tuple:
    if not db: return False, "Database error"
    try:
        docs = list(db.collection('users').where('email', '==', email).where('password_hash', '==', hash_password(password)).limit(1).stream())
        if docs:
            db.collection('users').document(docs[0].id).update({
                'last_active': datetime.now(IST),
                'total_sessions': firestore.Increment(1)
            })
            return True, "Login successful!"
        return False, "Invalid credentials"
    except Exception as e:
        return False, str(e)

def save_message(user_email: str, role: str, content: str, model_used: str = None):
    if db:
        try:
            db.collection('messages').add({
                'user_email': user_email, 'role': role, 'content': content,
                'timestamp': datetime.now(IST), 'model_used': model_used
            })
            docs = list(db.collection('users').where('email', '==', user_email).limit(1).stream())
            if docs:
                db.collection('users').document(docs[0].id).update({
                    'total_messages': firestore.Increment(1)
                })
        except Exception as e:
            st.warning(f"Save msg error: {e}")

def get_chat_history(user_email: str, limit: int = 50) -> List[Dict]:
    if not db: return []
    try:
        docs = db.collection('messages').where('user_email', '==', user_email).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit).stream()
        messages = [{'role': d.to_dict()['role'], 'content': d.to_dict()['content']} for d in docs]
        return list(reversed(messages))
    except Exception:
        return []

# ----------------------
# Model Initialization
# ----------------------
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")

def init_openrouter_model():
    api_key = st.secrets.get("OPENROUTER_API_KEY")
    if not api_key: return None
    return ChatOpenAI(model="openai/gpt-3.5-turbo", api_key=api_key, base_url="https://openrouter.ai/api/v1", streaming=True, temperature=0.7)

def init_groq_model():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return None
    from langchain_groq import ChatGroq
    return ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.7, streaming=True)

# ----------------------
# System Prompts
# ----------------------
SYSTEM_PROMPTS = {
    "default": "You are DeckChat Pro. Be helpful, concise, and use markdown formatting.",
    "code": "You are a Code Specialist. Provide clean, documented code solutions.",
}

# ----------------------
# Main Interface
# ----------------------
def main():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        
    global db
    db = init_firebase()
    
    if not st.session_state.authenticated:
        st.markdown("<h1 style='text-align: center; color: #667eea;'>✨ DeckChat Pro</h1>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        
        with tab1:
            with st.form("login"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Login", type="primary"):
                    success, msg = sign_in(email, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.user_email = email
                        st.rerun()
                    else:
                        st.error(msg)
                        
        with tab2:
            with st.form("signup"):
                new_email = st.text_input("Email")
                new_password = st.text_input("Password", type="password")
                if st.form_submit_button("Create Account", type="primary"):
                    success, msg = sign_up(new_email, new_password)
                    if success: st.success("Account created! Please login.")
                    else: st.error(msg)
    else:
        load_css()
        if 'current_model' not in st.session_state: st.session_state.current_model = "base"
        if 'messages' not in st.session_state: st.session_state.messages = get_chat_history(st.session_state.user_email)
        
        # Sidebar
        with st.sidebar:
            st.markdown(f"<div class='user-profile'><h3>{st.session_state.user_email.split('@')[0]}</h3></div>", unsafe_allow_html=True)
            st.session_state.current_model = st.radio("Model", ["base", "pro"], index=0 if st.session_state.current_model=="base" else 1)
            persona = st.selectbox("Persona", list(SYSTEM_PROMPTS.keys()))
            if st.button("🚪 Logout"):
                st.session_state.clear()
                st.rerun()

        st.markdown("<h1>DeckChat Pro</h1>", unsafe_allow_html=True)
        
        # Display chat
        for msg in st.session_state.messages:
            with st.chat_message(msg['role'], avatar="🧑" if msg['role'] == 'user' else "✨"):
                st.markdown(msg['content'])

        # Input handles
        text_prompt = st.chat_input("Type your message...")
        audio_prompt = st.audio_input("Record Voice")
        
        final_prompt = None
        
        if text_prompt:
            final_prompt = text_prompt
        elif audio_prompt:
            with st.spinner("Transcribing..."):
                final_prompt = speech_to_text(audio_prompt.getvalue())
                
        if final_prompt:
            with st.chat_message("user", avatar="🧑"):
                st.markdown(final_prompt)
            
            st.session_state.messages.append({"role": "user", "content": final_prompt})
            save_message(st.session_state.user_email, "user", final_prompt)
            
            # Setup Model & Prompt
            llm = init_openrouter_model() if st.session_state.current_model == "base" else init_groq_model()
            if not llm:
                st.error("API keys missing in Streamlit Secrets!")
                return
                
            messages_for_model = [SystemMessage(content=SYSTEM_PROMPTS[persona])]
            for m in st.session_state.messages[-10:]:
                if m['role'] == 'user': messages_for_model.append(HumanMessage(content=m['content']))
                else: messages_for_model.append(AIMessage(content=m['content']))
            
            # Stream Response
            with st.chat_message("assistant", avatar="✨"):
                placeholder = st.empty()
                stream_handler = StreamHandler(placeholder)
                llm.callbacks = [stream_handler]
                
                with st.spinner("Thinking..."):
                    response = llm.invoke(messages_for_model)
                
                placeholder.markdown(response.content)
                
                # TTS
                audio_b64 = text_to_speech(response.content)
                if audio_b64:
                    st.audio(base64.b64decode(audio_b64), format="audio/mp3", autoplay=True)
                
            st.session_state.messages.append({"role": "assistant", "content": response.content})
            save_message(st.session_state.user_email, "assistant", response.content, st.session_state.current_model)

if __name__ == "__main__":
    main()
