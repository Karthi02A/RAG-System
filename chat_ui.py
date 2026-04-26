import streamlit as st
import os
from query import ask
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Intelligence Nexus",
    page_icon="💠",
    layout="wide"
)

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    code {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .stApp {
        background: linear-gradient(135deg, #0A0D14 0%, #11151F 100%);
        color: #E0E0E0;
    }
    
    /* Premium Chat Bubbles Base */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.03);
        background: rgba(255, 255, 255, 0.015);
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        width: 85% !important;
    }
    
    /* User Message (Right Aligned) - Detected by custom avatar image */
    [data-testid="stChatMessage"]:has(img) {
        flex-direction: row-reverse !important;
        background: rgba(245, 245, 220, 0.05) !important;
        border-right: 3px solid #FF3131 !important;
        border-left: none !important;
        margin-left: auto !important;
        margin-right: 0px !important;
        text-align: right;
    }
    
    /* Assistant Message (Left Aligned) - Detected by absence of custom avatar image */
    [data-testid="stChatMessage"]:not(:has(img)) {
        background: rgba(245, 245, 220, 0.05) !important;
        border-left: 3px solid #F5F5DC !important;
        border-right: none !important;
        margin-right: auto !important;
        margin-left: 0px !important;
        text-align: left;
    }

    /* Centered Header Section */
    .header-section {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 40px;
        margin-bottom: 40px;
        border-radius: 24px;
        background: rgba(255, 255, 255, 0.01);
        border: 1px solid rgba(255, 49, 49, 0.2);
    }

    /* Neon Red Title (No Word Shadow) */
    .main-header {
        font-size: 3.8rem;
        font-weight: 900;
        letter-spacing: -3px;
        color: #FF3131;
        margin-bottom: 0.5rem;
    }
    
    /* Beige Sub-header */
    .sub-header {
        color: #F5F5DC; /* Sophisticated Beige */
        font-size: 1.2rem;
        font-weight: 300;
        letter-spacing: 1px;
        opacity: 0.9;
    }

    /* Sidebar - Glassmorphism */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 11, 14, 0.8);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Source Tags - Professional Pills */
    .source-tag {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.03);
        color: #999;
        font-size: 0.7rem;
        font-weight: 500;
        margin-right: 8px;
        margin-top: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.2s ease;
    }
    .source-tag:hover {
        background: rgba(255, 255, 255, 0.08);
        color: #FFF;
        border-color: rgba(255, 255, 255, 0.2);
    }

    /* Input Field Styling */
    .stChatInputContainer {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        background: rgba(255, 255, 255, 0.02) !important;
        padding: 5px !important;
    }

    /* Medium Typography for Messages */
    [data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] p {
        font-size: 18px !important;
        line-height: 1.6 !important;
    }

    [data-testid="stChatMessage"] h3 {
        font-size: 1.4rem !important;
        font-weight: 800 !important;
        color: #F5F5DC !important;
        margin-top: 20px !important;
        margin-bottom: 10px !important;
        border-bottom: 1px solid rgba(245, 245, 220, 0.1);
        padding-bottom: 5px;
    }

    [data-testid="stChatMessage"] h2, [data-testid="stChatMessage"] h1 {
        color: #FF3131 !important;
        margin-top: 25px !important;
    }
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.2rem !important;
            letter-spacing: -1px !important;
        }
        .sub-header {
            font-size: 1rem !important;
        }
        .header-section {
            padding: 20px !important;
            margin-bottom: 20px !important;
        }
        [data-testid="stChatMessage"] {
            width: 98% !important;
            padding: 15px !important;
        }
        [data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] p {
            font-size: 16px !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 💠 Control Center")
    show_sources = st.toggle("Show Reference Sources", value=False, help="Toggle to see which files were used for the answer.")
    
    st.divider()
    
    st.info("Vector Index: **Verified**")
    st.info("Intelligence: **Llama 3.1**")
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- MAIN UI ---
st.markdown("""
<div class="header-section">
    <div class="main-header">Intelligence Nexus</div>
    <div class="sub-header">Direct answers from your codebase. No noise, just precision.</div>
</div>
""", unsafe_allow_html=True)

# --- ASSETS ---
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
USER_AVATAR = os.path.join(ASSETS_DIR, "user_avatar.png")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if show_sources and "sources" in message and message["sources"]:
            src_html = "".join([f'<span class="source-tag">{s}</span>' for s in message["sources"]])
            st.markdown(f'<div style="margin-top: 15px; opacity: 0.8;">{src_html}</div>', unsafe_allow_html=True)

# React to user input
if prompt := st.chat_input("Query your codebase..."):
    # Display user message in chat message container
    st.chat_message("user", avatar=USER_AVATAR).markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Searching codebase...")
        
        # Get result from query.py
        result = ask(prompt)
        
        if "error" in result:
            st.error(result["error"])
            full_response = "I encountered an error processing your request."
        else:
            full_response = result["answer"]
            sources = result.get("sources", [])
            
            # Display response
            message_placeholder.markdown(full_response)
            
            # Display sources if toggled
            if show_sources and sources:
                src_html = "".join([f'<span class="source-tag">{s}</span>' for s in sources])
                st.markdown(f'<div style="margin-top: 15px; opacity: 0.8;">{src_html}</div>', unsafe_allow_html=True)

    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "sources": result.get("sources", [])
    })
