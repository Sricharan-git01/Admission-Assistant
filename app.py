import os
import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
import logging

# ---------------------------
# Logging Setup
# ---------------------------
logging.basicConfig(
    filename='chatbot.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
CHAT_MODEL = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

# ---------------------------
# Azure OpenAI Client
# ---------------------------
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# ---------------------------
# Load FAISS and text chunks
# ---------------------------
try:
    index = faiss.read_index("index.faiss")
except Exception as e:
    logging.error("Failed to load FAISS index: %s", e)
    st.error("Could not load FAISS index file. Please check 'index.faiss'.")
    st.stop()

try:
    with open("texts.txt", "r", encoding="utf-8") as f:
        chunks = f.read().split("\n\n---\n\n")
except Exception as e:
    logging.error("Failed to load text chunks: %s", e)
    st.error("Could not load document chunks. Please check 'texts.txt'.")
    st.stop()

# ---------------------------
# Retrieve matching chunks
# ---------------------------
def retrieve_context(query, k=3):
    try:
        response = client.embeddings.create(
            input=[query],
            model=EMBED_MODEL
        )
        vector = np.array(response.data[0].embedding).astype("float32")
        D, I = index.search(np.array([vector]), k)
        return "\n\n".join([chunks[i] for i in I[0]])
    except Exception as e:
        logging.exception("Error retrieving context for '%s': %s", query, e)
        return ""

# ---------------------------
# Dark mode minimal CSS
# ---------------------------
def load_css():
    st.markdown("""
    <style>
    /* Dark theme base */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    .main {
        background-color: #0e1117;
        padding: 2rem;
    }
    
    /* Header */
    .header {
        background-color: #1e2130;
        padding: 2rem;
        border-radius: 4px;
        border: 1px solid #262730;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .header h1 {
        color: #fafafa;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0;
    }
    
    .header p {
        color: #a6a6a6;
        font-size: 0.9rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Form container */
    .form-container {
        background-color: #1e2130;
        padding: 2rem;
        border: 1px solid #262730;
        border-radius: 4px;
        margin-bottom: 1.5rem;
    }
    
    /* Custom question section */
    .question-container {
        background-color: #1e2130;
        border: 2px solid #4a9eff;
        border-radius: 4px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        position: relative;
    }
    
    .question-container::before {
        content: '';
        position: absolute;
        top: -1px;
        left: -1px;
        right: -1px;
        height: 3px;
        background: linear-gradient(90deg, #4a9eff, #6bb6ff);
        border-radius: 4px 4px 0 0;
    }
    
    .question-container h3 {
        color: #4a9eff;
        font-size: 1rem;
        font-weight: 600;
        margin: 0 0 1rem 0;
    }
    
    /* Answer container */
    .answer-container {
        background-color: #1e2130;
        padding: 2rem;
        border: 1px solid #262730;
        border-radius: 4px;
        margin-top: 1rem;
    }
    
    .answer-container h3 {
        color: #fafafa;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 1px solid #262730;
        padding-bottom: 0.5rem;
    }
    
    .answer-container p {
        color: #d1d1d1;
        line-height: 1.6;
        margin: 0;
    }
    
    /* Mode selection buttons */
    .stButton > button[kind="primary"] {
        background-color: #4a9eff !important;
        color: #ffffff !important;
        border: none;
        padding: 0.6rem 1rem;
        font-size: 0.9rem;
        border-radius: 4px;
        font-weight: 600;
        transition: background-color 0.2s;
    }
    
    .stButton > button[kind="secondary"] {
        background-color: #262730 !important;
        color: #a6a6a6 !important;
        border: 1px solid #404040;
        padding: 0.6rem 1rem;
        font-size: 0.9rem;
        border-radius: 4px;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background-color: #2d2d3d !important;
        color: #fafafa !important;
    }
    
    /* Submit button styling */
    .stButton > button:not([kind]) {
        background-color: #fafafa;
        color: #0e1117;
        border: none;
        padding: 0.6rem 2rem;
        font-size: 0.9rem;
        border-radius: 4px;
        font-weight: 500;
        width: 100%;
        transition: background-color 0.2s;
    }
    
    .stButton > button:not([kind]):hover {
        background-color: #e0e0e0;
    }
    
    /* Form elements */
    .stSelectbox label, .stTextInput label {
        font-weight: 500;
        color: #fafafa;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }
    
    .stSelectbox > div > div {
        background-color: #262730;
        border: 1px solid #404040;
        color: #fafafa;
        border-radius: 4px;
    }
    
    .stTextInput > div > div > input {
        background-color: #262730;
        border: 1px solid #404040;
        color: #fafafa;
        border-radius: 4px;
        padding: 0.7rem;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #808080;
        font-style: italic;
    }
    
    /* Custom question input styling */
    .question-container .stTextInput > div > div > input {
        background-color: #1a1e2e;
        border: 1px solid #4a9eff;
        color: #fafafa;
    }
    
    .question-container .stTextInput > div > div > input:focus {
        border-color: #6bb6ff;
        box-shadow: 0 0 0 2px rgba(74, 158, 255, 0.2);
    }
    
    /* Info section */
    .info-section {
        background-color: #1e2130;
        border: 1px solid #262730;
        border-radius: 4px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .info-section h4 {
        color: #fafafa;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 1px solid #262730;
        padding-bottom: 0.5rem;
    }
    
    .info-section ul {
        margin: 0;
        padding-left: 1.2rem;
    }
    
    .info-section li {
        color: #a6a6a6;
        margin-bottom: 0.3rem;
        font-size: 0.85rem;
    }
    
    /* Warning styling */
    .warning-box {
        background-color: #2d1b1b;
        color: #ffb3b3;
        padding: 1rem;
        border: 1px solid #4d2d2d;
        border-radius: 4px;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    
    /* Streamlit specific overrides */
    .stSelectbox > div > div > div {
        background-color: #262730;
        color: #fafafa;
    }
    
    .stTextInput > div > div {
        background-color: #262730;
    }
    
    /* Hide elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    header {visibility: hidden;}
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #fafafa !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------
# Minimal Dark UI
# ---------------------------
def main():
    st.set_page_config(
        page_title="College Admission Assistant", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    load_css()
    
    # Header
    st.markdown("""
    <div class="header">
        <h1>College Admission Assistant</h1>
        <p>Academic information system</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Initialize session state for question mode
        if 'question_mode' not in st.session_state:
            st.session_state.question_mode = False
        
        # Mode selection buttons
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("Browse Topics", use_container_width=True, type="primary" if not st.session_state.question_mode else "secondary"):
                st.session_state.question_mode = False
        
        with col_b:
            if st.button("Ask a Question", use_container_width=True, type="primary" if st.session_state.question_mode else "secondary"):
                st.session_state.question_mode = True
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        user_input = ""
        
        if not st.session_state.question_mode:
            # Topic Selection Section
            
            main_topic = st.selectbox(
                "Select Topic",
                [
                    "Academic Programs",
                    "Admissions", 
                    "Tuition & Fees",
                    "Campus Life & Housing"
                ]
            )

            subtopics = {
                "Academic Programs": ["Undergraduate Degrees", "Graduate Degrees"],
                "Admissions": ["How to Apply", "International Students", "Test Score Requirements"], 
                "Tuition & Fees": ["Annual Budget", "Budget Worksheet"],
                "Campus Life & Housing": ["Undergraduate Housing", "Graduate Housing"]
            }

            subtopic = st.selectbox("Select Category", subtopics[main_topic])
            user_input = f"{main_topic} - {subtopic}"
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            # Custom Question Section
            st.markdown("""
            <div class="question-container">
                <h3>Ask Your Custom Question</h3>
            </div>
            """, unsafe_allow_html=True)
            
            user_input = st.text_input("Question", placeholder="Type your specific question here...", key="custom_question", label_visibility="collapsed")
        
        if st.button("Submit Query"):
            if user_input.strip():
                handle_query(user_input)
            else:
                if st.session_state.question_mode:
                    st.warning("Please enter your question.")
                else:
                    st.warning("Please select a topic and category.")
    
    with col2:
        st.markdown("""
        <div class="info-section">
            <h4>Quick Guide</h4>
            <ul>
                <li>Select from dropdown menus</li>
                <li>Use custom questions for specific queries</li>
                <li>Responses based on official data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-section">
            <h4>Topics</h4>
            <ul>
                <li>Application process</li>
                <li>Requirements</li>
                <li>Financial information</li>
                <li>Campus resources</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def handle_query(user_input):
    """Process and display response"""
    logging.info("User input: %s", user_input)
    
    with st.spinner("Processing..."):
        try:
            context = retrieve_context(user_input)
            if not context.strip():
                st.markdown("""
                <div class="warning-box">
                    No relevant information found. Try a different query.
                </div>
                """, unsafe_allow_html=True)
                return

            full_prompt = (
                "Provide clear information based on the context below.\n\n"
                f"Context: {context}\n\n"
                f"Question: {user_input}"
            )

            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a college admission assistant. Provide clear, accurate responses."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )

            answer = response.choices[0].message.content.strip()
            
            st.markdown(f"""
            <div class="answer-container">
                <h3>Response</h3>
                <p>{answer}</p>
            </div>
            """, unsafe_allow_html=True)
            
            logging.info("Response generated")

        except Exception as e:
            logging.exception("Error: %s", e)
            st.error("Error processing request. Please try again.")

if __name__ == "__main__":
    main()