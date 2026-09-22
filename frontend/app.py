import re
import streamlit as st
from api_client import ask_backend, check_backend_health, API_BASE_URL

st.set_page_config(
    page_title="Python for ML — Technical Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Modern CSS supporting both Light & Dark themes effortlessly
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* App Header Styling */
    .hero-container {
        padding: 1.5rem 0 1rem 0;
        margin-bottom: 1.25rem;
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 50%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 400;
        line-height: 1.5;
    }

    /* Status Pill */
    .status-badge-online {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        background: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.25);
        font-size: 0.825rem;
        font-weight: 600;
    }

    .status-badge-offline {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        background: rgba(239, 68, 68, 0.12);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.25);
        font-size: 0.825rem;
        font-weight: 600;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: currentColor;
    }

    /* Answer Card */
    .response-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 1.5rem 1.75rem;
        margin-top: 1.25rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(8px);
    }

    .response-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #38bdf8;
    }

    /* Clean typography inside response */
    .response-body {
        font-size: 1.025rem;
        line-height: 1.7;
        letter-spacing: -0.01em;
    }

    /* Subtle sample prompt buttons */
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease-in-out;
    }

    .stButton button:hover {
        border-color: #6366f1;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to separate prose and code
def parse_response_content(raw_text: str):
    """Separates pure prose explanation from code blocks."""
    # Check if backend gave explicit [CODE_SNIPPET] marker
    if "[CODE_SNIPPET]" in raw_text:
        parts = raw_text.split("[CODE_SNIPPET]")
        prose = parts[0].strip()
        code_raw = parts[1].strip()
        # Clean any markdown fences from code_raw
        code_clean = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", code_raw)
        code_clean = re.sub(r"\n```$", "", code_clean).strip()
        return prose, code_clean

    # Otherwise extract ``` fences
    code_matches = re.findall(r"```(?:python)?\n(.*?)```", raw_text, re.DOTALL)
    prose = re.sub(r"```(?:python)?\n.*?```", "", raw_text, flags=re.DOTALL).strip()
    
    code_clean = "\n\n".join(c.strip() for c in code_matches if c.strip())
    return prose, code_clean

# Check backend health
backend_online = check_backend_health()

# Sidebar Navigation & Settings
with st.sidebar:
    st.markdown("### ⚙️ System Status")
    if backend_online:
        st.markdown(
            '<div class="status-badge-online"><span class="pulse-dot"></span> Backend Active (Port 8000)</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-badge-offline"><span class="pulse-dot"></span> Backend Disconnected</div>',
            unsafe_allow_html=True
        )
        st.caption("Run `start_backend.bat` to launch the API service.")

    st.markdown("---")
    st.markdown("### 🎛️ Preferences")
    include_code_toggle = st.toggle("Include code examples", value=False, help="Enable to automatically display code snippets alongside explanations.")

    st.markdown("---")
    st.markdown("### 💡 Common Topics")
    st.caption("Select a query to test the assistant:")

    sample_questions = [
        "What is the common mistake with mutable default arguments in Python?",
        "Why does arr[1:4] modify the original NumPy array?",
        "What happens when performing arithmetic on uint8 image pixels?",
        "Why does df.pivot() crash when data contains duplicate keys?",
        "Why is x == np.nan always False in Pandas?",
        "Why should you wrap conditions in parentheses when filtering a DataFrame?",
        "What is the danger of catching a bare except in Python?"
    ]

    selected_sample = None
    for sq in sample_questions:
        if st.button(sq, key=f"btn_{sq}", use_container_width=True):
            selected_sample = sq

# Main View Area
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Python for ML Assistant</div>
    <div class="hero-subtitle">Intelligent, grounded technical assistance for NumPy, Pandas, and Core Python.</div>
</div>
""", unsafe_allow_html=True)

# Input Row
default_query = selected_sample if selected_sample else ""
question_input = st.text_input(
    label="Ask a question",
    value=default_query,
    placeholder="e.g., What is the common mistake with mutable default arguments in Python?",
    label_visibility="collapsed"
)

col_ask, col_opt, _ = st.columns([1.5, 2, 4])
with col_ask:
    ask_button = st.button("Ask Assistant", type="primary", use_container_width=True)

# Determine if user explicitly asked for code in their prompt
def user_requested_code(text: str) -> bool:
    keywords = ["code", "example", "snippet", "syntax", "how to write", "show me", "def ", "script", "implementation"]
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)

# Execution & Presentation
if (ask_button or selected_sample) and question_input.strip():
    with st.spinner("Analyzing technical documentation and preparing answer..."):
        try:
            result = ask_backend(question_input)
            raw_answer = result.get("answer", "")

            # Parse answer into prose explanation and optional code
            prose_explanation, extracted_code = parse_response_content(raw_answer)

            # Check if code was requested via toggle or prompt keywords
            wants_code = include_code_toggle or user_requested_code(question_input)

            # Render response
            st.markdown(f"""
            <div class="response-card">
                <div class="response-header">
                    <span>💡 Technical Explanation</span>
                </div>
                <div class="response-body">
            """, unsafe_allow_html=True)

            # Display clean conceptual explanation
            st.markdown(prose_explanation)

            # Handle code display cleanly: only if requested, or collapsible
            if extracted_code:
                if wants_code:
                    st.markdown("#### 💻 Code Example")
                    st.code(extracted_code, language="python")
                else:
                    # Provide an elegant collapsed expander so user can inspect only if interested
                    with st.expander("💻 View Code Example", expanded=False):
                        st.code(extracted_code, language="python")

            st.markdown("</div></div>", unsafe_allow_html=True)

            # Display cited source document(s) directly below the answer
            sources = result.get("sources", [])
            if sources:
                st.caption("Sources: " + ", ".join(sources))

        except Exception as e:
            st.error(f"Unable to complete request: {str(e)}")
