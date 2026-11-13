import streamlit as st
from llama_cpp import Llama

# Model configuration: LFM2.5-350M-Thinking, Q4_K_M quantization, approximately 220MB
MODEL_REPO = "KoarAI/LFM2.5-350M-Thinking-GGUF"
MODEL_FILE = "LFM2.5-350M-Thinking-Q4_K_M.gguf"

@st.cache_resource(show_spinner="Loading ultra-light model...")
def load_model():
    """
    Use @st.cache_resource to ensure the model is loaded only once and shared globally.
    This is crucial for the 1GB memory limit of Streamlit Cloud.
    """
    llm = Llama.from_pretrained(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        n_ctx=512,        # Strictly limit context to 512 to minimize memory usage
        n_threads=1,      # Limit to 1 thread to avoid CPU contention
        verbose=False
    )
    return llm

def generate_response(llm, user_input):
    """Call the model to generate a response."""
    response = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": user_input}
        ],
        max_tokens=200,
        temperature=0.6,
        top_p=0.95
    )
    return response["choices"][0]["message"]["content"]

# --- Streamlit UI ---
st.set_page_config(page_title="Ultra-Light Chatbot", page_icon="⚡")
st.title("⚡ Ultra-Light Chatbot")
st.caption("LFM2.5-350M | Q4_K_M | Running on Streamlit Cloud")

llm = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = generate_response(llm, prompt)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
