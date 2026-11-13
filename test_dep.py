import streamlit as st
from llama_cpp import Llama

# Model configuration: use the Q4_K_M quantized version (recommended, balancing speed and quality)
MODEL_REPO = "daniloreddy/gemma-3-270m-it_GGUF"
MODEL_FILE = "gemma-3-270m-it_Q4_K_M.gguf"

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """
    Load Gemma 3 270M GGUF model using llama-cpp-python.
    The @st.cache_resource decorator ensures the model is loaded only once,
    avoiding repeated memory usage on every interaction.
    """
    llm = Llama.from_pretrained(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        n_ctx=2048,          # Context window; 2048 is sufficient for simple Q&A
        n_threads=2,         # Limit thread count on Streamlit Cloud to avoid resource contention
        verbose=False
    )
    return llm

def generate_response(llm, user_input):
    """Call the model to generate a response."""
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {"role": "user", "content": user_input}
        ],
        max_tokens=256,
        temperature=0.7,
        top_p=0.9
    )
    return response["choices"][0]["message"]["content"]

# --- Streamlit UI ---
st.set_page_config(page_title="Gemma 3 270M Chatbot", page_icon="💬")
st.title("💬 Gemma 3 270M (Q4) Chatbot")
st.caption("Running locally with llama-cpp-python on Streamlit Community Cloud")

# Load model (will only actually execute on first run)
llm = load_model()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Type your message..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = generate_response(llm, prompt)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})