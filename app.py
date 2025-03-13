import streamlit as st
from langchain_ollama import OllamaLLM

# Initialize Ollama model
llm = OllamaLLM(model="dolphin3:latest")

# Define system prompt with reinforcement
SYSTEM_PROMPT = """
"""

# Streamlit UI
st.title("🗨️ Mommy - Your AI Assistant")
st.write("Talk to Mommy, your personal AI assistant!")

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input field
user_input = st.chat_input("Type your message here...")

# Handle chatbot response
if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Force the assistant to always identify as "Mommy"
    formatted_prompt = f"{SYSTEM_PROMPT}\nUser: {user_input}\nMommy:"
    response = llm.invoke(formatted_prompt)

    # Store & display the response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)
