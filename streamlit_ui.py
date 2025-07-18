import streamlit as st
from query import ask  # your existing vector search function

st.set_page_config(page_title="CocktailGPT", page_icon="🍸")

st.title("CocktailGPT")
st.caption("Ask about prep, fermentation, flavour science, ingredients, and more.")

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show existing chat messages
for i, chat in enumerate(st.session_state.messages):
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# New user input
user_input = st.chat_input("Ask your next question...")

if user_input:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = ask(user_input)

                # Try splitting answer and sources
                if "\n\n📚 Sources used:" in response:
                    answer, sources = response.strip().split("\n\n📚 Sources used:")
                    sources = sources.strip().split("\n")
                else:
                    answer = response.strip()
                    sources = []

                # Display assistant's answer
                st.markdown(answer.strip())

                # Display sources (if any)
                if sources:
                    st.markdown("#### 📚 Sources:")
                    for line in sources:
                        st.markdown(f"- {line}")

                # Save full assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.strip()
                })

            except Exception as e:
                st.error(f"Error: {e}")
