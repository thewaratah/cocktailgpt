import streamlit as st
from query import ask  # your existing vector search function

st.set_page_config(page_title="CocktailGPT", page_icon="🍸")

st.title("CocktailGPT")
st.caption("Ask about prep, fermentation, flavour science, ingredients, and more.")

# Initialise chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show the chat history
for i, chat in enumerate(st.session_state.messages):
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# New user message
user_input = st.chat_input("Ask your next question...")

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call CocktailGPT backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = ask(user_input)
                answer, *sources = response.strip().split("\n\n📚 Sources used:")
                st.markdown(answer.strip())

                if sources:
                    st.markdown("#### 📚 Sources:")
                    for line in sources[0].split("\n"):
                        st.markdown(f"- {line}")
                # Save assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer.strip()
                })
            except Exception as e:
                st.error(f"Error: {e}")
