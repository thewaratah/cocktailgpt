import streamlit as st
from query import ask

st.set_page_config(page_title="CocktailGPT", page_icon="🍸")

st.title("CocktailGPT")
st.caption("CocktailGPT · Ephemeral vector mode with live Supabase citations")

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for chat in st.session_state.messages:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])
        if "sources" in chat:
            st.markdown("#### 📚 Sources:")
            for line in chat["sources"]:
                st.markdown(f"- {line}")

# New input
user_input = st.chat_input("Ask your next question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Compile message history
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages if m["role"] in ["user", "assistant"]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = ask(user_input, message_history=history)

                if "\n\n📚 Sources:" in response:
                    answer, sources_block = response.strip().split("\n\n📚 Sources:")
                    sources = sources_block.strip().split("\n")
                else:
                    answer = response.strip()
                    sources = []

                st.markdown("### Answer:")
                st.markdown(answer.strip())

                if sources:
                    st.markdown("#### 📚 Sources:")
                    for line in sources:
                        st.markdown(f"- {line}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer.strip(),
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Error: {e}")
