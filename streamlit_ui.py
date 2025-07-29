import streamlit as st
import json
import requests
import re

API_URL = "https://cocktailgpt-ingestor-production.up.railway.app/ask"

st.set_page_config(page_title="CocktailGPT", page_icon="🍸")
st.title("CocktailGPT")
st.caption("CocktailGPT · Powered by Supabase + Vector Search")

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

# Enable refining the last user message
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    if st.button("🛠 Refine this answer"):
        for prev in reversed(st.session_state.messages):
            if prev["role"] == "user":
                st.session_state.refine_input = prev["content"]
                break

# --- Query Remote Ask API ---
def ask(question, message_history=None):
    try:
        payload = {"question": question}
        if message_history:
            payload["history"] = message_history
        res = requests.post(API_URL, json=payload)
        res.raise_for_status()
        return res.json().get("response", "[No response returned]")
    except Exception as e:
        return f"[Error calling backend] {e}"

# --- Handle chat input ---
user_input = st.chat_input("Ask your next question...")
if user_input is None and "refine_input" in st.session_state:
    user_input = st.session_state.pop("refine_input")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages if m["role"] in ["user", "assistant"]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = ask(user_input, message_history=history)

                split = re.split(r"\n+📚 Sources:\n+", response.strip(), maxsplit=1)
                if len(split) == 2:
                    answer, sources_block = split
                    sources = [line.strip() for line in sources_block.strip().split("\n") if line.strip()]
                else:
                    answer = response.strip()
                    sources = []

                st.markdown("### Answer:")
                st.markdown(answer.strip())

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
