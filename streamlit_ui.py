import streamlit as st
from query import ask
import json

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
            st.markdown("#### 📚 Sources and Tags:")
            for line in chat["sources"]:
                st.markdown(f"- {line}")

            try:
                with open("tags_by_chunk.json", "r") as f:
                    tag_data = json.load(f)
            except:
                tag_data = {}

            for line in chat["sources"]:
                if "(chunks " in line:
                    filename = line.split(" (chunks")[0].strip()
                    chunk_part = line.split("chunks")[1].strip(" )")
                    for chunk_id in chunk_part.split(","):
                        chunk_id = f"{filename.replace('.pdf', '').replace('.csv', '').replace(' ', '_')}_{chunk_id.strip()}"
                        tags = tag_data.get(chunk_id)
                        if tags:
                            st.markdown(f"📌 Tags for `{chunk_id}`:")
                            st.json(tags)

# Enable refining the last user message
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    if st.button("🛠 Refine this answer"):
        for prev in reversed(st.session_state.messages):
            if prev["role"] == "user":
                st.session_state.refine_input = prev["content"]
                break

# Handle chat input
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

                if "\n\n📚 Sources:" in response:
                    answer, sources_block = response.strip().split("\n\n📚 Sources:")
                    sources = sources_block.strip().split("\n")
                else:
                    answer = response.strip()
                    sources = []

                st.markdown("### Answer:")
                st.markdown(answer.strip())

                st.markdown("#### 📚 Sources and Tags:")
                for line in sources:
                    st.markdown(f"- {line}")

                try:
                    with open("tags_by_chunk.json", "r") as f:
                        tag_data = json.load(f)
                except:
                    tag_data = {}

                for line in sources:
                    if "(chunks " in line:
                        filename = line.split(" (chunks")[0].strip()
                        chunk_part = line.split("chunks")[1].strip(" )")
                        for chunk_id in chunk_part.split(","):
                            chunk_id = f"{filename.replace('.pdf', '').replace('.csv', '').replace(' ', '_')}_{chunk_id.strip()}"
                            tags = tag_data.get(chunk_id)
                            if tags:
                                st.markdown(f"📌c Tags for `{chunk_id}`:")
                                st.json(tags)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer.strip(),
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Error: {e}")
