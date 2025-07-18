import streamlit as st
import os
from openai import OpenAI
import chromadb

# Setup OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Setup ChromaDB
client = chromadb.PersistentClient(path="embeddings")
collection = client.get_or_create_collection(name="cocktailgpt")

# Define the `ask` function
def ask(question):
    # Step 1: Embed the question
    query_embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[question]
    ).data[0].embedding

    # Step 2: Query Chroma
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas"]
    )

    # Step 3: Construct prompt with first match
    docs = results["documents"][0]
    sources = results["metadatas"][0]
    context = "\n\n".join(docs)

    # Step 4: Send to GPT-4-turbo
    completion = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are CocktailGPT, a flavour-obsessed beverage expert. Answer clearly and accurately using the provided context."},
            {"role": "user", "content": f"Use the following context to answer:\n\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.2
    )

    # Step 5: Build response
    answer = completion.choices[0].message.content.strip()
    source_list = [f"{meta['source']} (chunk {meta['chunk_id']})" for meta in sources if 'source' in meta]
    source_block = "\n".join(source_list)

    return f"{answer}\n\n📚 Sources used:\n{source_block}"


# Streamlit UI
st.set_page_config(page_title="CocktailGPT", page_icon="🍸")

st.title("CocktailGPT")
st.caption("Ask a question about prep, flavour, modifiers, clarification, fermentation, and more.")

user_input = st.text_area("Ask your question:", height=120, placeholder="e.g. How can I stabilise clarified passionfruit juice?")

if st.button("Submit"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            try:
                response = ask(user_input.strip())
                answer, *source_lines = response.strip().split("\n\n📚 Sources used:")
                st.markdown("### Answer:")
                st.markdown(answer.strip())
                if source_lines:
                    st.markdown("#### 📚 Sources:")
                    for line in source_lines[0].split("\n"):
                        st.markdown(f"- {line}")
            except Exception as e:
                st.error(f"Something went wrong: {e}")
# Initialise session history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Render chat history
for i, (user, bot) in enumerate(st.session_state.chat_history):
    st.markdown(f"**You:** {user}")
    st.markdown(f"**CocktailGPT:** {bot}")

# Input box
user_input = st.text_input("Ask a question or follow-up:")

if st.button("Send") and user_input.strip():
    with st.spinner("Thinking..."):
        try:
            response = ask(user_input.strip())
            st.session_state.chat_history.append((user_input.strip(), response.strip()))
            st.rerun()
        except Exception as e:
            st.error(f"Something went wrong: {e}")
    else:
        st.warning("Please enter a question.")
