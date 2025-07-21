import streamlit as st
import streamlit_authenticator as stauth
from query import ask  # EphemeralClient + Supabase ingestion version

# --- AUTH SETUP ---
names = ['Cocktail Team']
usernames = ['team']
passwords = ['hospitality']  # replace with secure value

hashed_pw = stauth.Hasher(passwords).generate()

authenticator = stauth.Authenticate(
    names, usernames, hashed_pw,
    'cocktailgpt', 'abcdef', cookie_expiry_days=1
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status is False:
    st.error('Incorrect username or password')
elif authentication_status is None:
    st.warning('Please enter your credentials')
else:
    authenticator.logout('Logout', 'sidebar')
    st.set_page_config(page_title="CocktailGPT", page_icon="🍸")
    st.title("CocktailGPT")
    st.caption("CocktailGPT · Ephemeral vector mode with live Supabase citations")

    # Initialise chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                st.markdown("#### 📚 Sources:")
                for line in msg["sources"]:
                    st.markdown(f"- {line}")

    # User input
    user_input = st.chat_input("Ask your next question...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Construct history
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
                    st.error(f"Something went wrong: {e}")
