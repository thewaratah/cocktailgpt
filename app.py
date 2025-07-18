import streamlit as st
import streamlit_authenticator as stauth
from query import ask  # from your backend logic

# --- AUTH SETUP ---
names = ['Cocktail Team']
usernames = ['team']
passwords = ['hospitality']  # replace with a secure password

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
    # --- MAIN APP ---
    authenticator.logout('Logout', 'sidebar')
    st.set_page_config(page_title="CocktailGPT", page_icon="🍸")
    st.title("CocktailGPT")
    st.caption("Ask questions about prep, flavour, clarification, modifiers, fermentation, and more.")

    user_input = st.text_area("Ask your question:", height=120, placeholder="e.g. How do I stabilise green juices?")
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
        else:
            st.warning("Please enter a question.")


