import streamlit as st
from query import ask  # uses your existing logic

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
    else:
        st.warning("Please enter a question.")

