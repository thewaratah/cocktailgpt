import os
import json
import re
import requests
import streamlit as st

# ---- Config ----
# IMPORTANT: point to the *backend* service (cocktailgpt), not the ingestor.
BACKEND_URL = os.environ.get(
    "BACKEND_URL",
    "https://cocktailgpt-production.up.railway.app",  # <-- change to your actual backend service URL
).rstrip("/")

st.set_page_config(page_title="CocktailGPT", page_icon="🍹", layout="wide")
st.title("🍹 CocktailGPT")
st.caption("Ephemeral vector mode with live Supabase citations")

# ---- Helpers ----
def call_backend(question: str, history=None):
    url = f"{BACKEND_URL}/ask"
    payload = {"question": question}
    if history:
        payload["history"] = history
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

def parse_sources_from_text(answer: str):
    """
    If backend only returns a single 'response' string that *includes*
    a '📚 Sources:' block, extract it. Otherwise return [].
    """
    if "📚 Sources:" not in answer:
        return [], answer
    body, src_block = answer.split("📚 Sources:", 1)
    # lines like "- foo.pdf (chunk 12)" or "- foo.pdf"
    lines = [ln.strip(" \n\r-") for ln in src_block.strip().splitlines() if ln.strip()]
    return lines, body.strip()

# ---- UI ----
with st.sidebar:
    st.subheader("Settings")
    st.text_input("Backend URL", BACKEND_URL, key="backend_url")
    if st.button("Check health"):
        try:
            h = requests.get(f"{st.session_state.backend_url}/health", timeout=20).json()
            st.success(h)
        except Exception as e:
            st.error(str(e))

if "history" not in st.session_state:
    st.session_state.history = []

q = st.text_input("Ask your next question…", placeholder="e.g., What is fermentation?")
go = st.button("Ask")

if go and q.strip():
    try:
        # use the sidebar override if provided
        effective_backend = st.session_state.get("backend_url") or BACKEND_URL
        url_saved = BACKEND_URL  # show if we changed it

        # Call backend
        resp = call_backend(q.strip(), history=st.session_state.history)
        answer = resp.get("response", "")

        # prefer structured 'sources' if backend returns it
        sources = resp.get("sources", [])
        if not sources:
            sources, body = parse_sources_from_text(answer)
        else:
            # if sources exist, prefer not to re-parse the body
            body = answer.split("\n\n📚 Sources:")[0] if "📚 Sources:" in answer else answer

        # show answer
        st.markdown(body)

        # show sources if any
        if sources:
            with st.expander("📚 Sources", expanded=True):
                for s in sources:
                    st.markdown(f"- {s}")

        # update chat history we send back to backend (simple text chat)
        st.session_state.history.append({"role": "user", "content": q.strip()})
        st.session_state.history.append({"role": "assistant", "content": body})

    except requests.HTTPError as e:
        try:
            st.error(f"Backend error: {e.response.status_code} {e.response.text}")
        except Exception:
            st.error(f"Backend error: {e}")
    except Exception as e:
        st.error(str(e))
