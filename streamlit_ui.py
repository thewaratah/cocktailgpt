import os
import json
import time
import requests
import streamlit as st

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
BACKEND_URL = os.environ.get(
    "BACKEND_URL",
    "https://cocktailgpt-production.up.railway.app",  # set your backend URL here if not via env
).rstrip("/")

SERP_API_KEY = os.environ.get("SERPAPI_API_KEY", "").strip()  # put this on the UI service

PAGE_TITLE = "CocktailGPT"
PAGE_SUBTITLE = "Trained on thousands of industry documents — with optional live web search"

st.set_page_config(page_title="CocktailGPT", page_icon="🍹", layout="wide")


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def backend_health():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=6)
        r.raise_for_status()
        data = r.json()
        count = data.get("chroma_count", 0)
        return True, count, data
    except Exception:
        return False, 0, None


def call_backend(question: str, history=None):
    payload = {"question": question}
    if history:
        payload["history"] = history
    r = requests.post(f"{BACKEND_URL}/ask", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def split_sources_from_response(answer: str):
    """
    Fallback helper: if backend only returns a single text blob that includes
    a '📚 Sources:' section, split it. Otherwise return body and [].
    """
    if "📚 Sources:" not in answer:
        return answer, []
    body, src_block = answer.split("📚 Sources:", 1)
    lines = [ln.strip(" \n\r-") for ln in src_block.strip().splitlines() if ln.strip()]
    return body.strip(), lines


def serp_search(query: str, num_results: int = 6):
    """
    Google web search via SerpAPI. Returns a list of dicts:
    [{"title":..., "link":..., "snippet":...}, ...]
    """
    if not SERP_API_KEY:
        return []

    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": query,
        "num": min(max(num_results, 1), 10),
        "api_key": SERP_API_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        out = []
        # Prefer organic_results; fallback to news_results if present
        results = data.get("organic_results") or []
        for item in results[:num_results]:
            title = item.get("title") or ""
            link = item.get("link") or ""
            snippet = item.get("snippet") or item.get("snippet_highlighted_words") or ""
            if isinstance(snippet, list):
                snippet = " … ".join(snippet)
            if title and link:
                out.append({"title": title, "link": link, "snippet": snippet})
        return out
    except Exception:
        return []


def render_web_sources(sources):
    for s in sources:
        title = s.get("title", "Untitled")
        link = s.get("link", "")
        snippet = s.get("snippet", "")
        if link:
            st.markdown(f"- [{title}]({link})")
        else:
            st.markdown(f"- {title}")
        if snippet:
            st.caption(snippet)


# -------------------------------------------------------------------
# Header (no sidebar)
# -------------------------------------------------------------------
ok, count, health_json = backend_health()

colL, colM, colR = st.columns([0.08, 0.72, 0.20])
with colL:
    st.markdown("### 🍹")
with colM:
    st.markdown(f"## {PAGE_TITLE}")
    st.caption(PAGE_SUBTITLE)
with colR:
    if ok:
        st.markdown(
            f"<div style='text-align:right'>"
            f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:#16a34a;margin-right:6px;'></span>"
            f"<code>Healthy</code><br/><small>{count:,} chunks</small>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='text-align:right'>"
            f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:#dc2626;margin-right:6px;'></span>"
            f"<code>Backend unreachable</code>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.divider()

# Top-row controls (right aligned)
ctrlL, ctrlR = st.columns([0.7, 0.3])
with ctrlR:
    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.session_state.pop("msg_sources", None)
        st.session_state.pop("msg_web_sources", None)
        st.rerun()

# -------------------------------------------------------------------
# Chat state
# -------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "msg_sources" not in st.session_state:
    # backend (local) sources per assistant message index
    st.session_state.msg_sources = {}
if "msg_web_sources" not in st.session_state:
    # web sources per assistant message index
    st.session_state.msg_web_sources = {}

# -------------------------------------------------------------------
# Chat history render
# -------------------------------------------------------------------
for idx, m in enumerate(st.session_state.messages):
    role = m.get("role", "assistant")
    content = m.get("content", "")
    with st.chat_message(role):
        st.markdown(content)
        if role == "assistant":
            local_src = st.session_state.msg_sources.get(idx) or []
            web_src = st.session_state.msg_web_sources.get(idx) or []
            if local_src or web_src:
                with st.expander("📚 Sources", expanded=False):
                    if local_src:
                        st.markdown("**Local corpus**")
                        for s in local_src:
                            st.markdown(f"- {s}")
                    if web_src:
                        st.markdown("**🌐 Web sources**")
                        render_web_sources(web_src)

# -------------------------------------------------------------------
# Input row (with optional web-search toggle)
# -------------------------------------------------------------------
row1, row2 = st.columns([0.75, 0.25])
with row2:
    use_web = st.checkbox("Use web search", value=False,
                          help="Tick to search the web and blend results with the local knowledge base. "
                               "You can also force search by starting your prompt with `search:`")

prompt = st.chat_input("Ask a question or follow up…")

if prompt:
    # Force web search if user prefixed 'search:'
    forced_search = prompt.strip().lower().startswith("search:")
    clean_prompt = prompt[len("search:"):].strip() if forced_search else prompt.strip()

    # 1) Echo user message
    st.session_state.messages.append({"role": "user", "content": clean_prompt})
    with st.chat_message("user"):
        st.markdown(clean_prompt)

    # 2) Build recent history to send to backend
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-12:]  # last ~6 exchanges
        if m.get("role") in ("user", "assistant")
    ]

    # 3) Optional web search (no backend changes required)
    web_results = []
    if forced_search or use_web:
        web_results = serp_search(clean_prompt, num_results=6)

        if web_results:
            # Compose a compact "Web context" message to feed into GPT via history.
            # This lets the model blend web info with local context it gets from the backend.
            lines = []
            for i, r in enumerate(web_results, start=1):
                title = r.get("title", "")
                link = r.get("link", "")
                snippet = r.get("snippet", "")
                lines.append(f"[W{i}] {title}\n{snippet}\n{link}".strip())
            web_context_block = "Web context (external sources; may be non-authoritative):\n" + "\n\n".join(lines)

            # Add to history BEFORE the new user question so the backend includes it
            history.append({"role": "user", "content": web_context_block})

    # 4) Assistant placeholder while waiting
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Thinking…_")

    try:
        # 5) Call backend with history (which may include a Web context block)
        resp = call_backend(clean_prompt, history=history)
        answer = resp.get("response", "") or ""
        local_sources = resp.get("sources", []) or []

        # 6) Fallback if backend returned a combined block
        if not local_sources:
            body, parsed_sources = split_sources_from_response(answer)
            local_sources = parsed_sources
        else:
            body = answer.split("\n\n📚 Sources:")[0] if "📚 Sources:" in answer else answer

        # 7) Replace placeholder with the real answer
        placeholder.markdown(body)

        # 8) Append assistant message + sources to state
        idx = len(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": body})
        st.session_state.msg_sources[idx] = local_sources
        st.session_state.msg_web_sources[idx] = web_results or []

        # 9) Render sources under this assistant bubble as well
        if local_sources or web_results:
            with st.expander("📚 Sources", expanded=True):
                if local_sources:
                    st.markdown("**Local corpus**")
                    for s in local_sources:
                        st.markdown(f"- {s}")
                if web_results:
                    st.markdown("**🌐 Web sources**")
                    render_web_sources(web_results)

    except requests.HTTPError as e:
        placeholder.markdown(f"**Backend error:** {e.response.status_code} {e.response.text}")
    except Exception as e:
        placeholder.markdown(f"**Error:** {e}")
