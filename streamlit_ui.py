#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from typing import List, Dict, Any, Tuple
import requests
import streamlit as st

# ================================================================
# Backend adapters (HTTP)
# ================================================================
BACKEND_URL = os.environ.get(
    "BACKEND_URL",
    "https://cocktailgpt-production.up.railway.app"
).rstrip("/")

SERP_API_KEY = os.environ.get("SERPAPI_API_KEY", "").strip()

def check_backend_health() -> Tuple[bool, int]:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=6)
        r.raise_for_status()
        data = r.json()
        return True, int(data.get("chroma_count", 0))
    except Exception:
        return False, 0

def call_backend(prompt: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
    payload = {"question": prompt}
    if history:
        payload["history"] = history
    r = requests.post(f"{BACKEND_URL}/ask", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

def serp_search(query: str, num_results: int = 6) -> List[Dict[str, Any]]:
    if not SERP_API_KEY:
        return []
    try:
        r = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google",
                "q": query,
                "num": min(max(num_results, 1), 10),
                "api_key": SERP_API_KEY,
            },
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        out: List[Dict[str, Any]] = []
        for item in (data.get("organic_results") or [])[:num_results]:
            title = item.get("title") or ""
            link = item.get("link") or ""
            snippet = item.get("snippet") or ""
            if title and link:
                out.append({"title": title, "link": link, "snippet": snippet})
        return out
    except Exception:
        return []

# ================================================================
# Page config & session
# ================================================================
st.set_page_config(page_title="CocktailGPT", page_icon="", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, Any]] = []

st.session_state.setdefault("use_web", False)

# ================================================================
# Sidebar — Controls (no colour pickers)
# ================================================================
with st.sidebar:
    st.title("CocktailGPT 🍹")
    st.caption("An assistant for all things drinks, food science and recipe development, with optional live search.")

    st.session_state.use_web = st.checkbox("🌐 Use web search", value=st.session_state.use_web)

    st.markdown("---")
    st.markdown("### 💾 Conversation")
    st.download_button(
        "Download chat",
        data=json.dumps(st.session_state.messages, indent=2).encode("utf-8"),
        file_name="cocktailgpt-chat.json",
        mime="application/json",
        use_container_width=True
    )
    up = st.file_uploader("Restore chat", type=["json"])
    if up:
        try:
            st.session_state.messages = json.loads(up.read().decode("utf-8"))
            st.success("Conversation restored")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Could not restore: {e}")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Clear", use_container_width=True):
            st.session_state.messages = []
            st.experimental_rerun()
    with col_b:
        if st.button("Regenerate", use_container_width=True):
            last_user = None
            for m in reversed(st.session_state.messages):
                if m.get("role") == "user":
                    last_user = m.get("content", "")
                    break
            if last_user:
                st.session_state["__force_regen"] = last_user
            else:
                st.warning("No user prompt to regenerate.")

# ================================================================
# Header
# ================================================================
ok, count = check_backend_health()
colL, colM, colR = st.columns([0.08, 0.72, 0.20])
with colL:
    st.markdown("###")
with colM:
    st.markdown("## CocktailGPT")
    st.caption("A cocktail development assistant, informed by 2000+ .pdfs, textbooks, articles and literature")
with colR:
    if ok:
        st.success(f"Healthy — {count:,} chunks")
    else:
        st.error("Backend unreachable")

st.divider()

# ================================================================
# Helpers
# ================================================================
def _compact_history() -> List[Dict[str, str]]:
    return [{"role": m.get("role", ""), "content": m.get("content", "")}
            for m in st.session_state.messages if m.get("role") in ("user", "assistant")]

def _render_sources(sources: List[Any]) -> None:
    if not sources:
        return
    st.markdown("📚 **Sources:**")
    for s in sources:
        if isinstance(s, dict):
            title = s.get("title", "Source")
            link = s.get("link", "")
            snippet = s.get("snippet", "")
            if link:
                st.markdown(f"- [{title}]({link}) — {snippet}")
            else:
                st.markdown(f"- {title}")
        else:
            st.markdown(f"- {s}")

# ================================================================
# Render conversation
# ================================================================
for msg in st.session_state.messages:
    role = msg.get("role", "")
    content = msg.get("content", "")
    sources = msg.get("sources", []) or []
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(content)
        if role == "assistant":
            _render_sources(sources)

# ================================================================
# Input + message handling
# ================================================================
regen_prompt = st.session_state.pop("__force_regen", None)

prompt = st.chat_input("Ask me anything about recipes, prep, food or flavour science. If I don't get it right straight away, follow up with specific prompts.")
if prompt or regen_prompt:
    user_text = (regen_prompt or prompt or "").strip()
    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})

        web_results = []
        if st.session_state.use_web:
            try:
                web_results = serp_search(user_text, num_results=6)
            except Exception:
                web_results = []

        history = _compact_history()
        if web_results:
            lines = []
            for i, r in enumerate(web_results, start=1):
                lines.append(f"[W{i}] {r.get('title','')}\n{r.get('snippet','')}\n{r.get('link','')}".strip())
            web_block = "Web context:\n" + "\n\n".join(lines)
            history.append({"role": "user", "content": web_block})

        with st.chat_message("CocktailGPT"):
            placeholder = st.empty()
            placeholder.markdown("_BRB, changing a keg…_")

            try:
                resp = call_backend(user_text, history=history)
                answer = (resp.get("response") or "").strip()
                local_sources = resp.get("sources") or []
            except requests.HTTPError as e:
                answer = f"**Backend error:** {e.response.status_code} {e.response.text}"
                local_sources = []
            except Exception as e:
                answer = f"**Error:** {e}"
                local_sources = []

            compact_web = []
            for r in web_results or []:
                compact_web.append({
                    "title": r.get("title", ""),
                    "link":  r.get("link", ""),
                    "snippet": r.get("snippet", "")
                })

            placeholder.markdown(answer)
            if local_sources or compact_web:
                _render_sources(list(local_sources) + compact_web)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": list(local_sources) + compact_web
            })
