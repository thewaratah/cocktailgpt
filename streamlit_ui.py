#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from typing import List, Dict, Any, Tuple

import requests
import streamlit as st

# ================================================================
# Backend adapters (HTTP, no backend code import)
# ================================================================
BACKEND_URL = os.environ.get(
    "BACKEND_URL",
    "https://cocktailgpt-production.up.railway.app",  # your backend URL
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
    """Direct SerpAPI call from UI (optional)."""
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
            snippet = item.get("snippet") or item.get("snippet_highlighted_words") or ""
            if isinstance(snippet, list):
                snippet = " … ".join(snippet)
            if title and link:
                out.append({"title": title, "link": link, "snippet": snippet})
        return out
    except Exception:
        return []


# ================================================================
# Page config & session
# ================================================================
st.set_page_config(page_title="CocktailGPT", page_icon="🍹", layout="wide")

if "messages" not in st.session_state:
    # Each message: {"role": "user"|"assistant", "content": str, "sources": Optional[List[Any]]}
    st.session_state.messages: List[Dict[str, Any]] = []

# live theme
st.session_state.setdefault("brand_primary", "#0F766E")  # deep teal
st.session_state.setdefault("brand_accent",  "#F59E0B")  # amber
st.session_state.setdefault("brand_surface", "#0B1416")  # near-black for assistant bubbles
st.session_state.setdefault("brand_bg",      "#0E1A1C")  # page background
st.session_state.setdefault("use_web", False)


# ================================================================
# Global Styles
# ================================================================
st.markdown(f"""
<style>
:root {{
  --brand-primary: {st.session_state.brand_primary};
  --brand-accent:  {st.session_state.brand_accent};
  --brand-surface: {st.session_state.brand_surface};
  --brand-bg:      {st.session_state.brand_bg};
  --text: #E6F0F0;
  --muted: #9BB3B3;
  --border: rgba(255,255,255,0.08);
}}

html, body, .stApp {{
  background: var(--brand-bg);
  color: var(--text);
}}

.block-container {{
  max-width: 1100px;
  padding-top: 1rem;
}}

h1, h2, h3 {{ letter-spacing: .2px; }}

.header-wrap {{
  padding: 14px 18px;
  border: 1px solid var(--border);
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  border-radius: 14px;
  margin-bottom: 12px;
}}

.status-pill {{
  display:inline-flex;align-items:center;gap:8px;
  padding:4px 10px;border-radius:999px;
  border:1px solid var(--border);font-size:12px;color:var(--muted);
}}

.chat-wrap {{
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 8px 10px;
  background: rgba(255,255,255,0.02);
}}

.chat-bubble {{
  border-radius: 14px;
  padding: 12px 14px;
  border:1px solid var(--border);
  line-height: 1.55;
  word-break: break-word;
}}
.chat-bubble.user {{
  background: rgba(255,255,255,0.03);
}}
.chat-bubble.assistant {{
  background: color-mix(in srgb, var(--brand-surface) 86%, black);
  border-color: rgba(255,255,255,0.06);
}}
.small-muted {{ color: var(--muted); font-size: 12px; }}
.code, pre code {{
  white-space: pre-wrap;
  word-wrap: break-word;
}}
a {{ color: var(--brand-accent); text-decoration:none; }}
a:hover {{ text-decoration: underline; }}

@media (max-width: 640px) {{
  .block-container {{ padding-left: .5rem; padding-right: .5rem; }}
  .chat-bubble {{ padding: 10px 12px; }}
}}
</style>
""", unsafe_allow_html=True)


# ================================================================
# Sidebar — Branding + Conversation tools
# ================================================================
with st.sidebar:
    st.title("CocktailGPT 🍹")
    st.caption("Hospitality‑ready chat with optional live search.")

    st.markdown("### 🎨 Branding")
    st.session_state.brand_primary = st.color_picker("Primary", st.session_state.brand_primary)
    st.session_state.brand_accent  = st.color_picker("Accent",  st.session_state.brand_accent)
    st.session_state.brand_surface = st.color_picker("Surface (assistant bubble)", st.session_state.brand_surface)
    st.session_state.brand_bg      = st.color_picker("Background", st.session_state.brand_bg)

    st.markdown("---")
    st.session_state.use_web = st.checkbox("🌐 Use web search", value=st.session_state.use_web)

    st.markdown("---")
    st.markdown("### 💾 Conversation")
    # Download history
    st.download_button(
        "Download chat (.json)",
        data=json.dumps(st.session_state.messages, indent=2).encode("utf-8"),
        file_name="cocktailgpt-chat.json",
        mime="application/json",
        use_container_width=True
    )
    # Upload/restore history
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
        if st.button("Regenerate", use_container_width=True, help="Re-ask the last user prompt"):
            # Find last user prompt
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
    st.markdown("### 🍹")
with colM:
    st.markdown(
        "<div class='header-wrap'>"
        "<h2 style='margin:0'>CocktailGPT</h2>"
        "<div class='small-muted'>Designed for fast, reliable service support</div>"
        "</div>",
        unsafe_allow_html=True
    )
with colR:
    if ok:
        st.markdown(
            "<span class='status-pill'>"
            "<span style='width:8px;height:8px;border-radius:50%;background:#16a34a;display:inline-block'></span>"
            "Healthy"
            f"</span><div class='small-muted' style='text-align:right'>{count:,} chunks</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<span class='status-pill'>"
            "<span style='width:8px;height:8px;border-radius:50%;background:#dc2626;display:inline-block'></span>"
            "Backend unreachable</span>",
            unsafe_allow_html=True
        )

st.divider()


# ================================================================
# Helpers
# ================================================================
def _compact_history() -> List[Dict[str, str]]:
    """Return role/content pairs only, for passing to the backend."""
    return [{"role": m.get("role", ""), "content": m.get("content", "")}
            for m in st.session_state.messages if m.get("role") in ("user", "assistant")]


def _render_sources(sources: List[Any]) -> None:
    """Inline source list (no collapsibles)."""
    if not sources:
        return
    st.markdown("<div class='small-muted' style='margin-top:6px'>📚 Sources</div>", unsafe_allow_html=True)
    for s in sources:
        if isinstance(s, dict):
            title = s.get("title") or s.get("name") or "Source"
            link = s.get("link") or s.get("url")
            snippet = s.get("snippet", "")
            if link:
                st.markdown(f"- [{title}]({link})  \n  <span class='small-muted'>{snippet}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"- {title}", unsafe_allow_html=True)
        else:
            st.markdown(f"- {s}", unsafe_allow_html=True)


# ================================================================
# Render conversation so far
# ================================================================
st.markdown("<div class='chat-wrap'>", unsafe_allow_html=True)
for msg in st.session_state.messages:
    role = msg.get("role", "")
    content = msg.get("content", "")
    sources = msg.get("sources", []) or []

    with st.chat_message("user" if role == "user" else "assistant"):
        bubble_class = "user" if role == "user" else "assistant"
        st.markdown(f"<div class='chat-bubble {bubble_class}'>{content}</div>", unsafe_allow_html=True)
        if role == "assistant":
            _render_sources(sources)
st.markdown("</div>", unsafe_allow_html=True)


# ================================================================
# Input + message handling
# ================================================================
# Handle regenerate if requested
regen_prompt = st.session_state.pop("__force_regen", None)

prompt = st.chat_input("Ask me anything about recipes, prep, or service…")
if prompt or regen_prompt:
    user_text = (regen_prompt or prompt or "").strip()
    if user_text:
        # Append user turn
        st.session_state.messages.append({"role": "user", "content": user_text})

        # Optional web search (pre-query enrichment)
        web_results = []
        if st.session_state.use_web:
            try:
                web_results = serp_search(user_text, num_results=6)
            except Exception:
                web_results = []

        # Build history for backend; include a compact 'Web context' block if we have results
        history = _compact_history()
        if web_results:
            lines = []
            for i, r in enumerate(web_results, start=1):
                title = r.get("title", "")
                snippet = r.get("snippet", "")
                link = r.get("link", "")
                lines.append(f"[W{i}] {title}\n{snippet}\n{link}".strip())
            web_block = "Web context (external; verify if critical):\n" + "\n\n".join(lines)
            history.append({"role": "user", "content": web_block})

        # Assistant placeholder with hospitality-themed loading text
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("🛢️ _Changing a keg, brb…_")

            # Call backend
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

            # Compose compact web sources for display + persistence
            compact_web = []
            for r in web_results or []:
                compact_web.append({
                    "title": r.get("title", ""),
                    "link":  r.get("link", ""),
                    "snippet": r.get("snippet", "")
                })

            # Render final assistant bubble
            placeholder.markdown(f"<div class='chat-bubble assistant'>{answer}</div>", unsafe_allow_html=True)
            if local_sources or compact_web:
                _render_sources(list(local_sources) + compact_web)

            # Persist assistant turn
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": list(local_sources) + compact_web
            })
