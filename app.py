"""Streamlit demo: user text -> Layer 1 crisis classifier -> Layer 3 Llama response.

Run with:
    streamlit run app.py

Prerequisite: `hf auth login` and an accepted license for
meta-llama/Llama-3.2-1B-Instruct (gated base model for the response adapter).
"""

from __future__ import annotations

import streamlit as st

from src.pipeline import (
    DEFAULT_ADAPTER_DIR,
    DEFAULT_CLASSIFIER_DIR,
    append_turn,
    build_classifier_input,
    build_context,
    classify,
    generate_reply,
    load_classifier,
    load_response_model,
)

st.set_page_config(page_title="Crisis-Aware Dialogue Demo", page_icon="🕊️", layout="wide")


@st.cache_resource(show_spinner="Loading crisis-type classifier...")
def _load_classifier():
    return load_classifier(DEFAULT_CLASSIFIER_DIR)


@st.cache_resource(show_spinner="Loading Llama response model (first run downloads the base model)...")
def _load_response_model():
    return load_response_model(DEFAULT_ADAPTER_DIR)


def build_label_badge(result: dict) -> dict:
    """Map a classifier result to an st.badge() spec: label, color, icon."""
    if not result["flagged"]:
        return {"label": "clear", "color": "green", "icon": "🟢"}
    top = sorted(
        ((label, score) for label, score in result["scores"].items() if label in result["predicted_types"]),
        key=lambda item: item[1],
        reverse=True,
    )
    detail = ", ".join(f"{label} ({score:.2f})" for label, score in top)
    return {"label": f"FLAGGED: {detail}", "color": "red", "icon": "🔴"}


def render_user_turn(content: str, badge: dict) -> None:
    """User message and its classifier badge, in clearly separated columns."""
    text_col, badge_col = st.columns([4, 1], vertical_alignment="top")
    with text_col:
        st.write(content)
    with badge_col:
        st.badge(badge["label"], color=badge["color"], icon=badge["icon"])


st.title("Crisis-Aware Dialogue Demo")
st.warning(
    "**Research prototype, not a real crisis service.** Model outputs are generated "
    "text, not clinical advice. If you or someone else is in danger, call or text "
    "**988** (Suicide & Crisis Lifeline) or your local emergency number.",
    icon="⚠️",
)

if "history" not in st.session_state:
    st.session_state.history = []
if "known_events" not in st.session_state:
    st.session_state.known_events = []

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Classifier threshold", 0.0, 1.0, 0.5, 0.05)
    max_new_tokens = st.slider("Max reply tokens", 32, 400, 180, 8)
    if st.button("Reset conversation"):
        st.session_state.history = []
        st.session_state.known_events = []
        st.rerun()

    st.subheader("Session risk status")
    if st.session_state.known_events:
        # Every crisis label seen anywhere in this session, not just the latest
        # message -- this stays in the risk context sent to Llama even after a
        # later message looks clear on its own, so it's shown here too rather
        # than only in the per-message badge.
        st.warning(
            "Known risk this session: " + ", ".join(st.session_state.known_events),
            icon="⚠️",
        )
    else:
        st.caption("No crisis signals detected yet this session.")

# Load both models immediately on page load rather than on first chat submission,
# so the (potentially slow, first-run-only) download/load happens up front instead
# of stalling the first message. st.cache_resource makes every rerun after the
# first instant.
classifier_model, classifier_tokenizer = _load_classifier()
response_model, response_tokenizer = _load_response_model()

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        if turn["role"] == "user" and turn.get("badge"):
            render_user_turn(turn["content"], turn["badge"])
        else:
            st.write(turn["content"])

user_text = st.chat_input("What's on your mind?")

if user_text:
    classifier_input = build_classifier_input(st.session_state.history, user_text)
    result = classify(classifier_model, classifier_tokenizer, classifier_input, threshold=threshold)
    badge = build_label_badge(result)

    context, st.session_state.known_events = build_context(result, st.session_state.known_events)

    history_before_turn = list(st.session_state.history)
    turn = append_turn(st.session_state.history, "user", user_text)
    turn["badge"] = badge

    with st.chat_message("user"):
        render_user_turn(turn["content"], badge)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = generate_reply(
                response_model,
                response_tokenizer,
                history_before_turn,
                user_text,
                context,
                max_new_tokens=max_new_tokens,
            )
        st.write(reply)

    append_turn(st.session_state.history, "assistant", reply)
