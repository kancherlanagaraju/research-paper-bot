"""Research Paper Answer Bot -- Streamlit app (Milestone 7).

Run with:  streamlit run app.py

Ask a question about the indexed papers; get a grounded answer with inline
[Title, p. N] citations and the top-3 supporting passages. The retrieval
mode defaults to the configuration picked by the last evaluation run
(artifacts/evaluation/results.json), or hybrid + rerank if none has run.
Each question is answered independently; the history below is display-only.
"""
from __future__ import annotations

import streamlit as st

from src.app_support import default_mode, history_entry, mode_options, source_rows
from src.config import AppConfig
from src.evaluation import DEFAULT_MODES
from src.rag import answer_question


@st.cache_resource
def get_config() -> AppConfig:
    return AppConfig.from_env()


def render_answer(entry: dict) -> None:
    if entry["abstained"]:
        st.warning(entry["answer"])
        return
    st.markdown(entry["answer"])
    st.caption(f"Retrieval mode: `{entry['mode']}`")
    st.subheader("Sources (top 3)")
    for row in entry["sources"]:
        with st.expander(f"{row['rank']}. {row['title']} -- p. {row['page']}  (score {row['score']})"):
            st.caption(row["filename"])
            st.write(row["text"])


def main() -> None:
    st.set_page_config(page_title="Research Paper Answer Bot", page_icon="📄")
    st.title("Research Paper Answer Bot")
    st.caption("Answers grounded in five Generative AI papers: Attention, GPT-4, InstructGPT, Gemini, Mistral 7B.")

    try:
        config = get_config()
    except Exception as exc:  # noqa: BLE001 -- show config problems in the UI instead of a stack trace
        st.error(f"Configuration error: {exc}")
        st.stop()

    selected = default_mode(config.artifacts_dir)
    options = mode_options(selected)
    with st.sidebar:
        st.header("Settings")
        mode = st.selectbox(
            "Retrieval strategy",
            DEFAULT_MODES,
            index=DEFAULT_MODES.index(selected),
            format_func=lambda m: options[m],
        )
        if st.button("Clear history"):
            st.session_state.history = []

    history = st.session_state.setdefault("history", [])
    for entry in history:
        with st.chat_message("user"):
            st.write(entry["query"])
        with st.chat_message("assistant"):
            render_answer(entry)

    query = st.chat_input("Ask a question about the papers")
    if not query:
        return
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the papers..."):
                result = answer_question(query, retrieval_mode=mode, config=config)
        except Exception as exc:  # noqa: BLE001 -- missing keys / unreachable services should be readable
            st.error(f"Could not answer: {exc}")
            return
        entry = history_entry(result)
        render_answer(entry)
    history.append(entry)


main()
