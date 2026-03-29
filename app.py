import streamlit as st
from groq import Groq

#RUN BY USING COMMAND -  streamlit run app.py
import os
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL_PRIMARY = "llama-3.3-70b-versatile"
MODEL_FALLBACK = "llama-3.1-8b-instant"

MAX_HISTORY = 6
SUMMARY_TRIGGER = 10

SYSTEM_PROMPT = """
You are a helpful AI assistant.

Instructions:
1. Use previous conversation context to understand queries.
2. Resolve pronouns like 'it', 'they', 'this' using history.
3. Do NOT repeat previous answers unless necessary.

Accuracy Rules:
4. Do NOT hallucinate. If unsure, say "I’m not sure".
5. Ask clarification questions if the query is ambiguous.

Response Style:
6. Keep answers concise, clear, and relevant.
"""


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def summarize_history():
    if len(st.session_state.chat_history) < SUMMARY_TRIGGER:
        return

    summary_prompt = [
        {"role": "system", "content": "Summarize this conversation briefly."},
        {"role": "user", "content": str(st.session_state.chat_history)}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_PRIMARY,
            messages=summary_prompt
        )
    except:
        response = client.chat.completions.create(
            model=MODEL_FALLBACK,
            messages=summary_prompt
        )

    summary = response.choices[0].message.content


    st.session_state.chat_history = [
        {"role": "system", "content": f"Summary: {summary}"}
    ]


def build_messages(user_input):
    summarize_history()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    history = st.session_state.chat_history[-MAX_HISTORY:]
    messages.extend(history)

    messages.append({"role": "user", "content": user_input})

    return messages


def get_response(user_input):
    messages = build_messages(user_input)

    try:
        response = client.chat.completions.create(
            model=MODEL_PRIMARY,
            messages=messages,
            temperature=0.7
        )
    except:
        response = client.chat.completions.create(
            model=MODEL_FALLBACK,
            messages=messages,
            temperature=0.7
        )

    reply = response.choices[0].message.content


    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )
    st.session_state.chat_history.append(
        {"role": "assistant", "content": reply}
    )

    return reply


st.set_page_config(page_title="Context-Aware Chatbot", layout="centered")

st.title("🧠 Context-Aware Chatbot (Groq + Streamlit)")


for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant").write(msg["content"])


user_input = st.chat_input("Type your message...")

if user_input:
    st.chat_message("user").write(user_input)

    with st.spinner("Thinking..."):
        response = get_response(user_input)

    st.chat_message("assistant").write(response)
