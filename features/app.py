# streamlit_app.py
import streamlit as st

if "message_history" not in st.session_state:
    st.session_state['message_history'] = []

for message in st.session_state['message_history']:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state['message_history'].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    st.session_state['message_history'].append({"role": "assistant", "content": user_input})
    with st.chat_message("assistant"):
        st.text(user_input)