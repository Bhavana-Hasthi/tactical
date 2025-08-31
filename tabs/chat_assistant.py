import streamlit as st
from chat import get_predefined_response, query_leave_policy

def render(view_mode: str, swarm_size: int):
    st.markdown('<div class="subheader"><h2>Command Chat Assistant</h2></div>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! How can I assist you with defence-related information today?"}
        ]

    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])

    user_input = st.text_input("Write a message", key="chat_input_tab7")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        placeholder_idx = len(st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": "💭 Assistant is typing..."})
        with st.chat_message("assistant"):
            st.write("💭 Assistant is typing...")

        predefined = get_predefined_response(user_input, swarm_size, view_mode)
        if predefined:
            response_text = predefined
        else:
            response = query_leave_policy(user_input)
            if "error" in response:
                response_text = f"⚠️ {response['error']}"
            else:
                response_text = response.get("data", "No data received")

        st.session_state.chat_history[placeholder_idx] = {"role": "assistant", "content": response_text}
        st.rerun()