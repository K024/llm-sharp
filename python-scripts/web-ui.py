import requests
import streamlit as st


def chat(history: list[tuple[str, str]], question: str, temperature = 1.0, top_p: float = 0.8, **kwargs):
    body = {
        "history": history,
        "question": question,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    r = requests.post(
        "http://127.0.0.1:5137/api/Chat/QuestionAnswering",
        json=body,
        stream=True,
    )
    r.raise_for_status()
    assert "text/event-stream" in r.headers["content-type"]
    add_line_break = False
    for line in r.iter_lines(decode_unicode=True):
        if line:
            if line.startswith("data:"):
                line = line[len("data:"):]
                if line.startswith(" "):
                    line = line[1:]
                if line == "[DONE]":
                    break
                if add_line_break:
                    yield "\n"
                yield line
                add_line_break = True
        else:
            add_line_break = False


# page state

if "history" not in st.session_state:
    st.session_state["history"] = []


# parameters

with st.sidebar:
    st.markdown("## 采样参数")

    max_tokens = st.number_input("max_tokens", min_value=1, max_value=2000, value=800)
    temperature = st.number_input("temperature", min_value=0.1, max_value=4.0, value=1.0)
    top_p = st.number_input("top_p", min_value=0.1, max_value=1.0, value=0.8)
    top_k = st.number_input("top_k", min_value=1, max_value=100, value=50)

    if st.button("清空上下文"):
        st.session_state.history = []

    if st.button("撤销上一条消息"):
        st.session_state.history = st.session_state.history[:-1]


# main body

st.markdown("## llm-sharp")

history: list[tuple[str, str]] = st.session_state.history

if len(history) == 0:
    st.caption("请在下方输入消息开始会话")


for idx, (question, answer) in enumerate(history):
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(answer)

question = st.chat_input("消息", key="message")

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        empty = st.empty()
        with st.spinner("正在回复中"):
            answer = ""
            for chunk in chat(
                history=history,
                question=question,
                top_k=top_k,
                top_p=top_p,
                max_generated_tokens=max_tokens,
                temperature=temperature,
            ):
                answer += chunk
                empty.write(answer)

    st.session_state.history = history + [(question, answer)]
