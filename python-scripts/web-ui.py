import json
import requests
import streamlit as st


@st.cache_data(max_entries=1)
def get_models(endpoint: str, api_token: str):
    try:
        if "/chat/completions" not in endpoint:
            return []
        headers={"Authorization": f"Bearer {api_token}"} if api_token else {}
        models_endpoint = endpoint.replace("/chat/completions", "/models")
        models = requests.get(models_endpoint, headers=headers).json()
        return [m["id"] for m in models["data"]]
    except Exception as e:
        return []


def iter_sse_messages(lines: list[str]):
    buffer = []
    for line in lines:
        if line.startswith("data:"):
            line = line[len("data:"):]
            if line.startswith(" "):
                line = line[1:]
            buffer.append(line)
        else:
            if buffer:
                yield "\n".join(buffer)
                buffer = []


def chat(messages: list[tuple[str, str]], **kwargs):
    body = {
        "model": model,
        "messages": [dict(role=role, content=content) for role, content in messages],
        "stream": True,
        **kwargs,
    }
    headers={"Authorization": f"Bearer {api_token}"} if api_token else {}
    r = requests.post(
        endpoint,
        headers=headers,
        json=body,
        stream=True,
    )
    r.raise_for_status()
    assert "text/event-stream" in r.headers["content-type"]
    for line in iter_sse_messages(r.iter_lines(decode_unicode=True)):
        if line == "[DONE]":
            break
        data = json.loads(line)
        delta = data["choices"][0]["delta"]
        if "content" in delta:
            yield delta["content"]


# page state

if "history" not in st.session_state:
    st.session_state["history"] = []


# parameters

with st.sidebar:
    st.markdown("## API endpoint")

    endpoint = st.text_input(
        "Enpoint",
        value="http://127.0.0.1:5137/v1/chat/completions",
        placeholder="/v1/chat/completions",
    )
    api_token = st.text_input("Api Token", value="", placeholder="Empty", type="password")

    available_models = get_models(endpoint, api_token)

    if len(available_models):
        model = st.selectbox("Select model", available_models)
    else:
        model = st.text_input("Model name", value="gpt-3.5-turbo", placeholder="gpt-3.5-turbo")

    st.markdown("## Gneration parameters")

    system_prompt = st.text_input("system_prompt", value="", placeholder="Empty")
    max_tokens = st.number_input("max_tokens", min_value=1, max_value=2000, value=800)
    temperature = st.number_input("temperature", min_value=0.0, max_value=4.0, value=0.7)
    top_p = st.number_input("top_p", min_value=0.0, max_value=1.0, value=1.0)
    frequency_penalty = st.number_input("frequency_penalty", min_value=-2.0, max_value=2.0, value=0.0)

    if st.button("Clear history"):
        st.session_state.history = []

    if st.button("Revert last message"):
        st.session_state.history = st.session_state.history[:-1]


# main body

st.markdown("## llm-sharp web ui")

# [(role, content), ...]
history: list[tuple[str, str]] = st.session_state.history

if len(history) == 0:
    st.caption("Start chatting by typing a message below")


for idx, (role, content) in enumerate(history):
    with st.chat_message(role):
        st.write(content)

question = st.chat_input("Message", key="message")

if question:
    if len(history) <= 0 and system_prompt:
        history = [("system", system_prompt)]
        with st.chat_message("system"):
            st.write(system_prompt)

    history = history + [("user", question)]

    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        empty = st.empty()
        with st.spinner("Generating..."):
            answer = ""
            for chunk in chat(
                messages=history,
                top_p=top_p,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
            ):
                answer += chunk
                empty.write(answer)

    st.session_state.history = history + [("assistant", answer)]
