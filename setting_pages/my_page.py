import os
import streamlit as st

st.title("My Page")

if "_model_id" not in st.session_state:
    st.session_state._model_id = None

if "_model_api_key" not in st.session_state:
    st.session_state._model_api_key = None

# 모델 셀렉터 박스
model_names = ("OpenAI", "Anthropic", "Cohere", "Upstage", "Gemini")
model_ids = list(range(len(model_names)))
model_id = st.selectbox(
    "모델 선택",
    model_ids,
    index=st.session_state._model_id,
    format_func=lambda x: model_names[x],
    placeholder="Choose a model",
)


# 모델 설정 저장
def set_model_setting(model_id, api_key):
    st.session_state._model_id = model_id
    st.session_state._model_api_key = api_key


# 모델 API 키입력 필드
model_api_key = None
if (
    model_id == 0  # OPENAI
    or model_id == 1  # ANTHROPIC
    or model_id == 2  # COHERE
    or model_id == 3  # UPSTAGE
    or model_id == 4  # GEMINI
):
    model_api_key = st.text_input(
        "모델 API KEY",
        value=st.session_state._model_api_key,
        type="password",
        placeholder="Please enter your api key",
    )

st.button(
    "save",
    on_click=set_model_setting,
    args=(model_id, model_api_key),
    type="primary",
)


# LANGSMITH API KEY 입력 필드


def set_langsmith_api_key(api_key):
    st.session_state.langsmith_api_key = api_key
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "My GPT"


langsmith_api_key = st.text_input("LANGSMITH API KEY")

st.button(
    "save",
    key="lngsmh_save_btn",
    on_click=set_langsmith_api_key,
    args=(langsmith_api_key,),
    type="primary",
)
