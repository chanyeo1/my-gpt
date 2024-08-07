import streamlit as st

st.title("My Page")

# OPENAI API KEY 입력 필드
openai_api_key = st.text_input("OPENAI API KEY")

def set_openai_api_key(api_key):
    st.session_state.openai_api_key = api_key

st.button('save', key="openai_save_btn", on_click=set_openai_api_key, args=(openai_api_key, ), type="primary")

# LANGSMITH API KEY 입력 필드

def set_langsmith_api_key(api_key):
    st.session_state.langsmith_api_key = api_key

langsmith_api_key = st.text_input("LANGSMITH API KEY")

st.button('save', key="lngsmh_save_btn", on_click=set_langsmith_api_key, args=(langsmith_api_key, ), type="primary")