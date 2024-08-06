import streamlit as st

chat_page = st.Page("gpt_pages/chat_page.py", title="My ChatGPT")
my_page = st.Page("setting_pages/my_page.py", title="My Page")

pg = st.navigation(
    {
        "GPT": [chat_page],
        "SETTING": [my_page],
    }
)
pg.run()
