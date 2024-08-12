import streamlit as st

chat_page = st.Page("gpt_pages/chat_page.py", title="My ChatGPT")
pdf_reader_page = st.Page("gpt_pages/pdf_reader.py", title="PDF Reader")
my_page = st.Page("setting_pages/my_page.py", title="My Page")

pg = st.navigation(
    {
        "GPT": [chat_page, pdf_reader_page],
        "SETTING": [my_page],
    }
)
pg.run()
