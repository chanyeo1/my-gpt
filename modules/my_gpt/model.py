from langchain_openai import ChatOpenAI

# 설정 페이지에서 선택한 모델을 반환
def get_model(model_id, model_api_key):
    model = None
    if model_id == 0:  # OPENAI
        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=model_api_key,
        )
    elif model_id == 1:  # XIONIC
        model =  ChatOpenAI(
            model_name="xionic-1-72b-20240610",
            base_url="https://sionic.chat/v1/",
            api_key="934c4bbc-c384-4bea-af82-1450d7f8128d",
        )

    return model
