from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_upstage import ChatUpstage
from langchain_google_genai import ChatGoogleGenerativeAI


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
    elif model_id == 1:  # ANTHROPIC
        model = ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            temperature=0,
            max_tokens=1024,
            timeout=None,
            max_retries=2,
            api_key=model_api_key,
        )
    elif model_id == 2:  # COHERE
        model = ChatCohere(cohere_api_key=model_api_key)
    elif model_id == 3:  # UPSTAGE
        model = ChatUpstage(upstage_api_key=model_api_key)
    elif model_id == 4:  # GEMINI
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            google_api_key=model_api_key,
        )

    return model
