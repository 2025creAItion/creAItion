import os
from dotenv import load_dotenv  # type: ignore
from openai import OpenAI       # type: ignore

load_dotenv()  # .env 로드

# 사용할 모델 이름
LLM_MODEL = "gpt-4o-mini"

# 임베딩 모델 
EMBED_MODEL = "text-embedding-3-large"

# Chroma DB 저장 폴더
CHROMA_DIR = "./chroma_db"


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OpenAI Key가 설정되어 있지 않습니다."
            ".env 파일 확인 요망"
        )
    return OpenAI(api_key=api_key) # 리턴 추가 했습니다!
