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