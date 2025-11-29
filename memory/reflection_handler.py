from typing import List, Dict
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from core.agent_config import CHROMA_DIR, EMBED_MODEL, get_client, LLM_MODEL
from core.state import AgentState

class ReflectionHandler:
    """
    장기 기억(LTM) 관리 및 Reflection 로직 핸들러
    Long Term Memory & Reflection
    """

    def __init__(self):
        # LTM은 RAG와 분리된 별도의 Collection 사용
        self.collection_name = "long_term_memory"
        self.embedding_function = OpenAIEmbeddings(model=EMBED_MODEL)
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=CHROMA_DIR
        )
        self.client = get_client()

    def _summarize_messages(self, messages: List[Dict]) -> str:
        """LLM을 사용하여 대화 내역 요약"""
        if not messages:
            return ""

        # 메시지를 텍스트로 변환
        conversation_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        system_prompt = (
            "당신은 대화 내용을 핵심만 요약하여 장기 기억으로 저장하는 AI입니다. "
            "주어진 대화에서 사용자의 주요 정보, 선호도, 해결된 문제 등을 3문장 이내로 요약하세요."
        )

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"다음 대화를 요약해줘:\n{conversation_text}"}
            ]
        )
        
        return response.choices[0].message.content

    def save_reflection(self, state: AgentState) -> Dict[str, str]:
        """
        Graph Node에서 호출: 현재 상태의 메시지를 요약하여 LTM에 저장
        """
        messages = state.get("messages", [])
        if not messages:
            return {"status": "no_messages"}

        # 1. 대화 요약 (Reflection)
        summary = self._summarize_messages(messages)
        
        # 2. 요약 내용을 메타데이터와 함께 Document로 변환
        doc = Document(page_content=summary, metadata={"type": "reflection_summary"})
        
        # 3. Chroma DB(LTM Collection)에 저장
        self.vector_store.add_documents([doc])
        
        print(f"[ReflectionHandler] 장기 기억 저장 완료: {summary[:50]}...")
        
        return {"summary": summary}

    def retrieve_memories(self, query: str, k: int = 2) -> List[str]:
        """
        사용자 질문과 관련된 과거 기억 검색
        """
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]