import os
import shutil
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# core 패키지의 설정값 참조
from core.agent_config import CHROMA_DIR, EMBED_MODEL

class RagDB:
    """
    RAG 전용 벡터 데이터베이스 관리 클래스
    """

    def __init__(self, collection_name: str = "rag_collection"):
        self.persist_dir = CHROMA_DIR
        self.collection_name = collection_name
        self.embedding_function = OpenAIEmbeddings(model=EMBED_MODEL)
        
        # ChromaDB 클라이언트 초기화 (Persistent)
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_dir
        )
        # 내부 Chroma 클라이언트 참조 (연결 해제용)
        # LangChain Chroma의 내부 속성 _client를 가져와 저장
        self.client = getattr(self.vector_store, '_client', None)


    def index_documents(self, documents: List[Document]) -> None:
        """
        문서 리스트를 임베딩하여 DB에 저장 (색인)
        """
        if not documents:
            print("[RagDB] 저장할 문서가 없습니다.")
            return

        print(f"[RagDB] {len(documents)}개의 문서를 벡터 DB에 저장 중...")
        self.vector_store.add_documents(documents)
        print("[RagDB] 저장 완료.")


    def retrieve_documents(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        쿼리와 유사한 문서를 검색하여 반환
        """
        # 유사도 검색 수행
        docs = self.vector_store.similarity_search(query, k=k)
        
        # 결과 포맷팅
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 0)
            })
            
        return results

    def close(self):
        """DB 연결을 명시적으로 해제"""
        if self.client:
            try:
                if hasattr(self.client, 'close'):
                    self.client.close()
                self.vector_store = None
                self.client = None
                # print("[RagDB] ChromaDB 클라이언트 연결 해제 시도 완료.")
            except Exception as e:
                # 연결 해제 중 발생한 오류는 무시하고 진행
                pass

    def clear_db(self):
        """DB 초기화"""
        self.close()

        if os.path.exists(self.persist_dir):
            try:
                shutil.rmtree(self.persist_dir)
                print(f"[RagDB] {self.persist_dir} 삭제 완료.")
            except PermissionError:
                # 삭제 권한 오류가 발생하면, 외부에서 수동으로 처리하도록 메시지 출력
                print(f"[RagDB Error] {self.persist_dir} 삭제 실패 (파일 잠금). 수동 삭제 필요.")
            except Exception as e:
                print(f"[RagDB Error] 삭제 중 예상치 못한 오류 발생: {e}")