import os
import shutil # 추가
from rag.document_processor import DocumentProcessor
from rag.rag_db import RagDB

# PDF 파일이 저장된 폴더 경로
DATA_DIR = "./rag/data" 
from core.agent_config import CHROMA_DIR

def _pre_clear_db():
    """DB 인스턴스화 전에 DB 폴더를 강제로 삭제하여 파일 잠금을 우회"""
    if os.path.exists(CHROMA_DIR):
        print(f"[Ingest] 기존 DB 폴더 ({CHROMA_DIR}) 삭제 시도...")
        try:
            # 폴더 삭제
            shutil.rmtree(CHROMA_DIR)
            print(f"[Ingest] 기존 DB 폴더 삭제 성공.")
        except PermissionError:
            print("===================================================================")
            print(f" [치명적 오류] {CHROMA_DIR} 삭제 실패 (파일 잠금).")
            print("===================================================================")
            return False
        except Exception as e:
            print(f" [Ingest 오류] 폴더 삭제 중 예상치 못한 오류 발생: {e}")
            return False
    return True

def ingest_documents():
    """
    rag/data 폴더의 모든 PDF를 읽어 ChromaDB에 색인합니다.
    """
    
    # 1. DB 인스턴스화 전에 폴더 삭제 시도
    if not _pre_clear_db():
        print("[Ingest] DB 초기화 실패로 색인 작업을 중단합니다.")
        return

    # 2. DB 및 프로세서 초기화
    # 폴더가 삭제된 상태이므로 RagDB 인스턴스화 시 ChromaDB는 새 DB 파일을 생성
    processor = DocumentProcessor()
    rag_db = RagDB()

    all_documents = []
    
    # 2. PDF 파일 순회 및 처리
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            file_path = os.path.join(DATA_DIR, filename)
            print(f"--- 파일 처리 시작: {filename} ---")
            
            try:
                # DocumentProcessor를 사용하여 PDF 로드 및 청크 분할
                documents = processor.process_pdf(file_path)
                all_documents.extend(documents)
            except FileNotFoundError as e:
                print(f"오류: {e}")
            except Exception as e:
                print(f"'{filename}' 처리 중 예상치 못한 오류 발생: {e}")
                
    # 3. 모든 청크를 DB에 저장 (색인)
    if all_documents:
        rag_db.index_documents(all_documents)
        print("\n 모든 PDF 문서 색인 완료.")
    else:
        print("\n 처리할 PDF 파일이 없습니다. rag/data 폴더를 확인하십시오.")
        
    rag_db.close()

if __name__ == "__main__":
    ingest_documents()