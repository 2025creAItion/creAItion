import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentProcessor:
    """
    PDF 문서를 로드하고 텍스트를 분할하는 프로세서
    Document Processor
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def process_pdf(self, file_path: str) -> List[Document]:
        """
        PDF 파일을 읽어 Document 객체 리스트로 반환
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        # pypdf 기반 Loader 사용
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # 텍스트 분할 수행
        split_docs = self.text_splitter.split_documents(documents)
        
        print(f"[DocumentProcessor] '{file_path}' 처리 완료. {len(split_docs)}개의 청크 생성.")
        return split_docs