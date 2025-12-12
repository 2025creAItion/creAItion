**creAItion — LangGraph ReAct Agent**

간단 소개
- **설명**: LangGraph 기반의 ReAct 에이전트를 구현. OpenAI LLM을 호출해 툴(구글 검색/계산기/시간)을 사용하고, RAG(PDF 색인 + Chroma), 장기/단기 메모(Reflection)를 지원하며 Gradio UI를 FastAPI에 마운트하여 제공.

**Features**
- **LLM**: OpenAI 사용 (기본 모델 `gpt-4o-mini`, 설정: `core/agent_config.py`)  
- **Tool Calling**: Google Custom Search(`web_search`), `get_time`, `calculator` — 정의: `tools/tool_definitions.py`  
- **RAG**: PDF Reader(`rag/document_processor.py`), 텍스트 스플리터(`RecursiveCharacterTextSplitter`), OpenAI 임베딩(`text-embedding-3-large`), ChromaDB persistent 저장(`rag/rag_db.py`)  
- **Memory**: Short-term = LangGraph `AgentState` (Annotated lists), Long-term = Chroma persistent (Reflection 저장을 통해 자동 저장)  
- **Graph Engine**: LangGraph 기반 상태 그래프(`core/graph.py`), `call_llm`, `execute_tools`, `retrieve_rag`, `check_reflection`, `save_reflection` 노드, interrupt 및 stream 지원  
- **UI**: Gradio Blocks UI (`ui/gradio_interface.py`)를 FastAPI(`main.py`)에 마운트

**Quick Start (Windows PowerShell)**
1. 파이썬 3.10+ 설치 및 가상환경 생성/활성화
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2. 의존성 설치
```powershell
pip install -r requirements.txt
```
3. 환경변수 설정: 프로젝트 루트에 `.env` 파일 생성 또는 수정
```
OPENAI_API_KEY="sk-..."
GOOGLE_API_KEY="AIza..."
GOOGLE_SEARCH_ENGINE_ID="cx_..."
```
4. RAG용 PDF 준비: `rag/data/` 폴더에 PDF 파일 넣기
5. PDF 색인(옵션 — 색인 초기화 포함)
```powershell
python rag/rag_ingest.py
```
6. 앱 실행
```powershell
# FastAPI + Gradio mount
uvicorn main:app --reload
```
7. 브라우저로 접속: `http://127.0.0.1:8000/` (또는 `http://localhost:8000/`)

**How to test tools**
- 시간: "지금 서울 시간 알려줘" → `get_time` 호출  
- 계산: "3 * 7 해줘" → `calculator` 호출  
- 웹 검색: "최신 뉴스 검색해줘: <질문>" → `web_search` 호출

**Project layout (주요 파일)**
- `main.py` — FastAPI 앱 + Gradio mount
- `ui/gradio_interface.py` — Gradio Blocks UI 및 `chat_fn`
- `core/agent_config.py` — OpenAI 클라이언트 설정, `LLM_MODEL`, `EMBED_MODEL`, `CHROMA_DIR`
- `core/graph.py` — LangGraph 상태 그래프, 노드 구현
- `core/state.py` — `AgentState` 정의 (Annotated lists)
- `tools/tool_definitions.py` — 툴 정의 및 OpenAI function schema 생성
- `tools/tool_manager.py` — LLM이 만든 tool_calls 실행기
- `rag/document_processor.py` — PDF 로더 + 텍스트 분리
- `rag/rag_db.py` — Chroma 기반 RAG DB 래퍼
- `rag/rag_ingest.py` — PDF 폴더 색인 스크립트
- `memory/reflection_handler.py` — LTM(Chroma)에 Reflection 저장
- `docs/graph_diagram.svg` — 프로젝트 그래프 다이어그램 (SVG)

**Environment / Configuration**
- 기본적으로 `.env`에서 `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GOOGLE_SEARCH_ENGINE_ID`를 읽음.
- `CHROMA_DIR` 기본값은 `./chroma_db`이며 `core/agent_config.py`에서 변경할 수 있음.

**Behavior notes & troubleshooting**
- ChromaDB 파일 잠금: Windows에서 Chroma DB 파일이 잠겨 있으면 `rag_ingest.py`의 폴더 삭제가 실패합니다. 수동으로 `chroma_db` 폴더를 삭제하거나 프로세스를 중지.  
- 패키지/모듈 불일치: 레포에서 사용하는 일부 import (예: `langchain_chroma`, `langchain_openai`, `langgraph`)는 환경과 버전 차이로 설치 문제가 생길 수 있음.  

**Useful commands**
- Start server: `uvicorn main:app --reload`  
- Ingest PDFs: `python rag/rag_ingest.py`  
- Clear Chroma DB (manual): `rm -r chroma_db` (PowerShell: `Remove-Item -Recurse -Force chroma_db`)