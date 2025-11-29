from langgraph.graph import StateGraph, END
from typing import Any, Dict, List
from .state import AgentState
from .agent_config import get_client, LLM_MODEL
from rag.rag_db import RagDB
from memory.reflection_handler import ReflectionHandler 
import json


# ------------------------------
# 인스턴스 초기화
# ------------------------------

# RAG DB 인스턴스는 한 번만 생성
try:
    rag_db_instance = RagDB() 
except NameError:
    rag_db_instance = None 

# Reflection Handler 인스턴스 생성 (LTM 관리)
try:
    reflection_handler = ReflectionHandler()
except NameError:
    reflection_handler = None


# ------------------------------
# 노드(Node) 함수 – 구조만 정의
# ------------------------------


def call_llm(state: AgentState) -> Dict[str, Any]:
    """
    LLM 호출 노드. RAG 컨텍스트 또는 LTM을 시스템 프롬프트에 포함하여 호출합니다.
    """
    client = get_client()
    messages: List[Dict[str, Any]] = state.messages
    rag_context = state.rag_context
    
    system_messages = []
    
    # 최신 사용자 쿼리 추출 (LTM 검색 및 LLM 응답 결정을 위해 사용)
    user_query = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else ""

    # 1. Long-Term Memory (LTM) Retrieval (첫 번째 LLM 호출일 경우에만)
    # LTM은 RAG나 Tool 응답을 처리하기 전, 최초 질문에 대해서만 검색합니다.
    is_first_llm_call = not (rag_context or state.tool_calls) 
    
    if is_first_llm_call and reflection_handler and user_query:
        memories = reflection_handler.retrieve_memories(user_query, k=2)
        if memories:
            print(f"[LTM] 과거 기억 {len(memories)}개 검색 완료.")
            memory_str = "\n---\n".join([f"과거 요약: {m}" for m in memories])
            
            system_messages.append({
                "role": "system",
                "content": (
                    "당신은 과거 대화 내용인 '장기 기억'을 바탕으로 사용자의 맥락을 이해합니다. "
                    "답변 시 장기 기억을 활용하여 개인화된 응답을 제공합니다.\n\n"
                    f"--- 장기 기억 ---\n{memory_str}"
                )
            })


    # 2. RAG 컨텍스트 추가 (RAG 검색을 마친 후 재호출된 경우)
    if rag_context:
        print("[LLM_CALL] RAG 컨텍스트를 활용하여 최종 답변 생성.")
        
        context_items = []
        for doc in rag_context:
            source_info = f"({doc.get('source', 'unknown')} 페이지: {doc.get('page', 'N/A')})"
            context_items.append(f"- {doc['content']} {source_info}")
            
        context_str = "\n".join(context_items)
        
        # RAG 시스템 메시지 (LTM 시스템 메시지보다 답변에 더 큰 영향을 줌)
        system_messages.append({
            "role": "system",
            "content": (
                "당신은 사용자 질문에 대해 주어진 '검색 결과'를 기반으로만 답변하는 AI입니다. "
                "검색 결과를 충실히 요약하고, 결과에 출처가 있다면 함께 언급합니다. "
                "검색 결과에 없는 내용은 답변할 수 없습니다.\n\n"
                f"--- 검색 결과 ---\n{context_str}"
            )
        })

    # 시스템 메시지들을 실제 메시지 리스트의 맨 앞에 추가
    if system_messages:
        messages = system_messages + messages # LTM(1) -> RAG(2) -> User Messages 순서로 프롬프트에 구성

    print(f"[LLM_CALL] LLM 호출 시작. 메시지 수: {len(messages)}")
    
    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            # tools=... # 툴 사용을 위한 tools 필드 추가 필요
        )

        assistant_content = completion.choices[0].message.content or ""

        if not assistant_content:
            assistant_content = "LLM으로부터 응답을 받지 못했습니다. (빈 응답)"
        
        assistant_msg = {
            "role": "assistant",
            "content": assistant_content,
        }

        # LLM 응답을 State에 추가
        return {
            "messages": [assistant_msg]
        }

    except Exception as e:
        error_msg = f"LLM 호출 중 오류 발생: {e.__class__.__name__}. {e}"
        print(f" [LLM_CALL_ERROR] {error_msg}")
        return {
            "messages": [
                {"role": "assistant", "content": f"오류 발생: {e.__class__.__name__}."}
            ]
        }


def execute_tools(state: AgentState) -> AgentState:
    """Tool 실행 노드. (미구현)"""
    print("[Tools] Tool 실행 (미구현)")
    return {}

rag_db_instance = RagDB() 

def retrieve_rag(state: AgentState) -> Dict[str, Any]:
    """
    RAG 벡터 DB에서 문서를 검색하는 노드.
    """
    
    user_query = state.messages[-1]["content"] if state.messages else ""

    if not user_query or not rag_db_instance:
        print("[RAG] 검색 실행 불가: 쿼리 또는 DB 인스턴스 없음.")
        return {"rag_context": []}

    print(f"[RAG] 쿼리: '{user_query[:30]}...'로 RAG 검색 시작.")
    
    try:
        retrieved_docs = rag_db_instance.retrieve_documents(user_query, k=5)
    except Exception as e:
        print(f"❌ [RAG_ERROR] RAG DB 검색 중 오류 발생: {e}")
        return {"rag_context": []}
        
    print(f"[RAG] 검색 완료. {len(retrieved_docs)}개의 청크를 State에 추가합니다.")

    return {
        "rag_context": retrieved_docs
    }


def check_reflection(state: AgentState) -> AgentState:
    """Reflection 수행 여부 판단 노드. (미구현)"""
    # TODO: 대화 턴 수나 중요도 기준으로 reflection 필요 여부 판단 로직 구현
    print("[Reflection] Reflection 필요 여부 확인 (미구현)")
    # 모든 턴에서 저장하도록 가정하고 플래그를 State에 추가 가능
    # 예: return {"needs_reflection": True}
    return {}


def save_reflection(state: AgentState) -> Dict[str, Any]:
    """
    RAG 컨텍스트와 최종 답변을 바탕으로 LLM을 호출하여 Reflection 노트를 요약 생성합니다.
    """
    
    # 1. 필요한 정보 추출
    
    # messages[-2] = 사용자 질문, messages[-1] = LLM의 최종 답변
    if state.messages and len(state.messages) >= 2:
        last_user_query = state.messages[-2]["content"] 
        final_answer = state.messages[-1]["content"]
    else:
        last_user_query = "N/A (사용자 질문 없음)"
        final_answer = "N/A (최종 답변 없음)"

    # RAG 컨텍스트를 보기 좋은 문자열로 조합
    rag_context_list = [f"Source {i+1} Content:\n{c['content']}" for i, c in enumerate(state.rag_context)]
    rag_context_str = "\n---\n".join(rag_context_list)
    
    # 2. Reflection Note 생성을 위한 프롬프트 정의
    system_prompt = (
        "당신은 장기 기억(Long-Term Memory) 저장을 담당하는 Reflection Agent입니다. "
        "아래 제공된 '사용자 질문', '검색된 컨텍스트', 그리고 '최종 답변'을 바탕으로, "
        "이 상호작용의 핵심 내용을 요약하여 향후 검색에 유용할 수 있는 'Reflection Note'를 한국어로 작성하십시오. "
        "노트는 문장 형태로 3줄 이내로 간결하고, 사실을 요약하여 작성되어야 합니다. 다른 불필요한 서문이나 맺음말은 포함하지 마십시오."
    )
    
    user_content = (
        f"--- 사용자 질문 ---\n{last_user_query}\n\n"
        f"--- 검색된 컨텍스트 ---\n{rag_context_str}\n\n"
        f"--- 최종 답변 ---\n{final_answer}\n\n"
        "이 세 가지 요소를 종합하여 핵심 요약 Reflection Note를 생성하십시오."
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    # 3. LLM 호출 (요약 생성)
    summarized_note = "요약 생성 실패 (LLM 호출 전)"
    try:
        client = get_client() # 클라이언트 재호출 (안정성)
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.0  # 요약이므로 낮은 온도 설정
        )
        # 요약된 Reflection Note 내용
        summarized_note = completion.choices[0].message.content or "LLM으로부터 내용 받기 실패"
    except Exception as e:
        print(f"[Reflection Error] LLM 요약 중 오류 발생: {e}")
        summarized_note = f"LLM 요약 중 오류 발생: {e}"

    # 4. State에 저장할 결과 포맷팅
    reflection_note = {
        "user_query": last_user_query,
        "context_used_count": len(state.rag_context),
        "note": summarized_note, # LLM이 생성한 요약된 내용을 저장
    }

    # LangGraph State에 'reflection_notes'를 추가
    return {
        "reflection_notes": [reflection_note]
    }


# ------------------------------
# 조건부 Edge – Tool 사용/RAG 완료 여부
# ------------------------------


def should_use_tool(state: AgentState) -> str:
    """Tool 호출 여부, RAG 완료 여부에 따라 다음 노드를 결정합니다."""
    
    # 1. Tool 사용 여부 확인
    if state.tool_calls:
        print("[Router] Tool 호출 감지: execute_tools로 이동.")
        return "use_tool"

    # 2. RAG 컨텍스트 존재 여부 확인
    if state.rag_context and len(state.rag_context) > 0:
        print("[Router] RAG 컨텍스트 존재. Reflection 체크로 이동 (최종 답변 생성).")
        # RAG 컨텍스트를 사용했으므로 다음 턴을 위해 초기화 (AgentState의 Annotated[list, add]가 이를 처리함)
        return "final_answer_to_reflection" 

    # 3. Tool 호출도 없고 RAG 컨텍스트도 없으면 RAG 검색 시도
    print("[Router] Tool 호출 없음. RAG 검색 (retrieve_rag)으로 이동.")
    return "no_tool"


# ------------------------------
# 그래프 빌드 함수
# ------------------------------


def build_graph():
    """LangGraph용 StateGraph를 구성하고 컴파일해서 반환합니다."""
    graph = StateGraph(AgentState)

    # 노드 등록
    # call_llm에 interrupt=True를 설정해, 이 노드 이후에
    # LangGraph의 interrupt 기능을 사용할 수 있도록 설계.
    graph.add_node("call_llm", call_llm, interrupt=True)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("retrieve_rag", retrieve_rag)
    graph.add_node("check_reflection", check_reflection)
    graph.add_node("save_reflection", save_reflection)

    # 시작점 (START)
    graph.set_entry_point("call_llm")

    # 1. call_llm → tool / RAG / Reflection으로 분기
    graph.add_conditional_edges(
        "call_llm",
        should_use_tool,
        {
            "use_tool": "execute_tools",                   
            "no_tool": "retrieve_rag",                     
            "final_answer_to_reflection": "check_reflection", 
        },
    )

    # 2. Tool 실행 → LLM으로 복귀
    graph.add_edge("execute_tools", "call_llm") 

    # 3. RAG 검색 → LLM 재호출
    graph.add_edge("retrieve_rag", "call_llm") 

    # 4. Reflection → 저장 → 끝(END)
    graph.add_edge("check_reflection", "save_reflection")
    graph.add_edge("save_reflection", END)

    # 컴파일된 RunnableGraph 반환
    return graph.compile()


# ------------------------------
# (선택) Stream 실행 예시 헬퍼
# ------------------------------


def build_and_stream(initial_state: AgentState):
    """그래프를 빌드하고, stream 모드로 실행하는 예시 함수.

    - 실제 UI(FastAPI/Gradio)에서는 이 함수를 참고해
      runnable.stream(...) 패턴을 그대로 사용하면 된다.
    - 과제에서는 'Stream을 지원하는 구조로 설계했다'는 것을
      보여주는 용도의 데모 수준으로만 사용해도 충분하다.
    """
    runnable = build_graph()

    # LangGraph의 stream 기능 사용 예시
    # (실제 코드에서는 이 부분을 Gradio/웹소켓 등으로 연결 가능)
    for event in runnable.stream(initial_state.dict()):
        # TODO: event를 로그로 남기거나, UI로 중계하는 로직 추가 가능
        print("stream event:", event)
