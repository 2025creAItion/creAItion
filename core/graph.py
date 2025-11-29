from langgraph.graph import StateGraph, END  # type: ignore

from .state import AgentState
from .agent_config import get_client, LLM_MODEL


# ------------------------------
# 노드(Node) 함수 – 구조만 정의
# ------------------------------


def call_llm(state: AgentState) -> AgentState:
    """LLM 호출 노드.

    - 역할:
        - state.messages(대화 내역)을 기반으로 LLM을 호출
        - LLM 응답을 messages 리스트에 추가
        - Tool 호출이 필요한 경우 state.tool_calls에 함수 호출 정보를 추가
    - 현재는 구조만 두고, 실제 LLM 호출/Tool 호출 파싱은 TODO로 남김.
    """
    # TODO:
    # 1) client = get_client()로 OpenAI 클라이언트 생성
    # 2) state.messages를 prompt로 사용해 LLM_MODEL 호출
    # 3) 응답 메시지를 state.messages에 append
    # 4) Tool 호출 정보가 있으면 state.tool_calls에 추가
    return state


def execute_tools(state: AgentState) -> AgentState:
    """Tool 실행 노드.

    - 역할:
        - state.tool_calls에 쌓인 Tool 호출 요청들을 실제로 실행
        - 실행 결과를 state.tool_results(또는 messages)에 반영
    - 실제 ToolManager 연동은 TODO (tools 패키지 담당).
    """
    # TODO:
    # 1) tool_manager를 통해 state.tool_calls를 순회하며 실제 함수 호출
    # 2) 결과를 state.messages 또는 별도 필드(예: tool_results)에 추가
    return state


def retrieve_rag(state: AgentState) -> AgentState:
    """RAG 검색 노드.

    - 역할:
        - 사용자의 질문이나 현재 대화 context를 기반으로
          RAG 모듈(rag 패키지)에서 관련 문서를 검색
        - 검색된 문맥을 state에 저장 (예: state.rag_context)
    """
    # TODO:
    # 1) rag.rag_db / document_processor 등을 호출해 관련 문서 검색
    # 2) 검색 결과를 state에 저장 (예: state.rag_context)
    return state


def check_reflection(state: AgentState) -> AgentState:
    """Reflection 수행 여부 판단 노드.

    - 역할:
        - 대화 길이, 특정 키워드 등 기준으로
          이번 턴에서 Reflection(장기 메모리 저장)을 할지 말지 결정
        - 간단히는 state에 플래그를 넣거나, 조건만 평가하는 용도
    """
    # TODO:
    # 1) 대화 턴 수나 중요도 기준으로 reflection 필요 여부 판단
    # 2) 필요하면 state에 표시(예: state.needs_reflection = True)
    return state


def save_reflection(state: AgentState) -> AgentState:
    """Reflection 저장 노드.

    - 역할:
        - reflection_handler를 호출해 대화 내용을 요약/임베딩
        - 장기 메모리(LTM) Chroma DB에 저장
    """
    # TODO:
    # 1) memory.reflection_handler.save_reflection(state)를 호출
    # 2) 요약 결과/메타데이터를 state.reflection_notes 등에 저장
    return state


# ------------------------------
# 조건부 Edge – Tool 사용 여부
# ------------------------------


def should_use_tool(state: AgentState) -> str:
    """Tool 호출 여부에 따라 분기 레이블 반환.

    - 반환값:
        - 'use_tool'  : state.tool_calls가 비어 있지 않은 경우
        - 'no_tool'   : Tool 호출이 필요 없는 경우
    """
    return "use_tool" if state.tool_calls else "no_tool"


# ------------------------------
# 그래프 빌드 함수
# ------------------------------


def build_graph():
    """LangGraph용 StateGraph를 구성하고 컴파일해서 반환.

    - Nodes:
        - call_llm        : LLM 호출 + Tool Call 생성
        - execute_tools   : Tool 실제 실행
        - retrieve_rag    : RAG 검색
        - check_reflection: Reflection 필요 여부 판단
        - save_reflection : Reflection 결과를 LTM에 저장
    - Edges:
        - call_llm → (조건부) → execute_tools or retrieve_rag
        - retrieve_rag → check_reflection → save_reflection → END
    - Interrupt:
        - call_llm 노드에 interrupt=True 옵션을 걸어
          중간 상태를 확인하거나 UI에서 개입할 수 있도록 설계.
    """
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

    # call_llm → tool 여부로 분기
    graph.add_conditional_edges(
        "call_llm",
        should_use_tool,
        {
            "use_tool": "execute_tools",
            "no_tool": "retrieve_rag",
        },
    )

    # RAG → Reflection → 저장 → 끝(END)
    graph.add_edge("retrieve_rag", "check_reflection")
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
