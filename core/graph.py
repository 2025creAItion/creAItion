from langgraph.graph import StateGraph, END  # type: ignore
from typing import Any, Dict, List
import os

from .state import AgentState
from .agent_config import get_client, LLM_MODEL
from tools.tool_definitions import list_openai_tools
from tools.tool_manager import execute_tools as execute_tool_calls

# ------------------------------
# 노드(Node) 함수 – 구조만 정의
# ------------------------------

# 시스템 프롬프트 파일 경로
SYSTEM_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "prompts", "system.txt"
)

def call_llm(state: AgentState) -> Dict[str, Any]:
    client = get_client()

    # 1) 현재까지의 메시지 복사
    messages: List[Dict[str, Any]] = state.messages.copy()

    # 2) system.txt 파일을 읽어서 시스템 프롬프트를 추가
    system_content = ""
    if os.path.exists(SYSTEM_PROMPT_PATH):
        try:
            with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
                system_content = f.read().strip()
        except Exception:
            pass

    if system_content and (len(messages) == 0 or messages[0].get("role") != "system"):
        messages.insert(0, {"role": "system", "content": system_content})

    # 3) 메시지 정리 (tool_calls / tool_call_id를 pop 하고, user / assistant / system 은 그대로 쓴다.)
    cleaned_messages: List[Dict[str, Any]] = []

    for i, msg in enumerate(messages):
        role = msg.get("role")

        # (1) tool 메시지는 원칙적으로 그대로 둔다.
        #    단, 앞에 대응되는 assistant(tool_calls)가 없으면 버린다.
        if role == "tool":
            if i == 0:
                continue
            prev = messages[i - 1]
            if prev.get("role") != "assistant" or "tool_calls" not in prev:
                continue
            clean_msg = dict(msg)
            cleaned_messages.append(clean_msg)
            continue

        # (2) user / assistant / system 메시지는 그대로 추가한다.
        clean_msg = dict(msg)
        cleaned_messages.append(clean_msg)


    # 4) 마지막 메시지가 tool이면 tool_choice="none"으로 설정하여 더 이상 tool을 부르지 못하게 한다.
    has_tool_result = bool(cleaned_messages) and cleaned_messages[-1].get("role") == "tool"
    tool_choice = "none" if has_tool_result else "auto"

    # 5) OpenAI API에 메시지를 보낸다.
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=cleaned_messages,
        tools=list_openai_tools(),
        tool_choice=tool_choice,
    )

    message = completion.choices[0].message

    # 6) assistant 메시지를 생성한다.
    assistant_msg: Dict[str, Any] = {
        "role": "assistant",
        "content": message.content or "",
    }

    # 7) tool_calls 목록을 생성한다.
    tool_calls_list: List[Dict[str, Any]] = []
    if message.tool_calls:
        for tc in message.tool_calls:
            tool_calls_list.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            })
        
        assistant_msg["tool_calls"] = tool_calls_list

    # 8) 결과를 반환한다.
    result: Dict[str, Any] = {"messages": [assistant_msg]}
    if tool_calls_list:
        result["tool_calls"] = tool_calls_list

    return result


def execute_tools(state: AgentState) -> Dict[str, Any]:
    tool_results = execute_tool_calls(state.tool_calls)

    # tool_results를 tool_messages로 변환한다.
    tool_messages = [
        {
            "role": "tool",
            "content": result["content"],
            "tool_call_id": result["tool_call_id"]
        }
        for result in tool_results
    ]

    # tool_results를 state에 저장한다.
    return {
        "tool_results": tool_results,
        "messages": tool_messages,
        "tool_calls": []
    }


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
    return {}


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
    return {}


def save_reflection(state: AgentState) -> AgentState:
    """Reflection 저장 노드.

    - 역할:
        - reflection_handler를 호출해 대화 내용을 요약/임베딩
        - 장기 메모리(LTM) Chroma DB에 저장
    """
    # TODO:
    # 1) memory.reflection_handler.save_reflection(state)를 호출
    # 2) 요약 결과/메타데이터를 state.reflection_notes 등에 저장
    return {}


# ------------------------------
# 조건부 Edge – Tool 사용 여부
# ------------------------------


def should_use_tool(state: AgentState) -> str:
    """Tool 호출 여부에 따라 분기 레이블 반환.

    - 반환값:
        - 'use_tool'  : state.tool_calls가 비어 있지 않은 경우
        - 'no_tool'   : Tool 호출이 필요 없는 경우
    """
    # tool_calls가 있으면 execute_tools로
    if state.tool_calls:
        return "use_tool"
    
    # Tool 결과를 받은 후인지 확인 (마지막 메시지가 tool 메시지인 경우)
    # 이 경우 최종 답변을 생성하기 위해 retrieve_rag로 이동
    # (execute_tools → call_llm edge가 있으므로 call_llm에서 tool 결과를 처리한 후
    #  tool_calls가 없으면 retrieve_rag로 가서 최종 답변 생성)
    return "no_tool"


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

    # execute_tools → 다시 call_llm으로 돌아가서 tool 결과를 바탕으로 최종 답변 생성
    graph.add_edge("execute_tools", "call_llm")

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