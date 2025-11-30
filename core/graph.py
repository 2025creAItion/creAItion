from langgraph.graph import StateGraph, END  # type: ignore
from typing import Any, Dict, List
import os

from .state import AgentState
from .agent_config import get_client, LLM_MODEL
from tools.tool_definitions import list_openai_tools
from tools.tool_manager import execute_tools as execute_tool_calls

from rag.rag_db import RagDB
from memory.reflection_handler import ReflectionHandler

# ------------------------------
# 전역 인스턴스 (RAG / Reflection)
# ------------------------------

try:
    rag_db_instance = RagDB()
except Exception:
    rag_db_instance = None

try:
    reflection_handler = ReflectionHandler()
except Exception:
    reflection_handler = None

# 시스템 프롬프트 파일 경로
SYSTEM_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "prompts",
    "system.txt",
)

# ------------------------------
# call_llm
# ------------------------------


def call_llm(state: AgentState) -> Dict[str, Any]:
    """
    LLM 호출 노드.
    - system.txt 기반 기본 시스템 프롬프트
    - Long-Term Memory(LTM) 시스템 메시지
    - RAG 컨텍스트 시스템 메시지
    - Tool calling 지원
    """
    client = get_client()

    # -------------------------
    # 0. 현재까지의 messages / rag_context 가져오기
    # -------------------------
    raw_messages: List[Dict[str, Any]] = state.messages.copy()
    rag_context = state.rag_context

    # -------------------------
    # 1. system.txt + LTM + RAG 시스템 메시지 만들기
    # -------------------------
    system_messages: List[Dict[str, Any]] = []

    # 1-1) system.txt
    if os.path.exists(SYSTEM_PROMPT_PATH):
        try:
            with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
                base_system = f.read().strip()
            if base_system:
                system_messages.append(
                    {
                        "role": "system",
                        "content": base_system,
                    }
                )
        except Exception:
            # system.txt 읽기 실패 시 무시
            pass

    # 1-2) 최신 user 쿼리
    user_query = ""
    if raw_messages and raw_messages[-1].get("role") == "user":
        user_query = raw_messages[-1].get("content", "")

    # 1-3) LTM (최초 LLM 호출일 때만)
    is_first_llm_call = not (rag_context or state.tool_calls)

    if is_first_llm_call and reflection_handler and user_query:
        try:
            memories = reflection_handler.retrieve_memories(user_query, k=2)
        except Exception:
            memories = []

        if memories:
            print(f"[LTM] 과거 기억 {len(memories)}개 검색 완료.")
            memory_str = "\n---\n".join([f"과거 요약: {m}" for m in memories])

            system_messages.append(
                {
                    "role": "system",
                    "content": (
                        "당신은 과거 대화 내용인 '장기 기억'을 바탕으로 사용자의 맥락을 이해합니다. "
                        "답변 시 장기 기억을 활용하여 개인화된 응답을 제공합니다.\n\n"
                        f"--- 장기 기억 ---\n{memory_str}"
                    ),
                }
            )

    # 1-4) RAG 시스템 메시지
    if rag_context:
        print("[LLM_CALL] RAG 컨텍스트를 활용하여 최종 답변 생성.")

        context_items = []
        for doc in rag_context:
            source_info = f"({doc.get('source', 'unknown')} 페이지: {doc.get('page', 'N/A')})"
            context_items.append(f"- {doc.get('content', '')} {source_info}")

        context_str = "\n".join(context_items)

        system_messages.append(
            {
                "role": "system",
                "content": (
                    "당신은 사용자 질문에 대해 주어진 '검색 결과'를 기반으로만 답변하는 AI입니다. "
                    "검색 결과를 충실히 요약하고, 결과에 출처가 있다면 함께 언급합니다. "
                    "검색 결과에 없는 내용은 답변할 수 없습니다.\n\n"
                    f"--- 검색 결과 ---\n{context_str}"
                ),
            }
        )

    # system 메시지 + 기존 대화 메시지 합치기
    messages: List[Dict[str, Any]] = system_messages + raw_messages


    # -------------------------
    # 2. messages 정리 (tool 규칙 위반하는 메시지 제거)
    # -------------------------
    cleaned_messages: List[Dict[str, Any]] = []

    for i, msg in enumerate(messages):
        role = msg.get("role")

        if role == "tool":
            # tool 메시지는 바로 앞 assistant가 tool_calls를 가지고 있고,
            # 그 중 하나의 id와 tool_call_id가 매칭될 때만 유지
            if i == 0:
                # 맨 앞에 tool이 있을 수는 없음
                print(f"[LLM_CALL] 잘못된 tool 메시지 (맨 앞) 제거: {msg}")
                continue

            prev = messages[i - 1]
            if prev.get("role") != "assistant":
                print(f"[LLM_CALL] 앞 메시지가 assistant가 아닌 tool 메시지 제거: {msg}")
                continue

            prev_tool_calls = prev.get("tool_calls") or []
            tool_call_id = msg.get("tool_call_id")

            if not tool_call_id:
                print(f"[LLM_CALL] tool_call_id 없는 tool 메시지 제거: {msg}")
                continue

            # prev_tool_calls 는 우리가 assistant_msg["tool_calls"]로 넣은 리스트 형태
            if any(tc.get("id") == tool_call_id for tc in prev_tool_calls):
                cleaned_messages.append(msg)
            else:
                print(f"[LLM_CALL] 매칭되는 tool_call_id가 없는 tool 메시지 제거: {msg}")
            continue

        # user / assistant / system 은 그대로 넣되, 기존 필드는 건드리지 않음
        cleaned_messages.append(dict(msg))


    # -------------------------
    # 3. tool_choice 설정
    # -------------------------
    has_tool_result = bool(cleaned_messages) and cleaned_messages[-1].get("role") == "tool"
    tool_choice = "none" if has_tool_result else "auto"

    tools = list_openai_tools()

    print(f"[LLM_CALL] LLM 호출 시작. 메시지 수: {len(cleaned_messages)}, tool_choice={tool_choice}")

    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=cleaned_messages,
            tools=tools,
            tool_choice=tool_choice,
        )

        message = completion.choices[0].message

        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }

        # tool_calls 있으면 state에 넘길 구조로 변환
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

        result: Dict[str, Any] = {"messages": [assistant_msg]}
        if tool_calls_list:
            result["tool_calls"] = tool_calls_list

        return result

    except Exception as e:
        error_msg = f"LLM 호출 중 오류 발생: {e.__class__.__name__}. {e}"
        print(f"[LLM_CALL_ERROR] {error_msg}")
        return {
            "messages": [
                {"role": "assistant", "content": f"오류 발생: {e.__class__.__name__}."}
            ]
        }


# ------------------------------
# execute_tools
# ------------------------------


def execute_tools(state: AgentState) -> Dict[str, Any]:
    """
    Tool 실행 노드.
    - state.tool_calls 에 쌓인 tool 호출들을 실제로 실행
    - 실행 결과를 tool_messages(role='tool')로 변환해 messages 에 추가
    """
    if not state.tool_calls:
        print("[Tools] 실행할 tool_calls 가 없습니다.")
        return {}

    print(f"[Tools] {len(state.tool_calls)}개 tool 실행 시작.")
    tool_results = execute_tool_calls(state.tool_calls)

    tool_messages = [
        {
            "role": "tool",
            "content": result["content"],
            "tool_call_id": result["tool_call_id"],
        }
        for result in tool_results
    ]

    # 실행한 tool_calls 는 비워서 무한 루프 방지
    return {
        "tool_results": tool_results,
        "messages": tool_messages,
        "tool_calls": [],
    }


# ------------------------------
# retrieve_rag
# ------------------------------


def retrieve_rag(state: AgentState) -> Dict[str, Any]:
    """
    RAG 벡터 DB에서 문서를 검색하는 노드.
    """
    if not state.messages:
        print("[RAG] 메시지가 없어 검색할 수 없습니다.")
        return {"rag_context": []}

    user_query = state.messages[-1].get("content", "")

    if not user_query or not rag_db_instance:
        print("[RAG] 검색 실행 불가: 쿼리 또는 DB 인스턴스 없음.")
        return {"rag_context": []}

    print(f"[RAG] 쿼리: '{user_query[:30]}...'로 RAG 검색 시작.")

    try:
        retrieved_docs = rag_db_instance.retrieve_documents(user_query, k=5)
    except Exception as e:
        print(f"[RAG_ERROR] RAG DB 검색 중 오류 발생: {e}")
        return {"rag_context": []}

    if not retrieved_docs:
        # 검색 결과가 없더라도, 루프 방지를 위해 '빈 결과' 더미를 넣어줌
        print("[RAG] 검색 결과 없음. dummy 컨텍스트 1개 추가.")
        retrieved_docs = [
            {
                "content": "RAG 검색 결과가 없습니다. (dummy context)",
                "source": "RAG",
                "page": 0,
            }
        ]
    else:
        print(f"[RAG] 검색 완료. {len(retrieved_docs)}개의 청크를 State에 추가합니다.")

    return {"rag_context": retrieved_docs}


# ------------------------------
# check_reflection
# ------------------------------


def check_reflection(state: AgentState) -> Dict[str, Any]:
    """
    Reflection 수행 여부 판단 노드.

    간단한 규칙:
    - RAG 컨텍스트가 있고
    - 대화 메시지가 어느 정도 길면(예: 4개 이상)
    => 이번 턴을 장기 기억에 저장할 가치가 있다고 보고 should_reflect=True 플래그를 남긴다.
    """
    msg_len = len(state.messages)
    has_rag = bool(state.rag_context)

    should_reflect = has_rag and msg_len >= 4

    print(
        f"[Reflection] check_reflection: messages={msg_len}, "
        f"has_rag={has_rag} -> should_reflect={should_reflect}"
    )

    # save_reflection 에서 이 플래그를 참고할 수 있도록 reflection_notes 에 메타정보를 남김
    return {
        "reflection_notes": [
            {
                "meta": True,
                "should_reflect": should_reflect,
            }
        ]
    }


# ------------------------------
# save_reflection
# ------------------------------


def save_reflection(state: AgentState) -> Dict[str, Any]:
    """
    RAG 컨텍스트와 최종 답변을 바탕으로 LLM을 호출하여 Reflection 노트를 요약 생성하고,
    필요시 ReflectionHandler 를 이용해 LTM 에 저장한다.
    """
    # 0. check_reflection 에서 남긴 should_reflect 플래그 확인
    should_reflect = True  # 기본값: 하겠다고 가정
    for note in state.reflection_notes:
        if note.get("meta") and "should_reflect" in note:
            should_reflect = bool(note["should_reflect"])
            break

    if not should_reflect:
        print("[Reflection] 이번 턴은 Reflection 을 건너뜁니다.")
        return {}

    # 1. 필요한 정보 추출 (messages[-2] = user, messages[-1] = assistant 라고 가정)
    if state.messages and len(state.messages) >= 2:
        last_user_query = state.messages[-2].get("content", "")
        final_answer = state.messages[-1].get("content", "")
    else:
        last_user_query = "N/A (사용자 질문 없음)"
        final_answer = "N/A (최종 답변 없음)"

    rag_context_list = [
        f"Source {i+1} Content:\n{c.get('content', '')}"
        for i, c in enumerate(state.rag_context)
    ]
    rag_context_str = "\n---\n".join(rag_context_list)

    # 2. Reflection Note 생성을 위한 프롬프트 정의
    system_prompt = (
        "당신은 장기 기억(Long-Term Memory) 저장을 담당하는 Reflection Agent입니다. "
        "아래 제공된 '사용자 질문', '검색된 컨텍스트', 그리고 '최종 답변'을 바탕으로, "
        "이 상호작용의 핵심 내용을 요약하여 향후 검색에 유용할 수 있는 'Reflection Note'를 한국어로 작성하십시오. "
        "노트는 문장 형태로 3줄 이내로 간결하고, 사실을 요약하여 작성되어야 합니다. "
        "다른 불필요한 서문이나 맺음말은 포함하지 마십시오."
    )

    user_content = (
        f"--- 사용자 질문 ---\n{last_user_query}\n\n"
        f"--- 검색된 컨텍스트 ---\n{rag_context_str}\n\n"
        f"--- 최종 답변 ---\n{final_answer}\n\n"
        "이 세 가지 요소를 종합하여 핵심 요약 Reflection Note를 생성하십시오."
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # 3. LLM 호출 (요약 생성)
    summarized_note = "요약 생성 실패 (LLM 호출 전)"
    try:
        client = get_client()
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.0,
        )
        summarized_note = (
            completion.choices[0].message.content or "LLM으로부터 내용 받기 실패"
        )
    except Exception as e:
        print(f"[Reflection Error] LLM 요약 중 오류 발생: {e}")
        summarized_note = f"LLM 요약 중 오류 발생: {e}"

    reflection_note = {
        "user_query": last_user_query,
        "context_used_count": len(state.rag_context),
        "note": summarized_note,
    }

    # 4. 실제 LTM 에도 저장 (가능한 경우)
    if reflection_handler:
        try:
            if hasattr(reflection_handler, "save_reflection"):
                reflection_handler.save_reflection(reflection_note)
        except Exception as e:
            print(f"[Reflection] LTM 저장 중 오류 발생: {e}")

    print("[Reflection] Reflection Note 1개 생성 및 state에 추가.")

    return {
        "reflection_notes": [reflection_note]
    }


# ------------------------------
# 라우터: should_use_tool
# ------------------------------


def should_use_tool(state: AgentState) -> str:
    """
    Tool 호출 여부, RAG 완료 여부에 따라 다음 노드를 결정합니다.

    반환:
        - 'use_tool'                : tool_calls 가 있는 경우
        - 'no_tool'                 : 아직 RAG 검색을 수행해야 하는 경우
        - 'final_answer_to_reflection' : RAG 까지 끝났고, 이제 Reflection 단계로 넘어가는 경우
    """
    # 1. Tool 사용 여부 확인
    if state.tool_calls:
        print("[Router] Tool 호출 감지: execute_tools 로 이동.")
        return "use_tool"

    # 2. RAG 컨텍스트 존재 여부 확인
    if state.rag_context and len(state.rag_context) > 0:
        print(
            "[Router] RAG 컨텍스트 존재. 최종 답변까지 생성되었다고 보고 Reflection 단계로 이동."
        )
        return "final_answer_to_reflection"

    # 3. Tool 호출도 없고 RAG 컨텍스트도 없으면 (대개 첫 호출) → RAG 검색 시도
    print("[Router] Tool 호출 없음. RAG 검색(retrieve_rag)으로 이동.")
    return "no_tool"


# ------------------------------
# 그래프 빌드 함수
# ------------------------------


def build_graph():
    """LangGraph용 StateGraph를 구성하고 컴파일해서 반환합니다."""
    graph = StateGraph(AgentState)

    # 노드 등록
    graph.add_node("call_llm", call_llm, interrupt=True)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("retrieve_rag", retrieve_rag)
    graph.add_node("check_reflection", check_reflection)
    graph.add_node("save_reflection", save_reflection)

    # 시작점 (START)
    graph.set_entry_point("call_llm")

    # 1. call_llm → tool / RAG / Reflection 으로 분기
    graph.add_conditional_edges(
        "call_llm",
        should_use_tool,
        {
            "use_tool": "execute_tools",
            "no_tool": "retrieve_rag",
            "final_answer_to_reflection": "check_reflection",
        },
    )

    # 2. Tool 실행 → 다시 LLM으로 복귀 (tool 결과를 바탕으로 최종 답변 생성)
    graph.add_edge("execute_tools", "call_llm")

    # 3. RAG 검색 → RAG 컨텍스트를 포함해 LLM 재호출
    graph.add_edge("retrieve_rag", "call_llm")

    # 4. Reflection → 저장 → 끝(END)
    graph.add_edge("check_reflection", "save_reflection")
    graph.add_edge("save_reflection", END)

    return graph.compile()


# ------------------------------
# (선택) Stream 실행 예시 헬퍼
# ------------------------------


def build_and_stream(initial_state: AgentState):
    """그래프를 빌드하고, stream 모드로 실행하는 예시 함수."""
    runnable = build_graph()

    for event in runnable.stream(initial_state.dict()):
        print("stream event:", event)