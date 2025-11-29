from __future__ import annotations

from typing import Any, Dict, List

import gradio as gr

from core.graph import build_graph
from core.state import AgentState

# ------------------------------
# LangGraph runnable 준비
# ------------------------------

# 앱 시작 시 한 번만 그래프 컴파일
graph = build_graph()


def _extract_attr(result: Any, name: str, default: Any):
    """
    result가 dict일 수도 있고 AgentState 인스턴스일 수도 있으니
    둘 다 대응하기 위한 헬퍼 함수.
    """
    if isinstance(result, dict):
        return result.get(name, default)
    # AgentState 같은 객체인 경우
    return getattr(result, name, default)


# ------------------------------
# Gradio에서 쓰는 채팅 함수
# ------------------------------

def chat_fn(
    message: str,
    history: List[Dict[str, Any]] | None,
    tool_log_state: Any,
    rag_log_state: Any,
    memory_log_state: Any,
):
    """
    - message: 사용자가 방금 입력한 텍스트
    - history: 지금까지의 대화 (Chatbot이 들고 있는 messages)
    - tool_log_state, rag_log_state, memory_log_state:
        오른쪽 JSON 패널들이 들고 있는 상태
    """
    if history is None:
        history = []

    # 공백 입력시 그냥 그대로 반환
    if not str(message).strip():
        return history, tool_log_state, rag_log_state, memory_log_state

    # 1) 유저 메시지를 history에 추가
    updated_messages = history + [
        {"role": "user", "content": str(message)}
    ]

    # 2) LangGraph에 넘길 초기 state 구성
    #    (tool_* / rag_* / reflection_* 은 그래프 안에서 채워질 것)
    state_dict: Dict[str, Any] = {
        "messages": updated_messages,
        "tool_calls": [],
        "tool_results": [],
        "rag_context": [],
        "reflection_notes": [],
    }

    # 3) LangGraph 실행
    result = graph.invoke(state_dict)

    # 4) 결과에서 각 필드를 꺼냄
    raw_messages = _extract_attr(result, "messages", updated_messages)
    tool_calls = _extract_attr(result, "tool_calls", tool_log_state or [])
    rag_context = _extract_attr(result, "rag_context", rag_log_state or [])
    reflection_notes = _extract_attr(result, "reflection_notes", memory_log_state or [])

    # messages에는 LLM 응답까지 포함된 전체 대화가 들어있다고 가정
    # (call_llm 노드가 state.messages에 assistant 메시지를 append 하는 구조)
    normalized_messages : List[Dict[str,str]] = []

    if isinstance(raw_messages, dict):
        raw_messages = [raw_messages]
    
    if not isinstance(raw_messages, list):
        normalized_messages.append(
            {"role": "assistant", "content": str(raw_messages)}
        )
    else:
        for m in raw_messages:
            if isinstance(m, dict) and "role" in m and "content" in m:
                normalized_messages.append(
                    {
                        "role" : str(m["role"]),
                        "content": str(m["content"]),
                    }
                )
            else:
                normalized_messages.append(
                    {
                        "role": "assistant",
                        "content": str(m),
                    }
                )

    # Chatbot, Tool 로그, RAG 로그, Memory 로그를 한 번에 업데이트
    return normalized_messages, tool_calls, rag_context, reflection_notes


# ------------------------------
# Gradio Blocks UI 정의
# ------------------------------

def create_gradio_app():
    """
    FastAPI에서 mount할 Gradio Blocks를 반환하는 함수.
    """
    with gr.Blocks(title="LangGraph ReAct Agent") as demo:
        gr.Markdown(
            """
            # LangGraph ReAct Agent 🤖 made by creAItion TEAM
            - LLM + Tool-calling + RAG + Memory + ReAct
            - 아래 채팅창에 질문을 입력해보세요.
            """
        )

        with gr.Row():
            # ------------------------------
            # 왼쪽: 기본 챗봇 영역
            # ------------------------------
            with gr.Column(scale=3):
                chat = gr.Chatbot(
                    label="ReAct Agent Chat",
                    height=500,
                    value=[],  # 초기 메시지 리스트 (비어있음)
                )
                user_input = gr.Textbox(
                    label="메시지 입력",
                    placeholder="질문을 입력하세요...",
                    lines=2,
                )
                send_btn = gr.Button("전송", variant="primary")

                # ------------------------------
                # 오른쪽: Tool / RAG / Memory 상태 패널
                # ------------------------------
            with gr.Column(scale=2):
                gr.Markdown("### 🔧 Tool / RAG / Memory 상태")

                tool_log = gr.JSON(
                    label="Tool 호출 로그",
                    value=[],
                )
                rag_log = gr.JSON(
                    label="RAG 검색 결과",
                    value=[],
                )
                memory_log = gr.JSON(
                    label="Memory / Reflection 상태",
                    value=[],
                )

        # ------------------------------
        # 이벤트 연결
        # ------------------------------

        # Enter로 전송
        user_input.submit(
            fn=chat_fn,
            inputs=[user_input, chat, tool_log, rag_log, memory_log],
            outputs=[chat, tool_log, rag_log, memory_log],
        )

        # 버튼 클릭으로 전송
        send_btn.click(
            fn=chat_fn,
            inputs=[user_input, chat, tool_log, rag_log, memory_log],
            outputs=[chat, tool_log, rag_log, memory_log],
        )

        return demo


if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
