from typing import Annotated, List, Dict
import operator

class AgentState:

    # 누적되는 대화 메시지
    messages: Annotated[List[Dict], operator.add] = []

    # LLM이 만든 Tool 호출 정보
    tool_calls: Annotated[List[Dict], operator.add] = []

    # Tool 실행 결과
    tool_results: Annotated[List[Dict], operator.add] = []

    # RAG 검색 결과
    rag_context: Annotated[List[Dict], operator.add] = []

    # Reflection / LTM 관련 메모
    reflection_notes: Annotated[List[Dict], operator.add] = []


def get_initial_state() -> AgentState:
    """초기 상태 반환 (비어 있는 리스트로 구성)."""
    return AgentState()
