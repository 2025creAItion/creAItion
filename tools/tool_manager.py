import json
from typing import List, Dict, Any
from pydantic import ValidationError  # type: ignore

from .tool_definitions import get_default_tool_specs, ToolSpec


def call_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    단일 툴을 호출합니다.
    
    Args:
        name: 툴 이름
        args: 툴 인자 딕셔너리
    
    Returns:
        툴 실행 결과 딕셔너리
    
    Raises:
        KeyError: 해당 이름의 툴이 없을 경우
        ValidationError: 인자 검증 실패 시
        RuntimeError: 툴 실행 중 오류 발생 시
    """
    tool_specs = get_default_tool_specs()
    tool_spec_map: Dict[str, ToolSpec] = {spec.name: spec for spec in tool_specs}
    
    if name not in tool_spec_map:
        raise KeyError(f"알 수 없는 툴: '{name}'")
    
    spec = tool_spec_map[name]
    
    # Pydantic 모델로 검증
    validated_input = spec.input_model(**args)
    
    # 함수 실행
    result = spec.handler(args)
    
    return result


def execute_tools(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    LLM이 생성한 tool_calls를 실제 함수로 실행하고 결과를 반환합니다.

    Args:
        tool_calls: OpenAI API 응답 형식의 tool_calls 리스트

    Returns:
        Tool 실행 결과 리스트 (OpenAI 형식)
    """
    tool_specs = get_default_tool_specs()
    tool_spec_map: Dict[str, ToolSpec] = {spec.name: spec for spec in tool_specs}

    results: List[Dict[str, Any]] = []

    for tool_call in tool_calls:
        try:
            tool_call_id = tool_call.get("id", "")
            function_info = tool_call.get("function", {})
            function_name = function_info.get("name", "")
            arguments_str = function_info.get("arguments", "{}")

            if not function_name:
                results.append({
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "name":"unknown",
                    "content": "툴 이름이 제공되지 않았습니다."
                })
                continue

            if function_name not in tool_spec_map:
                results.append({
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "name": function_name,
                    "content": f"알 수 없는 툴: '{function_name}'"
                })
                continue

            spec = tool_spec_map[function_name]

            # arguments를 JSON으로 파싱
            try:
                arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
            except json.JSONDecodeError:
                results.append({
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "name": function_name,
                    "content": f"인자 파싱 오류: '{arguments_str}'는 유효한 JSON이 아닙니다."
                })
                continue

            # Pydantic 모델로 검증 및 실행
            try:
                validated_input = spec.input_model(**arguments)
                result = spec.handler(arguments)
                
                # 결과를 JSON 문자열로 변환
                if isinstance(result, dict):
                    content = json.dumps(result, ensure_ascii=False, indent=2)
                elif isinstance(result, (int, float)):
                    content = str(result)
                elif isinstance(result, str):
                    content = result
                else:
                    content = str(result)

            except ValidationError as e:
                content = f"인자 검증 오류: {str(e)}"
            except RuntimeError as e:
                content = f"툴 실행 중 오류 발생: {str(e)}"
            except Exception as e:
                content = f"예상치 못한 오류 발생: {str(e)}"

            results.append({
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name":function_name,
                "content": content
            })

        except Exception as e:
            results.append({
                "tool_call_id": tool_call.get("id", "unknown"),
                "role": "tool",
                "name": tool_call.get("function", {}).get("name","unknown"),
                "content": f"툴 호출 처리 중 오류 발생: {str(e)}"
            })

    return results
