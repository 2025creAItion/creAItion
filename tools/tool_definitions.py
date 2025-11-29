from __future__ import annotations

from typing import Any, Dict, Callable
import os

from pydantic import BaseModel, Field

from datetime import datetime
from dateutil import tz  # type: ignore
import requests  # type: ignore

# -----------------
# Tool: web_search (구글 검색)
# -----------------

class WebSearchInput(BaseModel):
    query: str = Field(..., description="검색할 키워드나 질문을 입력해주세요.")

def web_search(input: WebSearchInput) -> Dict[str, Any]:
    """구글 검색 API를 사용하여 검색 결과를 반환합니다."""
    api_key = os.getenv("GOOGLE_API_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    if not api_key or not search_engine_id:
        raise RuntimeError(
            "구글 검색 API 키가 설정되어 있지 않습니다. "
            ".env 파일에 GOOGLE_API_KEY와 GOOGLE_SEARCH_ENGINE_ID를 모두 설정해주세요."
        )

    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": search_engine_id,
            "q": input.query,
            "num": 5
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        if "items" not in data or len(data["items"]) == 0:
            return {
                "query": input.query,
                "results": [],
                "message": f"'{input.query}'에 대한 검색 결과를 찾을 수 없습니다."
            }

        # 검색 결과 포맷팅
        results = []
        for item in data["items"]:
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", "")
            })

        return {
            "query": input.query,
            "results": results,
            "count": len(results)
        }

    except requests.exceptions.ConnectionError:
        raise RuntimeError("인터넷 연결을 확인할 수 없습니다. 네트워크 연결 상태를 확인해주세요.")
    except requests.exceptions.Timeout:
        raise RuntimeError("검색 요청 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.")
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else "알 수 없음"
        raise RuntimeError(f"검색 서버 오류가 발생했습니다. (상태 코드: {status_code})")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"검색 요청 중 오류가 발생했습니다: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"예상치 못한 오류가 발생했습니다: {str(e)}")

# -----------------
# Tool: get_time (시간)
# -----------------

class GetTimeInput(BaseModel):
    timezone: str = Field(default="Asia/Seoul", description="시간을 얻고자 하는 지역의 IANA timezone name을 입력해주세요. 예: 'Asia/Seoul'")

def get_time(input: GetTimeInput) -> Dict[str, Any]:
    try:
        target_tz = tz.gettz(input.timezone)
        if target_tz is None:
            raise ValueError(f"알 수 없는 시간대: {input.timezone}")

        now = datetime.now(target_tz)

        return {
            "timezone": input.timezone,
            "iso": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
        }
    except Exception as e:
        raise RuntimeError(f"예상치 못한 오류가 발생했습니다: {str(e)}")

# -----------------
# Tool: calculator (계산기)
# -----------------

class CalculatorInput(BaseModel):
    a: float = Field(..., description="첫 번째 피연산자를 입력해주세요.")
    op: str = Field(..., pattern=r"^[+\-*/]$", description="연산자를 입력해주세요. +, -, *, / 중 하나를 선택해주세요.")
    b: float = Field(..., description="두 번째 피연산자를 입력해주세요.")

def calculator(input: CalculatorInput) -> Dict[str, Any]:
    try:
        if input.op == '+':
            val = input.a + input.b
        elif input.op == '-':
            val = input.a - input.b
        elif input.op == '*':
            val = input.a * input.b
        elif input.op == '/':
            if input.b == 0:
                raise RuntimeError("0으로 나눌 수 없습니다.")
            val = input.a / input.b
        else:
            raise RuntimeError(f"지원하지 않는 연산자: {input.op}")

        return {"expression": f"{input.a} {input.op} {input.b}", "value": val}
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"계산 중 오류가 발생했습니다: {str(e)}")

# -------------
# helper: tool spec (schema + handler)
# -------------

class ToolSpec(BaseModel):
    name: str
    description: str
    input_model: Any
    handler: Callable[[Any], Dict[str, Any]]

def as_openai_tool_spec(spec: ToolSpec) -> Dict[str, Any]:
    """OpenAI tools[] spec for function calling (JSON Schema)을 반환합니다."""
    # pydantic 모델의 JSON schema 활용
    schema = spec.input_model.model_json_schema()

    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": schema,
        },
    }

# -------------
# default tool list
# -------------

def get_default_tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="web_search",
            description="구글 검색을 통해 최신 정보를 검색합니다. 날씨, 뉴스, 일반적인 질문 등에 대한 정보를 찾을 때 사용합니다.",
            input_model=WebSearchInput,
            handler=lambda args: web_search(WebSearchInput(**args)),
        ),
        ToolSpec(
            name="get_time",
            description=(
                "주어진 IANA timezone의 **정확한 현재 시간**을 반환하는 도구입니다. "
                "LLM은 시스템 시계를 직접 볼 수 없으므로, "
                "사용자가 '지금 몇 시야?', '현재 시간', '서울/뉴욕 시간' 등을 물어보면 "
                "반드시 이 도구를 호출해서 답해야 합니다."
    ),
            input_model=GetTimeInput,
            handler=lambda args: get_time(GetTimeInput(**args)),
        ),
        ToolSpec(
            name="calculator",
            description="+, -, *, / 연산을 수행하는 계산기입니다.",
            input_model=CalculatorInput,
            handler=lambda args: calculator(CalculatorInput(**args)),
        ),
    ]


def list_openai_tools() -> list[Dict[str, Any]]:
    """LLM에 전달할 OpenAI tools 형식의 리스트를 반환합니다."""
    tool_specs = get_default_tool_specs()
    return [as_openai_tool_spec(spec) for spec in tool_specs]
