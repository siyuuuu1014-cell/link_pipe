"""Optional Ollama advice polishing with strict fallback behavior."""

from __future__ import annotations

import httpx

from backend.config import get_settings


SYSTEM_CONSTRAINT = "你只负责润色给定建议。不得补充未检测到的故障；不得确认内部状态；不得确定精确泄漏原因；不得替代人工和传感器检查；输出简洁中文。"


async def polish_advice(context: dict, fixed: list[str], enabled: bool) -> list[str]:
    settings = get_settings()
    if not enabled or not settings.enable_llm:
        return fixed
    prompt = {"检测结果": context, "固定规则建议": fixed, "限制条件": SYSTEM_CONSTRAINT}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                f"{settings.ollama_base_url.rstrip('/')}/api/generate",
                json={"model": settings.ollama_model, "prompt": str(prompt), "stream": False},
            )
            response.raise_for_status()
            text = response.json().get("response", "").strip()
            return [text] if text else fixed
    except (httpx.HTTPError, ValueError):
        return fixed
