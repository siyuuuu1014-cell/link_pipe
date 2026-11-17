"""Generate bounded, class-specific safety guidance."""

from __future__ import annotations


DISCLAIMER = "本结果仅基于可见光图像中的外部视觉特征生成，不能直接确认泄漏介质、内部腐蚀、内部堵塞或精确故障原因，不能替代压力、流量、可燃气体、无损检测及现场人工安全检查。"

RULES = {
    "leak": [
        "设置现场警戒区域，避免无关人员接近。",
        "确认泄漏介质、压力状态和附近点火源。",
        "检查泄漏区域附近的接头、法兰、阀门和密封件。",
        "结合压力、流量和人工现场检查确认泄漏源。",
        "不得仅依据图像直接拆卸或处置带压设备。",
    ],
    "crack": [
        "检查裂纹长度、宽度及是否继续扩展。",
        "安排超声测厚、渗透检测或其他无损检测。",
        "结合设计压力和剩余壁厚评估是否需要降压、修补或更换。",
    ],
    "water": [
        "排查水体来源，判断是否与管道、接头或周边排水有关。",
        "检查是否存在持续滴流、喷射或积水扩展。",
        "避免把普通积水直接判断为管道泄漏。",
    ],
    "none": [
        "当前图像未发现模型可识别的明显异常。",
        "仍需按照既定巡检规程检查压力、流量、设备状态和不可见区域。",
    ],
}


def fixed_advice(detections: list[dict]) -> list[str]:
    present = {item["class_name"] for item in detections}
    if not present:
        return list(RULES["none"])
    advice: list[str] = []
    for name in ("leak", "crack", "water"):
        if name in present:
            advice.extend(RULES[name])
    return advice
