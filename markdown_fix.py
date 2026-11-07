"""모델이 만든 Markdown의 한국어 굵은 글씨 깨짐 보정.

CommonMark 규칙상 `**27%**로`처럼 닫는 `**` 앞이 문장부호이고 뒤가 글자이면
굵은 글씨로 인식되지 않아 `**`가 그대로 보인다(여는 쪽 `로**'8구간'**`도 동일).
굵은 글씨 안의 앞뒤 문장부호를 바깥으로 옮겨 `**27**%로` 형태로 바꾼다.
"""

import re

_BOLD = re.compile(r"\*\*(?=\S)([^*\n]+?)(?<=\S)\*\*")
_LEAD = re.compile(r"^[^\w\s]+")
_TRAIL = re.compile(r"[^\w\s]+$")


def _move_punct(m: re.Match) -> str:
    inner = m.group(1)
    lead = _LEAD.search(inner)
    trail = _TRAIL.search(inner)
    lead_s = lead.group() if lead else ""
    trail_s = trail.group() if trail else ""
    core = inner[len(lead_s): len(inner) - len(trail_s)]
    if not core.strip() or core != core.strip():
        return m.group(0)
    return f"{lead_s}**{core}**{trail_s}"


def fix_markdown(text: str) -> str:
    return _BOLD.sub(_move_punct, text)
