"""Helpers shared by the exporter: reading time and lesson numbering."""
from __future__ import annotations

import re

INDEX_PLACEHOLDER = "<p>&lt;&lt;ROADMAP&gt;&gt;</p>"  # where the React app renders the course roadmap


def reading_minutes(markdown: str) -> int:
    """Rough reading time: ~200 words/min of prose plus 1 min per 40 lines of code."""
    prose = re.sub(r"```.*?```", "", markdown, flags=re.S)
    code_lines = sum(block.count("\n") for block in re.findall(r"```.*?```", markdown, flags=re.S))
    return max(2, round(len(prose.split()) / 200 + code_lines / 40))


def lesson_number(slug: str) -> str:
    """'03_4_encapsulation' -> '3.4'; practice sets and the intro page have no number."""
    m = re.match(r"(\d\d)_(\d)_", slug)
    return f"{int(m.group(1))}.{m.group(2)}" if m else ""
