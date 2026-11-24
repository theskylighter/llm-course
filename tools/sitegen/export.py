"""Export the course as JSON for the React app (``web/src/data/lessons.json``).

Each lesson's Markdown is rendered to an HTML fragment here, so the React app only has to
lay it out. The validator and diagram inlining stay in Python.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config
from .builder import INDEX_PLACEHOLDER, lesson_number, reading_minutes
from .config import Course
from .markdown import inline, render_markdown
from .search_index import build_index

INDEX_INTRO = "index"  # content/lessons/index.md is the home page


def export(course: Course | None = None, out_file: Path = config.WEB_DATA_FILE,
           search_file: Path = config.WEB_SEARCH_FILE) -> dict:
    course = course or config.load_course()
    lessons = []
    for lesson in course.lessons:
        text = lesson.source.read_text(encoding="utf-8")
        body, ctx = render_markdown(text)
        lessons.append({
            "slug": lesson.slug,
            "number": lesson_number(lesson.slug),
            "title": lesson.title,
            "chapter": lesson.chapter,
            "syllabus": lesson.syllabus,
            "minutes": reading_minutes(text),
            "html": body,
            "toc": [{"level": lvl, "id": hid, "html": inline(txt)} for lvl, hid, txt in ctx.toc],
        })

    intro_body, intro_ctx = render_markdown((config.LESSONS_DIR / f"{INDEX_INTRO}.md").read_text(encoding="utf-8"))
    intro_parts = intro_body.split(INDEX_PLACEHOLDER)
    data = {
        "site": config.load_site(),
        "courseTitle": course.title,
        "intro": {
            "beforeRoadmap": intro_parts[0],
            "afterRoadmap": intro_parts[1] if len(intro_parts) > 1 else "",
            "toc": [{"level": lvl, "id": hid, "html": inline(txt)} for lvl, hid, txt in intro_ctx.toc],
        },
        "chapters": [
            {"title": ch.title, "lessons": [l.slug for l in ch.lessons]} for ch in course.chapters
        ],
        "lessons": lessons,
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    search_file.write_text(json.dumps(build_index(lessons), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return data
