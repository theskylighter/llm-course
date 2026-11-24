"""Build the site search index (``web/src/data/search.json``) from the rendered lesson HTML.

One record per searchable section: every ``h2``/``h3`` heading of a lesson, plus every
collapsible deep dive (``details.deep``), which gets its own record so a hit opens that
walkthrough directly. The browser loads this file lazily the first time search is opened.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser

_SKIP_TAGS = {"script", "style", "svg", "button"}
_BLOCK_TAGS = {"p", "div", "li", "tr", "td", "th", "br", "pre", "ul", "ol", "table", "h1", "h2", "h3", "h4", "details"}
_TEX_COMMAND = re.compile(r"\\([a-zA-Z]+)")
# Formatting-only commands are dropped; symbol commands become the symbol; any other command
# (\pi, \beta, \log, \softmax ...) keeps its name so "beta" finds the formulas that use it.
_TEX_DROP = {"mathrm", "mathcal", "mathbf", "mathbb", "mathit", "mathsf", "text", "textbf", "left", "right",
             "big", "Big", "bigg", "Bigg", "frac", "quad", "qquad", "operatorname", "displaystyle", "limits",
             "hat", "tilde", "bar", "vec", "overline", "underbrace", "mid"}
_TEX_SYMBOL = {"times": "×", "cdot": "·", "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "to": "→",
               "rightarrow": "→", "approx": "≈", "in": "∈", "ne": "≠", "pm": "±", "sim": "~", "ldots": "…",
               "dots": "…", "infty": "∞", "top": "ᵀ", "Vert": "‖", "lVert": "‖", "rVert": "‖", "prec": "≺",
               "succ": "≻", "odot": "⊙", "mapsto": "→", "propto": "∝"}


def _tex_command(m: re.Match) -> str:
    name = m.group(1)
    if name in _TEX_SYMBOL:
        return " " + _TEX_SYMBOL[name] + " "
    return " " if name in _TEX_DROP else " " + name + " "


def tex_to_words(tex: str) -> str:
    """``\\mathrm{KeepTopK}(H(x), k)`` -> ``KeepTopK (H(x), k)``: drop the markup, keep the words."""
    s = tex.replace("}{", " ").replace("\\\\", " ")
    s = _TEX_COMMAND.sub(_tex_command, s)
    return re.sub(r"[{}\\]", "", s)


class _Collector(HTMLParser):
    def __init__(self, slug: str, title: str, chapter: str, number: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base = {"slug": slug, "title": title, "chapter": chapter, "number": number}
        self.records: list[dict] = []
        self.section = self._new(title, "", "section")  # text before the first heading
        self.deep: dict | None = None
        self.deep_depth = 0       # <details> nesting depth inside the open deep dive
        self.skip_depth = 0
        self.heading: dict | None = None  # the h2/h3 being read
        self.in_deep_title = False
        self.in_deep_meta = False
        self.in_summary = False   # summary text ("Show Answer", labels) is noise, except a deep dive's title
        self.math_depth = 0

    def _new(self, heading: str, anchor: str, kind: str) -> dict:
        rec = {**self.base, "heading": heading, "anchor": anchor, "kind": kind, "_text": []}
        self.records.append(rec)
        return rec

    @property
    def current(self) -> dict:
        return self.deep or self.section

    # -- tags --------------------------------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        classes = (a.get("class") or "").split()
        if tag in _SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag == "summary":
            self.in_summary = True
            return
        if tag == "details":
            if self.deep is not None:
                self.deep_depth += 1
            elif "deep" in classes:
                self.deep = self._new("", a.get("id") or "", "deep")
                self.deep_depth = 1
            return
        if self.deep is not None and tag == "span":
            if "deep-title" in classes:
                self.in_deep_title = True
            elif "deep-meta" in classes:
                self.in_deep_meta = True
        if self.deep is None and tag in ("h2", "h3") and a.get("id"):
            self.heading = self._new("", a["id"], "section")
            self.section = self.heading
            return
        data_tex = a.get("data-tex")
        if data_tex is not None and "math" in classes:
            self.current["_text"].append(" " + tex_to_words(data_tex) + " ")
            self.math_depth += 1
            return
        if tag in _BLOCK_TAGS:
            self.current["_text"].append(" ")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK_TAGS and not self.skip_depth:
            self.current["_text"].append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            if self.skip_depth:
                self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag == "summary":
            self.in_summary = False
            return
        if tag == "details" and self.deep is not None:
            self.deep_depth -= 1
            if self.deep_depth == 0:
                self.deep = None
            return
        if tag in ("h2", "h3"):
            self.heading = None
        if tag == "span":
            self.in_deep_title = self.in_deep_meta = False
        if self.math_depth and tag in ("span", "div"):
            self.math_depth -= 1
            return
        if tag in _BLOCK_TAGS:
            self.current["_text"].append(" ")

    def handle_data(self, data: str) -> None:
        if self.skip_depth or self.math_depth:
            return
        if self.in_summary and not (self.in_deep_title or self.in_deep_meta):
            return
        if self.heading is not None:
            self.heading["heading"] += data
        elif self.in_deep_title and self.deep is not None:
            self.deep["heading"] += data
        elif self.in_deep_meta and self.deep is not None:
            self.deep["_text"].append(" " + data + " ")  # authors/year are searchable, not the heading
        else:
            self.current["_text"].append(data)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def lesson_records(lesson: dict) -> list[dict]:
    parser = _Collector(lesson["slug"], lesson["title"], lesson["chapter"], lesson["number"])
    parser.feed(lesson["html"])
    parser.close()
    out = []
    for rec in parser.records:
        text = _clean("".join(rec.pop("_text")))
        heading = _clean(rec["heading"])
        if rec["kind"] == "deep":
            heading = "Deep dive: " + heading
        if not text and not heading:
            continue
        if rec["kind"] == "section" and not rec["anchor"] and len(text) < 40:
            continue  # an empty lead-in before the first heading
        out.append({**rec, "heading": heading, "text": text})
    for i, rec in enumerate(out):
        rec["id"] = f'{rec["slug"]}#{rec["anchor"]}' if rec["anchor"] else rec["slug"]
        rec["n"] = i
    return out


def build_index(lessons: list[dict]) -> list[dict]:
    records: list[dict] = []
    for lesson in lessons:
        records += lesson_records(lesson)
    return records
