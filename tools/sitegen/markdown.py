"""A small, purpose-built Markdown dialect renderer.

Supported syntax
  # / ## / ### / ####       headings (h2/h3 feed the "On this page" table of contents)
  - item / 1. item           lists (indent 2+ spaces to nest)
  | a | b |                  tables (2nd row = |---|---|)
  > text                     quote / formal definition
  ```cpp / ```python / ...   code blocks (cpp and python are syntax highlighted; others render plain)
  :::type Title ... :::      boxes: def, analogy, mistakes, exam, key, note, tip, warn,
                             example, summary, q, ans, raw
  <<SVG:name>>               inline ``content/diagrams/name.svg``
  `code` **bold** *italic* [link](url)
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

from .config import DIAGRAMS_DIR

# --------------------------------------------------------------------------- inline


def math_span(tex: str, display: bool = False) -> str:
    """A TeX span; the web app renders it with KaTeX (the escaped source is the fallback)."""
    esc = html.escape(tex.strip(), quote=True)
    if display:
        return f'<div class="math display" data-tex="{esc}">{esc}</div>'
    return f'<span class="math" data-tex="{esc}">{esc}</span>'


def inline(text: str) -> str:
    """Render inline formatting. Code spans are stashed first so their contents stay literal."""
    codes: list[str] = []

    def stash(m: re.Match) -> str:
        codes.append("<code>" + html.escape(m.group(1), quote=False) + "</code>")
        return f"\x00{len(codes) - 1}\x00"

    def stash_math(m: re.Match) -> str:
        codes.append(math_span(m.group(1) or m.group(2), display=m.group(1) is not None))
        return f"\x00{len(codes) - 1}\x00"

    s = re.sub(r"`([^`]+)`", stash, text)
    s = re.sub(r"\$\$(.+?)\$\$|(?<![\\$])\$(?=\S)([^$\n]+?)(?<=\S)\$(?!\d)", stash_math, s)
    s = html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*(?=[^\s*])([^*\n]+?)(?<=[^\s*])\*(?![\w*])", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', s)
    s = s.replace("&lt;br&gt;", "<br>")
    return re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], s)


def slugify(text: str, used: set[str]) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower().replace("`", "")).strip("-")[:60] or "sec"
    base, n = slug, 2
    while slug in used:
        slug = f"{base}-{n}"
        n += 1
    used.add(slug)
    return slug


# --------------------------------------------------------------------------- code blocks

CPP_KEYWORDS = set(
    """alignas alignof and asm auto bool break case catch char class const constexpr
const_cast continue decltype default delete do double dynamic_cast else enum explicit export
extern false float for friend goto if inline int long mutable namespace new noexcept not nullptr
operator or private protected public register reinterpret_cast return short signed sizeof static
static_assert static_cast struct switch template this throw true try typedef typeid typename union
unsigned using virtual void volatile wchar_t while override final""".split()
)
CPP_LIBRARY = set(
    """cout cin endl cerr clog string vector std main getline setw setprecision fixed
setfill left right ifstream ofstream fstream ios showpoint boolalpha noboolalpha scientific hex oct dec
size_t""".split()
)
_TOKEN = re.compile(
    r"(?P<com>//[^\n]*|/\*.*?\*/)|(?P<str>\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*')"
    r"|(?P<pre>^[ \t]*#[^\n]*)|(?P<num>\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?[fFlLuU]*\b)"
    r"|(?P<id>\b[A-Za-z_]\w*\b)",
    re.S | re.M,
)
PY_KEYWORDS = set(
    """False None True and as assert async await break class continue def del elif else except finally
for from global if import in is lambda nonlocal not or pass raise return try while with yield""".split()
)
PY_BUILTINS = set(
    """print input len range int float str bool list dict set tuple sum min max abs round sorted
enumerate zip map filter open type isinstance self""".split()
)
_PY_TOKEN = re.compile(
    r"(?P<com>#[^\n]*)"
    r"|(?P<str>[rbfRBF]{0,2}(?:\"\"\"[\s\S]*?\"\"\"|'''[\s\S]*?'''|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'))"
    r"|(?P<num>\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)"
    r"|(?P<id>\b[A-Za-z_]\w*\b)"
)
_CODE_LABELS = {"cpp": "C++", "python": "Python", "output": "Output", "text": "", "uml": "Diagram"}


def highlight_cpp(code: str) -> str:
    out, last = [], 0
    for m in _TOKEN.finditer(code):
        out.append(html.escape(code[last:m.start()], quote=False))
        token = html.escape(m.group(0), quote=False)
        kind = m.lastgroup
        if kind == "id":
            if m.group(0) in CPP_KEYWORDS:
                out.append(f'<span class="kw">{token}</span>')
            elif m.group(0) in CPP_LIBRARY:
                out.append(f'<span class="lib">{token}</span>')
            else:
                out.append(token)
        else:
            out.append(f'<span class="{kind}">{token}</span>')
        last = m.end()
    out.append(html.escape(code[last:], quote=False))
    return "".join(out)


def highlight_python(code: str) -> str:
    out, last = [], 0
    for m in _PY_TOKEN.finditer(code):
        out.append(html.escape(code[last:m.start()], quote=False))
        token = html.escape(m.group(0), quote=False)
        kind = m.lastgroup
        if kind == "id":
            if m.group(0) in PY_KEYWORDS:
                out.append(f'<span class="kw">{token}</span>')
            elif m.group(0) in PY_BUILTINS:
                out.append(f'<span class="lib">{token}</span>')
            else:
                out.append(token)
        else:
            out.append(f'<span class="{kind}">{token}</span>')
        last = m.end()
    out.append(html.escape(code[last:], quote=False))
    return "".join(out)


HIGHLIGHTERS = {"cpp": highlight_cpp, "python": highlight_python}  # add a language here


def render_code(lang: str, lines: list[str]) -> str:
    prefix = {"cpp": "//@", "python": "#@"}.get(lang)
    if prefix:  # hide checker directives such as //@input: or #@input: from the rendered page
        lines = [l for l in lines if not l.strip().startswith(prefix)]
    code = "\n".join(lines)
    highlight = HIGHLIGHTERS.get(lang)
    body = highlight(code) if highlight else html.escape(code, quote=False)
    label = _CODE_LABELS.get(lang, lang)
    label_html = f'<div class="codelabel">{label}</div>' if label else ""
    return f'<div class="codewrap {lang or "text"}">{label_html}<pre><code>{body}</code></pre></div>'


# --------------------------------------------------------------------------- tables / lists


def _split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    cells, current, in_tick, i = [], "", False, 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line) and line[i + 1] == "|":
            current += "|"
            i += 2
            continue
        if c == "`" or (c == "$" and not in_tick and (i == 0 or line[i - 1] != "\\")):
            in_tick = not in_tick
        if c == "|" and not in_tick:
            cells.append(current.strip())
            current = ""
        else:
            current += c
        i += 1
    cells.append(current.strip())
    return cells


def render_table(rows: list[str]) -> str:
    head = _split_row(rows[0])
    has_separator = len(rows) > 1 and re.match(r"^\s*\|?\s*:?-", rows[1])
    body = [_split_row(r) for r in (rows[2:] if has_separator else rows[1:])]
    out = '<div class="tablewrap"><table><thead><tr>' + "".join(f"<th>{inline(c)}</th>" for c in head)
    out += "</tr></thead><tbody>"
    for row in body:
        out += "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>"
    return out + "</tbody></table></div>"


_LIST_ITEM = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")


def render_list(items: list[list]) -> str:
    def rec(idx: int, indent: int) -> tuple[str, int]:
        tag = "ol" if items[idx][1] else "ul"
        out = [f"<{tag}>"]
        while idx < len(items) and items[idx][0] >= indent:
            item_indent, _ordered, text = items[idx]
            if item_indent > indent:
                sub, idx = rec(idx, item_indent)
                out[-1] = out[-1][:-5] + sub + "</li>"
                continue
            out.append("<li>" + inline(text) + "</li>")
            idx += 1
        out.append(f"</{tag}>")
        return "".join(out), idx

    return rec(0, items[0][0])[0]


# --------------------------------------------------------------------------- boxes

BOXES = {
    "def": ("def", "Exam-Ready Definition"),
    "analogy": ("analogy", "Real-World Analogy"),
    "mistakes": ("mistakes", "Common Mistakes Students Make"),
    "exam": ("exam", "How This Is Asked in Exams"),
    "key": ("key", "Key Takeaways"),
    "note": ("note", "Note"),
    "tip": ("tip", "Tip"),
    "warn": ("warn", "Important"),
    "example": ("example", "Example"),
    "summary": ("summary", "Quick Revision"),
    "lecture": ("lecture", "Lecture"),
    "slides": ("slides", "Slides"),
    "compiler": ("compiler", "Compiler's Note"),
    "scribe": ("scribe", "Worked Example"),
    "code": ("codebox", "Notebook"),
}
CALLOUT = re.compile(r"^\[!(lecture|slides|compiler|scribe|code)\][ \t]*(.*)$")


@dataclass
class RenderContext:
    """Per-page state: used heading ids and the table-of-contents entries."""

    used_ids: set[str] = field(default_factory=set)
    toc: list[tuple[int, str, str]] = field(default_factory=list)


def _render_container(kind: str, title: str, inner: list[str], ctx: RenderContext) -> str:
    if kind == "raw":
        return "\n".join(inner)
    body = render_blocks(inner, ctx)
    if kind == "q":
        return f'<div class="q"><div class="qhead">{inline(title)}</div>{body}</div>'
    if kind == "ans":
        label = inline(title) if title else "Show Answer"
        return f'<details class="ans"><summary>{label}</summary><div class="ansbody">{body}</div></details>'
    css_class, default_title = BOXES.get(kind, ("note", kind))
    heading = inline(title) if title else default_title
    return f'<div class="box {css_class}"><div class="boxtitle">{heading}</div>{body}</div>'


def _svg_include(name: str) -> str:
    path = DIAGRAMS_DIR / f"{name}.svg"
    if not path.exists():
        raise FileNotFoundError(f"missing diagram: {path}")
    return f'<div class="svgwrap">{path.read_text(encoding="utf-8")}</div>'


# --------------------------------------------------------------------------- blocks


def render_blocks(lines: list[str], ctx: RenderContext) -> str:
    out: list[str] = []
    para: list[str] = []
    i, n = 0, len(lines)

    def flush() -> None:
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()

    while i < n:
        line = lines[i]
        s = line.strip()

        if s.startswith("```"):
            flush()
            lang, j, buf = s[3:].strip(), i + 1, []
            while j < n and lines[j].strip() != "```":
                buf.append(lines[j])
                j += 1
            out.append(render_code(lang, buf))
            i = j + 1
            continue

        if s.startswith(":::") and len(s) > 3:
            flush()
            kind, _, title = s[3:].strip().partition(" ")
            depth, j, in_fence = 1, i + 1, False
            while j < n:
                t = lines[j].strip()
                if t.startswith("```"):
                    in_fence = not in_fence
                elif not in_fence:
                    if t == ":::":
                        depth -= 1
                        if depth == 0:
                            break
                    elif t.startswith(":::"):
                        depth += 1
                j += 1
            if j >= n:
                raise ValueError(f"unclosed container near: {line[:70]!r}")
            out.append(_render_container(kind, title.strip(), lines[i + 1:j], ctx))
            i = j + 1
            continue

        m = re.match(r"^<<SVG:(\w+)>>$", s)
        if m:
            flush()
            out.append(_svg_include(m.group(1)))
            i += 1
            continue

        if s == "":
            flush()
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            flush()
            level, text = len(m.group(1)), m.group(2)
            heading_id = slugify(text, ctx.used_ids)
            if level in (2, 3):
                ctx.toc.append((level, heading_id, text))
            out.append(f'<h{level} id="{heading_id}">{inline(text)}</h{level}>')
            i += 1
            continue

        if s.startswith("$$"):
            flush()
            buf, j = [s], i
            while not (len(" ".join(buf)) >= 4 and " ".join(buf).endswith("$$")):
                j += 1
                if j >= n:
                    raise ValueError(f"unclosed $$ near: {s[:70]!r}")
                buf.append(lines[j].strip())
            tex = " ".join(buf).strip()[2:-2]
            out.append(math_span(tex, display=True))
            i = j + 1
            continue

        if s == "---":
            flush()
            out.append("<hr>")
            i += 1
            continue

        if s.startswith(">"):
            flush()
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                raw = lines[i].strip()[1:]
                buf.append(raw[1:] if raw.startswith(" ") else raw)
                i += 1
            callout = CALLOUT.match(buf[0]) if buf else None
            if callout:
                out.append(_render_container(callout.group(1), callout.group(2).strip(), buf[1:], ctx))
            else:
                out.append("<blockquote>" + inline(" ".join(buf)) + "</blockquote>")
            continue

        if s.startswith("|"):
            flush()
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            out.append(render_table(rows))
            continue

        if _LIST_ITEM.match(line):
            flush()
            items: list[list] = []
            while i < n:
                mm = _LIST_ITEM.match(lines[i])
                if mm:
                    items.append([len(mm.group(1).expandtabs()), mm.group(2)[0].isdigit(), mm.group(3)])
                    i += 1
                elif lines[i].strip() and lines[i].startswith("  ") and items:
                    items[-1][2] += " " + lines[i].strip()  # wrapped continuation line
                    i += 1
                else:
                    break
            out.append(render_list(items))
            continue

        para.append(s)
        i += 1

    flush()
    return "\n".join(out)


def render_markdown(text: str) -> tuple[str, RenderContext]:
    """Render a whole document; returns the HTML body and the context (holding the TOC)."""
    ctx = RenderContext()
    return render_blocks(text.splitlines(), ctx), ctx
