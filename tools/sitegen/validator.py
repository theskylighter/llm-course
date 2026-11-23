"""Run every code example in the lessons and compare it with the documented output.

Why: lesson code is read by students who will copy it. An example that doesn't compile, or whose
shown output is wrong, quietly destroys trust, so every runnable example is executed for real.

Supported fence languages (see LANGUAGES below; adding another is ~10 lines):
  ```cpp     compiled with g++ -std=c++17   (a block counts as a program if it contains ``int main``)
  ```python  run with the current interpreter in isolated mode (-I)

If the *next* fenced block is ```output, the program's stdout must match it exactly
(trailing whitespace ignored). A program with no ```output block is only checked for success.

Directives are comments inside the fence, hidden on the rendered page:
  //@input: text         (python: #@input: text)    one line fed to stdin; repeat for more lines
  //@expect: compile-error   (python: #@expect: error)  the program must FAIL
  //@norun               (python: #@norun)          compile / syntax-check only (e.g. infinite loops)
  //@nocheck             (python: #@nocheck)        skip this block entirely
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import config

# Where to look for g++ when it is not on PATH. Override with the SITEGEN_GXX environment variable.
_FALLBACK_COMPILERS = ["C:/msys64/ucrt64/bin/g++.exe", "C:/msys64/mingw64/bin/g++.exe"]


@dataclass(frozen=True)
class Language:
    comment: str                 # directive prefix, e.g. "//@"
    is_program: "re.Pattern"     # does this block count as a runnable program?
    expect_failure: str          # directive value meaning "must fail"
    suffix: str


LANGUAGES = {
    "cpp": Language("//@", re.compile(r"\bint\s+main\s*\("), "compile-error", ".cpp"),
    "python": Language("#@", re.compile(r"\S"), "error", ".py"),
}


@dataclass
class Program:
    lesson: str
    line: int
    lang: str
    code: str
    expected: str | None
    inputs: list[str]
    expect_failure: bool
    no_run: bool


@dataclass
class Result:
    program: Program
    status: str  # ok | fail | nodoc
    message: str = ""


def find_compiler() -> str:
    for candidate in (os.environ.get("SITEGEN_GXX"), shutil.which("g++"), *_FALLBACK_COMPILERS):
        if not candidate:
            continue
        if Path(candidate).exists():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    raise RuntimeError("g++ not found: install it, or set SITEGEN_GXX to its full path")


def extract_programs(lesson: str, text: str) -> list[Program]:
    lines = text.splitlines()
    blocks: list[tuple[int, str, list[str]]] = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("```"):
            lang, j, buf = s[3:].strip(), i + 1, []
            while j < len(lines) and lines[j].strip() != "```":
                buf.append(lines[j])
                j += 1
            blocks.append((i + 1, lang, buf))
            i = j + 1
        else:
            i += 1

    programs = []
    for k, (line_no, lang, buf) in enumerate(blocks):
        spec = LANGUAGES.get(lang)
        if spec is None:
            continue
        code = "\n".join(buf)
        directives = [l.strip() for l in buf if l.strip().startswith(spec.comment)]
        if f"{spec.comment}nocheck" in directives or not spec.is_program.search(code):
            continue
        expected = "\n".join(blocks[k + 1][2]) if k + 1 < len(blocks) and blocks[k + 1][1] == "output" else None
        programs.append(Program(
            lesson=lesson, line=line_no, lang=lang, code=code, expected=expected,
            inputs=[d[len(f"{spec.comment}input:"):].strip() for d in directives if d.startswith(f"{spec.comment}input:")],
            expect_failure=f"{spec.comment}expect: {spec.expect_failure}" in directives,
            no_run=f"{spec.comment}norun" in directives,
        ))
    return programs


def _normalize(s: str) -> str:
    return "\n".join(l.rstrip() for l in s.replace("\r\n", "\n").split("\n")).rstrip("\n")


def _build(program: Program, tmp: str, compiler: str | None, env: dict[str, str]) -> tuple[list[str] | None, str]:
    """Prepare the program. Returns (command to run it, error text); command is None on failure."""
    spec = LANGUAGES[program.lang]
    source = os.path.join(tmp, "p" + spec.suffix)
    Path(source).write_text(program.code, encoding="utf-8")
    if program.lang == "cpp":
        exe = os.path.join(tmp, "p.exe")
        built = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-static", "-o", exe, source],
                               capture_output=True, text=True, env=env)
        return ([exe], "") if built.returncode == 0 else (None, built.stderr[:1500])
    checked = subprocess.run([sys.executable, "-I", "-m", "py_compile", source], capture_output=True, text=True, env=env)
    return ([sys.executable, "-I", source], "") if checked.returncode == 0 else (None, checked.stderr[:1500])


def _run_one(program: Program, compiler: str | None, env: dict[str, str], cache: dict[str, str]) -> tuple[Result, str | None]:
    key = hashlib.sha1(json.dumps([program.lang, program.code, program.expected, program.inputs,
                                   program.expect_failure, program.no_run]).encode()).hexdigest()
    if cache.get(key) == "ok":
        return Result(program, "ok"), key
    with tempfile.TemporaryDirectory() as tmp:
        command, error = _build(program, tmp, compiler, env)
        if program.expect_failure:
            if command is None:
                return Result(program, "ok"), key
            if program.no_run:
                return Result(program, "fail", "expected a build error but it built"), None
        if command is None:
            return Result(program, "fail", "BUILD ERROR:\n" + error), None
        if program.no_run:
            return Result(program, "ok"), key
        try:
            run = subprocess.run(command, input="\n".join(program.inputs) + "\n", capture_output=True,
                                 text=True, timeout=10, cwd=tmp, env=env)
        except subprocess.TimeoutExpired:
            return Result(program, "fail", "TIMEOUT (mark intentional infinite loops with the norun directive)"), None
        if program.expect_failure:
            return (Result(program, "ok"), key) if run.returncode != 0 else (Result(program, "fail", "expected failure but exited 0"), None)
        if run.returncode != 0:
            return Result(program, "fail", f"EXIT CODE {run.returncode}\n{run.stderr[:800]}"), None
        if program.expected is None:
            return Result(program, "nodoc", _normalize(run.stdout)[:300]), key
        if _normalize(run.stdout) != _normalize(program.expected):
            return Result(program, "fail",
                          f"OUTPUT MISMATCH\n--- expected ---\n{_normalize(program.expected)}"
                          f"\n--- actual ---\n{_normalize(run.stdout)}"), None
        return Result(program, "ok"), key


def check(only: str | None = None, workers: int = 8) -> int:
    """Run the validator. Returns the number of failing programs."""
    course = config.load_course()
    programs: list[Program] = []
    for lesson in course.lessons:
        if only and only not in lesson.slug:
            continue
        if lesson.source.exists():
            programs += extract_programs(lesson.slug, lesson.source.read_text(encoding="utf-8"))

    if not programs:
        print("No runnable code examples found.")
        return 0

    compiler = None
    env = dict(os.environ)
    if any(p.lang == "cpp" for p in programs):
        compiler = find_compiler()
        # put the compiler's own folder first: on Windows, other toolchains on PATH can shadow its DLLs
        env["PATH"] = str(Path(compiler).parent) + os.pathsep + env.get("PATH", "")

    config.CACHE_DIR.mkdir(exist_ok=True)
    cache_file = config.CACHE_DIR / "validated.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}

    print(f"Checking {len(programs)} programs ...")
    failures = undocumented = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result, key in pool.map(lambda p: _run_one(p, compiler, env, cache), programs):
            where = f"{result.program.lesson}.md:{result.program.line}"
            if result.status == "fail":
                failures += 1
                print(f"\n[FAIL] {where} ({result.program.lang})\n{result.message}")
            else:
                if key:
                    cache[key] = "ok"
                if result.status == "nodoc":
                    undocumented += 1
                    print(f"[no output block] {where} -> prints: {result.message!r}")
    cache_file.write_text(json.dumps(cache))
    print(f"\nDone: {len(programs)} programs, {failures} failures, {undocumented} without documented output.")
    return failures
