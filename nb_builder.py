#!/usr/bin/env python3
"""
Notebook builder — converts _build_*.md content scripts into .ipynb files.

Usage:
    python3 nb_builder.py course/weekNN_TOPIC/_build_lecture.md
    python3 nb_builder.py course/weekNN_TOPIC/_build_seminar.md
    python3 nb_builder.py course/weekNN_TOPIC/_build_hw.md

The content script is a markdown file with ~~~python fenced code blocks.
Everything outside fences becomes markdown cells; everything inside
becomes code cells. This script parses the fences and writes the .ipynb.
"""
import json
import pathlib
import sys


def _is_md_break(line):
    """Return True if this line should start a new markdown cell.

    A markdown break occurs at:
    - Section headings (## or deeper)
    - Horizontal rules (--- on its own)
    This mirrors how Jupyter notebooks typically split prose into
    separate cells at section boundaries.
    """
    stripped = line.strip()
    if stripped.startswith("## ") or stripped.startswith("### "):
        return True
    if stripped == "---":
        return True
    return False


def parse_md(text):
    """Split markdown+fenced-code text into notebook cells.

    Markdown regions are further split at section headings (##, ###)
    and horizontal rules (---) so that each section becomes its own
    markdown cell.  This enables lecture notebooks to meet the 60 %
    prose-ratio target without artificial code-cell padding.
    """
    cells = []
    lines = text.split("\n")
    i = 0
    md_buf = []

    def flush_md():
        content = "\n".join(md_buf).strip()
        if content:
            cells.append({"cell_type": "markdown", "metadata": {}, "source": content})
        md_buf.clear()

    while i < len(lines):
        if lines[i].rstrip() == "~~~python":
            flush_md()
            i += 1
            code_buf = []
            while i < len(lines) and lines[i].rstrip() != "~~~":
                code_buf.append(lines[i])
                i += 1
            # Strip trailing blank lines from code cell
            while code_buf and not code_buf[-1].strip():
                code_buf.pop()
            cells.append({
                "cell_type": "code",
                "metadata": {},
                "source": "\n".join(code_buf),
                "execution_count": None,
                "outputs": [],
            })
            i += 1  # skip closing ~~~
        else:
            # Check if this line starts a new markdown section
            if md_buf and _is_md_break(lines[i]):
                flush_md()
            md_buf.append(lines[i])
            i += 1

    flush_md()
    return cells


def quality_check(cells, out_name, nb_type):
    """Check quality and return (passed, report_lines).

    Rejects on: consecutive code cells (lecture) or excessive consecutive
    code (seminar/hw), prose ratio below minimum.
    Warns on: oversized code cells, thin prose.
    """
    n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    total = len(cells)
    pct_md = n_md / total * 100 if total else 0
    passed = True
    report = []

    report.append(f"{'✓' if True else '✗'} {out_name}: {total} cells "
                  f"({n_md} md / {n_code} code, {pct_md:.0f}% prose)")

    # --- Consecutive code cells (REJECT for lecture, REJECT if >2 for seminar/hw) ---
    consecutive = []
    for i in range(1, total):
        if cells[i]["cell_type"] == "code" and cells[i - 1]["cell_type"] == "code":
            consecutive.append(i)
    if consecutive:
        if nb_type == "lecture":
            report.append(f"  ✗ REJECT: {len(consecutive)} consecutive code cell "
                          f"pair(s) at positions: {consecutive} (lecture allows 0)")
            passed = False
        else:
            # Check for runs of >2
            runs_over_2 = []
            run_len = 1
            for i in range(1, total):
                if cells[i]["cell_type"] == "code" and cells[i - 1]["cell_type"] == "code":
                    run_len += 1
                else:
                    if run_len > 2:
                        runs_over_2.append((i - run_len, run_len))
                    run_len = 1
            if run_len > 2:
                runs_over_2.append((total - run_len, run_len))
            if runs_over_2:
                report.append(f"  ✗ REJECT: code cell runs >2 at: {runs_over_2} "
                              f"(max 2 in solution blocks)")
                passed = False
            else:
                report.append(f"  ⚠ {len(consecutive)} consecutive code cell "
                              f"pair(s) at positions: {consecutive}")

    # --- Prose ratio (REJECT if below minimum) ---
    min_ratio = {"lecture": 60, "seminar": 45, "hw": 40}.get(nb_type, 40)
    if pct_md < min_ratio:
        report.append(f"  ✗ REJECT: prose ratio {pct_md:.0f}% < {min_ratio}% minimum "
                      f"for {nb_type}")
        passed = False

    # --- Oversized code cells (WARN) ---
    for i, c in enumerate(cells):
        if c["cell_type"] == "code":
            n_lines = c["source"].count("\n") + 1
            if n_lines > 25:
                report.append(f"  ⚠ Code cell {i} has {n_lines} lines "
                              f"(recommended ≤25, ok for visualizations)")

    # --- Thin markdown cells between code cells (WARN) ---
    for i in range(1, total - 1):
        if (cells[i]["cell_type"] == "markdown"
                and cells[i - 1]["cell_type"] == "code"
                and cells[i + 1]["cell_type"] == "code"):
            words = len(cells[i]["source"].split())
            if words < 15:
                report.append(f"  ⚠ Markdown cell {i} between code cells has "
                              f"only {words} words (min ~15)")

    # Update first line with pass/fail
    status = "✓" if passed else "✗"
    report[0] = report[0].replace("✓", status, 1)

    return passed, report


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 nb_builder.py <_build_*.md>", file=sys.stderr)
        sys.exit(1)

    script = pathlib.Path(sys.argv[1])
    if not script.exists():
        print(f"Error: {script} not found", file=sys.stderr)
        sys.exit(1)

    cells = parse_md(script.read_text())

    # Derive output path: _build_lecture.md → lecture.ipynb
    name = script.stem.replace("_build_", "") + ".ipynb"
    out = script.parent / name

    # Detect notebook type from filename
    nb_type = script.stem.replace("_build_", "")
    if nb_type not in ("lecture", "seminar", "hw"):
        nb_type = "hw"  # default to most permissive

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "cells": cells,
    }

    passed, report = quality_check(cells, out.name, nb_type)
    for line in report:
        print(line)

    if not passed:
        print(f"\nBuild REJECTED. Fix the issues above and rebuild.")
        sys.exit(1)

    with open(out, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\nWrote {out}")
