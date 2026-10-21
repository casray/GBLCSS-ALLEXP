#!/usr/bin/env python3
"""Render the GB-LCSS Markdown manuscript as a readable standalone HTML file."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


def inline_markup(value: str) -> str:
    placeholders: dict[str, str] = {}

    def hold(rendered: str) -> str:
        key = f"@@HTML{len(placeholders)}@@"
        placeholders[key] = rendered
        return key

    value = re.sub(
        r"!\[([^\]]*)\]\(([^)]+)\)",
        lambda match: hold(
            '<img src="{}" alt="{}" loading="lazy">'.format(
                html.escape(match.group(2), quote=True),
                html.escape(match.group(1), quote=True),
            )
        ),
        value,
    )
    value = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda match: hold(
            '<a href="{}">{}</a>'.format(
                html.escape(match.group(2), quote=True),
                html.escape(match.group(1)),
            )
        ),
        value,
    )
    value = re.sub(
        r"`([^`]+)`",
        lambda match: hold(f"<code>{html.escape(match.group(1))}</code>"),
        value,
    )
    value = re.sub(
        r"(?<!\$)\$([^\n$]+?)\$(?!\$)",
        lambda match: hold(f'<span class="inline-math">\\({html.escape(match.group(1))}\\)</span>'),
        value,
    )
    value = html.escape(value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", value)
    for key, rendered in placeholders.items():
        value = value.replace(key, rendered)
    return value


def slugify(title: str, used: set[str]) -> str:
    base = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", title).strip("-").lower()
    base = base or "section"
    slug = base
    index = 2
    while slug in used:
        slug = f"{base}-{index}"
        index += 1
    used.add(slug)
    return slug


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def render_markdown(source: str) -> tuple[str, list[tuple[int, str, str]]]:
    lines = source.splitlines()
    blocks: list[str] = []
    headings: list[tuple[int, str, str]] = []
    used_ids: set[str] = set()
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue

        if stripped.startswith("```"):
            language = stripped[3:].strip()
            code_lines: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            index += 1
            blocks.append(
                '<pre class="algorithm"><code class="language-{}">{}</code></pre>'.format(
                    html.escape(language, quote=True), html.escape("\n".join(code_lines))
                )
            )
            continue

        if stripped == "$$":
            formula: list[str] = []
            index += 1
            while index < len(lines) and lines[index].strip() != "$$":
                formula.append(lines[index])
                index += 1
            index += 1
            blocks.append(
                '<div class="display-math">\\[{}\\]</div>'.format(
                    html.escape("\n".join(formula))
                )
            )
            continue

        heading_match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            anchor = slugify(re.sub(r"[*`]", "", title), used_ids)
            headings.append((level, re.sub(r"[*`]", "", title), anchor))
            blocks.append(f'<h{level} id="{anchor}">{inline_markup(title)}</h{level}>')
            index += 1
            continue

        if stripped == "---":
            blocks.append("<hr>")
            index += 1
            continue

        if stripped.startswith("|") and index + 1 < len(lines):
            table_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            rows = [split_table_row(row) for row in table_lines]
            if len(rows) >= 2 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in rows[1]):
                header = rows[0]
                body = rows[2:]
                rendered = ['<div class="table-wrap"><table><thead><tr>']
                rendered.extend(f"<th>{inline_markup(cell)}</th>" for cell in header)
                rendered.append("</tr></thead><tbody>")
                for row in body:
                    rendered.append("<tr>")
                    rendered.extend(f"<td>{inline_markup(cell)}</td>" for cell in row)
                    rendered.append("</tr>")
                rendered.append("</tbody></table></div>")
                blocks.append("".join(rendered))
            else:
                blocks.append(f"<pre>{html.escape(chr(10).join(table_lines))}</pre>")
            continue

        unordered = re.match(r"^-\s+(.+)$", stripped)
        ordered = re.match(r"^\d+\.\s+(.+)$", stripped)
        if unordered or ordered:
            tag = "ul" if unordered else "ol"
            items: list[str] = []
            pattern = r"^-\s+(.+)$" if unordered else r"^\d+\.\s+(.+)$"
            while index < len(lines):
                match = re.match(pattern, lines[index].strip())
                if not match:
                    break
                items.append(f"<li>{inline_markup(match.group(1))}</li>")
                index += 1
            blocks.append(f"<{tag}>{''.join(items)}</{tag}>")
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate:
                break
            if (
                candidate == "$$"
                or candidate == "---"
                or candidate.startswith("```")
                or candidate.startswith("|")
                or re.match(r"^#{1,4}\s+", candidate)
                or re.match(r"^-\s+", candidate)
                or re.match(r"^\d+\.\s+", candidate)
            ):
                break
            paragraph_lines.append(candidate)
            index += 1
        paragraph = " ".join(paragraph_lines)
        css_class = ""
        if re.match(r"^\*\*(图|表|算法)\s*\d+", paragraph):
            css_class = ' class="caption"'
        elif paragraph.startswith("**关键词**"):
            css_class = ' class="keywords"'
        blocks.append(f"<p{css_class}>{inline_markup(paragraph)}</p>")

    return "\n".join(blocks), headings


def build_toc(headings: list[tuple[int, str, str]]) -> str:
    items = []
    for level, title, anchor in headings:
        if level == 1 and title != "补充材料":
            continue
        if level > 3:
            continue
        items.append(
            f'<a class="toc-level-{level}" href="#{anchor}">{html.escape(title)}</a>'
        )
    return "\n".join(items)


def build_html(title: str, body: str, toc: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [["\\\\(", "\\\\)"]],
        displayMath: [["\\\\[", "\\\\]"]],
        tags: "ams"
      }},
      chtml: {{ scale: 0.96 }}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <style>
    :root {{ color-scheme: light; --paper: #fff; --ink: #171717; --muted: #68707a; --line: #cfd4da; --accent: #1f5d50; }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{ margin: 0; color: var(--ink); background: #eef1f3; font-family: "Times New Roman", "Noto Serif SC", "Songti SC", SimSun, serif; font-size: 17px; line-height: 1.78; }}
    .layout {{ display: grid; grid-template-columns: 270px minmax(0, 1160px); gap: 26px; width: min(1500px, calc(100% - 36px)); margin: 22px auto 70px; align-items: start; }}
    nav {{ position: sticky; top: 18px; max-height: calc(100vh - 36px); overflow: auto; padding: 18px 15px; background: var(--paper); border: 1px solid #dfe3e6; box-shadow: 0 2px 12px rgba(0,0,0,.05); }}
    nav strong {{ display: block; margin-bottom: 10px; font-family: Arial, "Microsoft YaHei", sans-serif; font-size: 14px; }}
    nav a {{ display: block; padding: 3px 4px; color: #3c454d; text-decoration: none; font-family: Arial, "Microsoft YaHei", sans-serif; font-size: 12px; line-height: 1.45; border-left: 2px solid transparent; }}
    nav a:hover {{ color: var(--accent); border-left-color: var(--accent); }}
    .toc-level-3 {{ padding-left: 15px; color: #66717a; }}
    main {{ min-width: 0; padding: 58px 72px 82px; background: var(--paper); box-shadow: 0 2px 18px rgba(0,0,0,.08); }}
    h1, h2, h3, h4 {{ color: #111; line-height: 1.32; scroll-margin-top: 18px; }}
    h1 {{ margin: 0 0 24px; text-align: center; font-size: 30px; }}
    h1:not(:first-child) {{ margin-top: 54px; padding-top: 18px; border-top: 2px solid #222; }}
    h2 {{ margin-top: 43px; padding-bottom: 7px; border-bottom: 1px solid #aeb5bb; font-size: 25px; }}
    h3 {{ margin-top: 31px; font-size: 21px; }}
    h4 {{ margin-top: 25px; font-size: 18px; }}
    p {{ margin: .78em 0; text-align: justify; }}
    li {{ margin: .3em 0; }}
    hr {{ margin: 34px 0; border: 0; border-top: 1px solid #bcc3c9; }}
    img {{ display: block; max-width: 100%; height: auto; margin: 24px auto 13px; }}
    p:has(> img) {{ margin: 0; text-align: center; }}
    .caption {{ margin: 8px auto 25px; color: #30363b; font-size: 15px; line-height: 1.58; text-align: left; }}
    .keywords {{ padding: 10px 13px; background: #f5f7f7; border-left: 3px solid var(--accent); }}
    .table-wrap {{ width: 100%; margin: 20px 0 27px; overflow-x: auto; }}
    table {{ width: 100%; min-width: max-content; border-collapse: collapse; font-size: 13.5px; line-height: 1.45; }}
    th, td {{ padding: 7px 9px; border: 1px solid #aeb4ba; vertical-align: middle; }}
    th {{ background: #edf1f2; font-weight: 700; text-align: center; }}
    td:first-child {{ text-align: left; }}
    code {{ padding: .08em .3em; border-radius: 3px; background: #eef1f3; font-family: Consolas, "Courier New", monospace; font-size: .9em; }}
    pre.algorithm {{ margin: 16px 0 26px; padding: 18px 21px; overflow-x: auto; border: 1px solid #aeb6bc; border-left: 4px solid var(--accent); background: #f7f9f9; line-height: 1.55; }}
    pre.algorithm code {{ padding: 0; background: transparent; font-size: 14px; }}
    .display-math {{ margin: 1.15em 0; padding: .2em 0; overflow-x: auto; text-align: center; }}
    .inline-math {{ white-space: nowrap; }}
    mjx-container[jax="CHTML"][display="true"] {{ margin: .8em 0 !important; }}
    @media (max-width: 1040px) {{ .layout {{ display: block; width: min(100%, 1160px); margin-top: 0; }} nav {{ display: none; }} main {{ padding: 42px 38px 65px; }} }}
    @media (max-width: 680px) {{ body {{ font-size: 16px; }} main {{ padding: 30px 18px 52px; box-shadow: none; }} h1 {{ font-size: 25px; }} h2 {{ font-size: 22px; }} }}
    @media print {{ body {{ background: #fff; font-size: 10.5pt; }} .layout {{ display: block; width: auto; margin: 0; }} nav {{ display: none; }} main {{ width: auto; margin: 0; padding: 0; box-shadow: none; }} .table-wrap {{ overflow: visible; }} table {{ min-width: 0; font-size: 7.8pt; }} h1, h2, h3 {{ break-after: avoid; }} img, table, pre {{ break-inside: avoid; }} }}
  </style>
</head>
<body>
  <div class="layout">
    <nav><strong>文章目录</strong>{toc}</nav>
    <main>{body}</main>
  </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.read_text(encoding="utf-8")
    title_match = re.search(r"^#\s+(.+)$", source, re.MULTILINE)
    title = title_match.group(1) if title_match else args.source.stem
    body, headings = render_markdown(source)
    rendered = build_html(title, body, build_toc(headings))
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Rendered {args.output} ({len(headings)} headings)")


if __name__ == "__main__":
    main()
