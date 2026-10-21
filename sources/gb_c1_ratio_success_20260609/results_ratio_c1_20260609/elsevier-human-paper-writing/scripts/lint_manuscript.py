#!/usr/bin/env python3
"""Flag common AI-style patterns in academic manuscripts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


RULES = (
    (
        "defensive-negation",
        re.compile(
            r"并不声称|不声称|不主张|并不主张|并不意味着|不意味着|"
            r"目的不是|不是为了|不是.{0,24}而是|不等同于|不构成"
        ),
        "把范围条件写进正面陈述，并将必要局限集中放在协议或局限段。",
    ),
    (
        "internal-experiment-id",
        re.compile(
            r"\bMLP-\d+\b|\bPhase\s*\d+[A-Za-z-]*\b|"
            r"\bseed\s*0\s*smoke\b|\bseed0\s*smoke\b|"
            r"\bruns[/\\][A-Za-z0-9_.\-/\\]+"
        ),
        "删除内部实验编号、阶段名和运行目录，改成可公开理解的方法或实验名称。",
    ),
    (
        "reviewer-dialogue",
        re.compile(r"审稿人可能|为了回应审稿|为回应审稿|避免审稿人|审稿意见"),
        "论文正文不应写成审稿回复。直接陈述研究设计、证据与边界。",
    ),
    (
        "revision-meta",
        re.compile(r"本轮新增|本轮实验|早期版本|此前版本|修正了之前|历史代理"),
        "删除版本演化和内部研究过程，保留最终方法与可验证事实。",
    ),
)

TEMPLATE_PHRASES = (
    "值得注意的是",
    "需要指出的是",
    "需要强调的是",
    "综上所述",
)


def _excerpt(text: str, start: int, end: int, radius: int = 28) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return text[left:right].replace("\n", " ")


def lint_text(text: str) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    for code, pattern, message in RULES:
        for match in pattern.finditer(text):
            issues.append(
                {
                    "code": code,
                    "message": message,
                    "start": match.start(),
                    "excerpt": _excerpt(text, match.start(), match.end()),
                }
            )

    for phrase in TEMPLATE_PHRASES:
        matches = list(re.finditer(re.escape(phrase), text))
        if len(matches) >= 3:
            issues.append(
                {
                    "code": "repeated-template-transition",
                    "message": f"“{phrase}”出现 {len(matches)} 次。保留必要位置，其余改为自然的逻辑衔接。",
                    "start": matches[2].start(),
                    "excerpt": _excerpt(
                        text, matches[2].start(), matches[2].end()
                    ),
                }
            )

    for paragraph in re.split(r"\n\s*\n", text):
        compact = re.sub(r"\s+", "", paragraph)
        if (
            len(compact) > 700
            and not compact.startswith("|")
            and not paragraph.lstrip().startswith("- ")
            and not compact.startswith("```")
            and not compact.startswith("$$")
        ):
            start = text.find(paragraph)
            issues.append(
                {
                    "code": "overlong-paragraph",
                    "message": "段落超过 700 个非空白字符。检查是否承担了多个论证任务。",
                    "start": start,
                    "excerpt": compact[:80],
                }
            )

    return sorted(issues, key=lambda item: int(item["start"]))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lint an academic manuscript for common defensive AI-writing patterns."
    )
    parser.add_argument("manuscript", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    text = args.manuscript.read_text(encoding="utf-8")
    issues = lint_text(text)
    if args.as_json:
        print(json.dumps(issues, ensure_ascii=False, indent=2))
    elif not issues:
        print("No configured AI-writing patterns found.")
    else:
        for issue in issues:
            print(
                f"[{issue['code']}] {issue['message']}\n"
                f"  {issue['excerpt']}"
            )
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
