from __future__ import annotations

import argparse
import base64
import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_MD = ROOT / "docs" / "paper_manuscript_v1_zh_20260706.md"
OUT_HTML = ROOT / "docs" / "paper_manuscript_v1_zh_20260706_embedded.html"
ASSET_DIR = ROOT / "paper_assets_20260706"


TABLE_ASSETS = {
    "appendix_B": ASSET_DIR / "tables" / "appendix_B_paired_significance.md",
    "appendix_C": ASSET_DIR / "tables" / "appendix_C_graph_dependent_references.md",
    "appendix_D1": ASSET_DIR / "tables" / "appendix_D1_parameter_sensitivity.md",
    "appendix_D2": ASSET_DIR / "tables" / "appendix_D2_beta_audit.md",
    "appendix_F": ASSET_DIR / "tables" / "appendix_F_cut_value.md",
    "appendix_H_mono": ASSET_DIR / "tables" / "appendix_H_labelrate_monotonicity.md",
    "table_B": ASSET_DIR / "tables" / "table_B_graphfree_family.md",
    "extension_distill": ASSET_DIR / "tables" / "extension_denoised_teacher_distill.md",
}


FIGURE_ASSETS = {
    "fig1": (
        ROOT / "docs" / "figures" / "fig1_lcss_pipeline_20260716.png",
        "图 1 LCSS 从训练期图结构到仅特征推理的整体流程",
    ),
    "fig_E": (
        ASSET_DIR / "figures" / "appendix_E_feature_noise_curves.png",
        "图 E.1 测试特征噪声完整曲线",
    ),
    "fig_G": (
        ASSET_DIR / "figures" / "appendix_G_tsne_three_coloring.png",
        "图 G.1 t-SNE 三着色可视化",
    ),
    "fig_G2": (
        ASSET_DIR / "figures" / "appendix_G2_grace_ball_cover.png",
        "图 G.2 GRACE-H 空间的粒球覆盖可视化",
    ),
    "fig_G3": (
        ASSET_DIR / "figures" / "appendix_G3_noise_soft_correction.png",
        "图 G.3 含噪标签到局部共识软标签的修正机制",
    ),
    "fig_H": (
        ASSET_DIR / "figures" / "appendix_H_labelrate_curves.png",
        "图 H.1 低标签率完整曲线",
    ),
}


def inline_format(text: str) -> str:
    text = text.replace("\\*", "*")
    placeholders: list[str] = []

    def stash_code(match: re.Match[str]) -> str:
        placeholders.append(f"<code>{html.escape(match.group(1))}</code>")
        return f"\u0000CODE{len(placeholders) - 1}\u0000"

    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", stash_code, escaped)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)

    for idx, value in enumerate(placeholders):
        escaped = escaped.replace(f"\u0000CODE{idx}\u0000", value)
    return escaped


def split_table_row(row: str) -> list[str]:
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [cell.strip() for cell in row.split("|")]


def is_table_separator(line: str) -> bool:
    cells = split_table_row(line)
    if not cells:
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def table_to_html(lines: list[str], start: int) -> tuple[str, int]:
    headers = split_table_row(lines[start])
    rows: list[list[str]] = []
    idx = start + 2
    while idx < len(lines) and lines[idx].strip().startswith("|"):
        rows.append(split_table_row(lines[idx]))
        idx += 1

    thead = "".join(f"<th>{inline_format(cell)}</th>" for cell in headers)
    body_rows = []
    for row in rows:
        padded = row + [""] * (len(headers) - len(row))
        body_rows.append(
            "<tr>" + "".join(f"<td>{inline_format(cell)}</td>" for cell in padded[: len(headers)]) + "</tr>"
        )
    table = (
        '<div class="table-wrap"><table>'
        f"<thead><tr>{thead}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )
    return table, idx


def markdown_to_html(markdown_text: str, html_blocks: dict[str, str] | None = None) -> str:
    html_blocks = html_blocks or {}
    lines = markdown_text.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    idx = 0

    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()

        if not stripped:
            idx += 1
            continue

        if stripped.startswith("{{HTML:") and stripped.endswith("}}"):
            key = stripped[len("{{HTML:") : -2]
            out.append(html_blocks[key])
            idx += 1
            continue

        if re.fullmatch(r"-{3,}", stripped):
            out.append("<hr>")
            idx += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            level = len(heading.group(1))
            text = inline_format(heading.group(2))
            slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", heading.group(2)).strip("-")
            out.append(f'<h{level} id="{html.escape(slug)}">{text}</h{level}>')
            idx += 1
            continue

        if stripped.startswith(">"):
            quote_lines = []
            while idx < len(lines) and lines[idx].strip().startswith(">"):
                quote_lines.append(lines[idx].strip().lstrip(">").strip())
                idx += 1
            out.append("<blockquote>" + markdown_to_html("\n\n".join(quote_lines), html_blocks) + "</blockquote>")
            continue

        if stripped.startswith("|") and idx + 1 < len(lines) and is_table_separator(lines[idx + 1]):
            table, idx = table_to_html(lines, idx)
            out.append(table)
            continue

        if stripped.startswith("- "):
            items = []
            while idx < len(lines) and lines[idx].strip().startswith("- "):
                items.append(lines[idx].strip()[2:])
                idx += 1
            out.append("<ul>" + "".join(f"<li>{inline_format(item)}</li>" for item in items) + "</ul>")
            continue

        para_lines = [stripped]
        idx += 1
        while idx < len(lines):
            nxt = lines[idx].strip()
            if (
                not nxt
                or nxt.startswith("#")
                or nxt.startswith(">")
                or nxt.startswith("- ")
                or nxt.startswith("{{HTML:")
                or re.fullmatch(r"-{3,}", nxt)
                or (nxt.startswith("|") and idx + 1 < len(lines) and is_table_separator(lines[idx + 1]))
            ):
                break
            para_lines.append(nxt)
            idx += 1
        out.append(f"<p>{inline_format(' '.join(para_lines))}</p>")

    return "\n".join(out)


def image_html(key: str) -> str:
    path, caption = FIGURE_ASSETS[key]
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        '<figure class="paper-figure">'
        f'<img src="data:image/png;base64,{data}" alt="{html.escape(caption)}">'
        f"<figcaption>{html.escape(caption)}</figcaption>"
        "</figure>"
    )


def asset_table_html(key: str, title: str) -> str:
    text = TABLE_ASSETS[key].read_text(encoding="utf-8")
    text = sanitize_asset_markdown(key, text)
    return (
        '<section class="embedded-asset">'
        f"<h4>{html.escape(title)}</h4>"
        f"{markdown_to_html(text)}"
        "</section>"
    )


def drop_markdown_table_column(text: str, column_name: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    idx = 0
    while idx < len(lines):
        if idx + 1 < len(lines) and lines[idx].strip().startswith("|") and is_table_separator(lines[idx + 1]):
            headers = split_table_row(lines[idx])
            drop_idx = None
            for pos, header in enumerate(headers):
                if header.strip() == column_name:
                    drop_idx = pos
                    break
            if drop_idx is None:
                out.append(lines[idx])
                idx += 1
                continue

            table_lines = [lines[idx], lines[idx + 1]]
            idx += 2
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                table_lines.append(lines[idx])
                idx += 1

            for table_line in table_lines:
                cells = split_table_row(table_line)
                kept = [cell for pos, cell in enumerate(cells) if pos != drop_idx]
                out.append("| " + " | ".join(kept) + " |")
            continue

        out.append(lines[idx])
        idx += 1
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def sanitize_asset_markdown(key: str, text: str) -> str:
    if key == "appendix_C":
        text = drop_markdown_table_column(text, "source_file")
    text = re.sub(r"\n?Raw per-seed rows are preserved in `[^`]+`\.\n?", "\n", text)
    text = re.sub(r"runs[\\/][^\s|`]+", "source-run", text)
    return text


def remove_review_file_index(text: str) -> str:
    return re.sub(
        r"\n审阅用文件索引（直接复制路径打开；不使用本地 markdown 链接格式）：\n\n(?:- .+\n)+",
        "\n",
        text,
    )


def strip_local_paths(text: str) -> str:
    text = re.sub(
        r"!\[[^\]]*\]\([^\n)]*figures/fig1_lcss_pipeline_20260716\.png\)",
        "{{HTML:fig1}}",
        text,
    )
    local_figure_placeholders = {
        "appendix_E_feature_noise_curves.png": "fig_E",
        "appendix_G_tsne_three_coloring.png": "fig_G",
        "appendix_G2_grace_ball_cover.png": "fig_G2",
        "appendix_G3_noise_soft_correction.png": "fig_G3",
        "appendix_H_labelrate_curves.png": "fig_H",
    }
    for filename, key in local_figure_placeholders.items():
        text = re.sub(
            rf"!\[[^\]]*\]\([^\n)]*paper_assets_20260706/figures/{re.escape(filename)}\)",
            f"{{{{HTML:{key}}}}}",
            text,
        )
    text = re.sub(r"`C:\\Users\\Admin\\Desktop\\实验8去噪5\.19-\\external\\NoisyGL\\paper_assets_20260706[^`]*`", "已内嵌", text)
    text = text.replace("（产出物位于 已内嵌，LaTeX 版可直接 `\\input`；以下为审阅版实体内容，23 项对拍全部通过。）", "（本 HTML 已内嵌所有可审阅图表，未保留本地文件路径；23 项对拍全部通过。）")
    text = text.replace("v1.2（2026-07-06）：附录 A-H 产出物生成完毕（已内嵌，23 项对拍全 pass）。", "v1.2（2026-07-06）：附录 A-H 产出物已内嵌至 HTML 审阅版，23 项对拍全 pass。")
    text = re.sub(r"`(?:figures|tables)/[^`]+`", "已内嵌", text)
    text = re.sub(r"`paper_assets_20260706/[^`]*`", "已内嵌", text)
    text = re.sub(r"(?:\.\./)?paper_assets_20260706/(?:figures/)?", "已内嵌资产/", text)
    text = re.sub(r"见 已内嵌[；，。]?", "见下方内嵌内容。", text)
    text = re.sub(r"LaTeX 版见 已内嵌[；，。]?", "", text)
    return text


def inject_assets(text: str) -> str:
    text = re.sub(
        r"完整表（339 行）见下方内嵌内容。(?:LaTeX 版见下方内嵌内容。)?LCSS 与每个 baseline 在每个数据集 × 噪声设置下的逐种子配对差、paired t-test 与 Wilcoxon signed-rank p 值。此处给出汇总计数：",
        "完整配对显著性表已内嵌如下。LCSS 与每个 baseline 在每个数据集 × 噪声设置下的逐种子配对差、paired t-test 与 Wilcoxon signed-rank p 值。此处另给出汇总计数：\n\n{{HTML:appendix_B}}\n",
        text,
    )
    text = re.sub(
        r"TFR 的迁移配置结果与逐种子明细见下方内嵌内容。(?:LaTeX 版见下方内嵌内容。)?；?来源哈希已记录，协议元数据整理在投稿前完成。",
        "TFR 的迁移配置结果与逐种子明细已内嵌如下；来源哈希已记录，协议元数据整理在投稿前完成。\n\n{{HTML:appendix_C}}",
        text,
    )
    text = re.sub(
        r"\*\*D\.1 构造参数敏感性（27 配置网格，uniform 0\.4）。\*\* 完整网格见下方内嵌内容。(?:LaTeX 版见下方内嵌内容。)?三个结论：",
        "**D.1 构造参数敏感性（27 配置网格，uniform 0.4）。** 完整网格已内嵌如下。\n\n{{HTML:appendix_D1}}\n\n三个结论：",
        text,
    )
    text = re.sub(
        r"\*\*D\.2 β 选择审计。\*\* 完整表见下方内嵌内容。(?:LaTeX 版见下方内嵌内容。)?若按 per-setting 验证集精度选择",
        "**D.2 β 选择审计。** 完整表已内嵌如下。\n\n{{HTML:appendix_D2}}\n\n若按 per-setting 验证集精度选择",
        text,
    )
    text = re.sub(
        r"图 E\.1（PDF：已内嵌；PNG：已内嵌）：",
        "图 E.1：\n\n{{HTML:fig_E}}\n\n",
        text,
    )
    text = re.sub(
        r"完整表见(?:上文所引资产|下方内嵌内容。(?:LaTeX 版见下方内嵌内容。)?)（6 方法臂 × 5 数据集，uniform 0\.4，5-seed）。",
        "完整 Cut Value 表已内嵌如下（6 方法臂 × 5 数据集，uniform 0.4，5-seed）。\n\n{{HTML:appendix_F}}\n",
        text,
    )
    text = re.sub(
        r"图 G\.1（PDF：已内嵌；PNG：已内嵌）：",
        "图 G.1：\n\n{{HTML:fig_G}}\n\n",
        text,
    )
    text = re.sub(
        r"图 G\.2（PDF：已内嵌；PNG：已内嵌）：",
        "图 G.2：\n\n{{HTML:fig_G2}}\n\n",
        text,
    )
    text = re.sub(
        r"图 H\.1（PDF：已内嵌；PNG：已内嵌）：",
        "图 H.1：\n\n{{HTML:fig_H}}\n\n",
        text,
    )
    text = text.replace(
        "**单调性核验**（已内嵌）：",
        "**单调性核验**（完整表如下）：\n\n{{HTML:appendix_H_mono}}\n\n",
    )
    text = re.sub(
        r"\*\*表 I\.1（正文表 B 的完整版）\*\*：graph-free 家族对照全表见下方内嵌内容。(?:LaTeX 版见下方内嵌内容。)?——注意",
        "**表 I.1（正文表 B 的完整版）**：graph-free 家族对照全表已内嵌如下。\n\n{{HTML:table_B}}\n\n注意",
        text,
    )
    text = re.sub(
        r"\*\*表 I\.2（扩展节五臂全表）\*\*：Markdown 版见下方内嵌内容。(?:LaTeX 版见下方内嵌内容。)?——五个 teacher 臂",
        "**表 I.2（扩展节五臂全表）**：完整五臂表已内嵌如下。\n\n{{HTML:extension_distill}}\n\n五个 teacher 臂",
        text,
    )
    return text


def build_html(source_markdown: str | Path = SRC_MD) -> str:
    raw = Path(source_markdown).read_text(encoding="utf-8")
    raw = remove_review_file_index(raw)
    raw = strip_local_paths(raw)
    raw = inject_assets(raw)

    html_blocks = {
        "fig1": image_html("fig1"),
        "fig_E": image_html("fig_E"),
        "fig_G": image_html("fig_G"),
        "fig_G2": image_html("fig_G2"),
        "fig_G3": image_html("fig_G3"),
        "fig_H": image_html("fig_H"),
        "appendix_B": asset_table_html("appendix_B", "附录 B 完整配对显著性表"),
        "appendix_C": asset_table_html("appendix_C", "附录 C 完整 graph-dependent 参考表"),
        "appendix_D1": asset_table_html("appendix_D1", "附录 D.1 完整参数敏感性网格"),
        "appendix_D2": asset_table_html("appendix_D2", "附录 D.2 完整 β 选择审计表"),
        "appendix_F": asset_table_html("appendix_F", "附录 F 完整 Cut Value 表"),
        "appendix_H_mono": asset_table_html("appendix_H_mono", "附录 H 单调性核验表"),
        "table_B": asset_table_html("table_B", "表 I.1 graph-free 家族对照全表"),
        "extension_distill": asset_table_html("extension_distill", "表 I.2 蒸馏扩展五臂全表"),
    }
    body = markdown_to_html(raw, html_blocks)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LCSS 论文初稿 HTML 审阅版</title>
  <style>
    :root {{
      --fg: #1f2933;
      --muted: #5b6778;
      --border: #d8dee8;
      --soft: #f6f8fb;
      --accent: #2454a6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #ffffff;
      color: var(--fg);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans CJK SC", "Microsoft YaHei", Arial, sans-serif;
      line-height: 1.72;
      font-size: 16px;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 28px 72px;
    }}
    h1, h2, h3, h4 {{
      line-height: 1.25;
      margin: 2.0em 0 0.75em;
      color: #17202a;
    }}
    h1 {{ font-size: 2rem; border-bottom: 2px solid var(--fg); padding-bottom: 0.35em; }}
    h2 {{ font-size: 1.55rem; border-bottom: 1px solid var(--border); padding-bottom: 0.3em; }}
    h3 {{ font-size: 1.25rem; }}
    h4 {{ font-size: 1.05rem; color: var(--accent); }}
    p {{ margin: 0.72em 0; }}
    blockquote {{
      border-left: 4px solid var(--accent);
      margin: 1.2em 0;
      padding: 0.6em 1em;
      background: var(--soft);
      color: var(--muted);
    }}
    code {{
      background: #eef2f7;
      padding: 0.1em 0.32em;
      border-radius: 4px;
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 0.92em;
    }}
    .table-wrap {{
      width: 100%;
      overflow-x: auto;
      margin: 1em 0 1.5em;
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      min-width: 760px;
      font-size: 0.92rem;
      background: white;
    }}
    th, td {{
      border: 1px solid var(--border);
      padding: 7px 9px;
      vertical-align: top;
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      background: #eef3fa;
      font-weight: 650;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    tr:nth-child(even) td {{ background: #fbfcfe; }}
    .paper-figure {{
      margin: 1.4em auto 1.8em;
      padding: 12px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #fbfcfe;
    }}
    .paper-figure img {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 4px;
    }}
    figcaption {{
      color: var(--muted);
      font-size: 0.92rem;
      margin-top: 0.65em;
      text-align: center;
    }}
    .embedded-asset {{
      margin: 1.25em 0 1.8em;
      padding: 0.9em 1em 0.4em;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #fcfdff;
    }}
    hr {{
      border: 0;
      border-top: 1px solid var(--border);
      margin: 2em 0;
    }}
    @media print {{
      main {{ max-width: none; padding: 18mm; }}
      .table-wrap {{ overflow: visible; }}
      table {{ font-size: 8.5pt; }}
      h1, h2, h3 {{ page-break-after: avoid; }}
      .paper-figure, .embedded-asset {{ page-break-inside: avoid; }}
    }}
  </style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the embedded manuscript review HTML.")
    parser.add_argument("--src", type=Path, default=SRC_MD, help="Source Markdown manuscript.")
    parser.add_argument("--out", type=Path, default=OUT_HTML, help="Output HTML path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source_markdown = args.src.resolve()
    output_html = args.out.resolve()
    missing = [
        path
        for path in [source_markdown, *TABLE_ASSETS.values(), *(p for p, _ in FIGURE_ASSETS.values())]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required assets:\n" + "\n".join(str(path) for path in missing))

    html_text = build_html(source_markdown)
    forbidden_patterns = [
        r"C:\\Users\\Admin\\Desktop",
        r"paper_assets_20260706",
        r"`(?:figures|tables)/",
        r"runs[\\/]",
        r"\.csv",
        r"\.pdf",
        r"\.png",
        r"\.tex",
        r"\.md",
        r"见下方内嵌内容",
    ]
    for pattern in forbidden_patterns:
        if re.search(pattern, html_text):
            raise RuntimeError(f"Generated HTML still contains local path-like text: {pattern}")

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_text, encoding="utf-8")
    print(f"Wrote {output_html}")
    print(f"Size: {output_html.stat().st_size} bytes")


if __name__ == "__main__":
    main()
