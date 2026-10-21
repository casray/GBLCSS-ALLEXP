"""Validate the reading registry and assemble saved agent reports (no training)."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIELDS = {
    "A_granular": "粒球生成与粗粒度表示",
    "B_coarsening": "图粗化、凝缩与计算保持",
    "C_noisy_graph": "带噪图学习与鲁棒压缩",
    "D_heterophily": "异配表示、传播与粗化",
    "E_statistics": "可靠分裂、剪枝与预算分配",
}
REQUIRED = {
    "id", "title", "authors", "year", "venue", "doi_or_arxiv",
    "primary_url", "fulltext_url", "code_url", "selection_reason",
    "read_sections", "reading_depth", "core_counted", "mechanism",
    "assumptions", "protocol", "complexity", "limitation",
    "design_implication", "evidence_locator",
}


def main():
    registry, reports, counts, errors = [], [], {}, []
    seen = {}
    seen_identifiers = {}
    for folder, field in FIELDS.items():
        path = ROOT / folder
        for name in ("papers.json", "REPORT.md", "SEARCH_LOG.md"):
            if not (path / name).is_file():
                errors.append(f"Missing {folder}/{name}")
        if not (path / "papers.json").is_file():
            continue
        papers = json.loads((path / "papers.json").read_text(encoding="utf-8-sig"))
        count = 0
        for paper in papers:
            missing = sorted(REQUIRED - paper.keys())
            if missing:
                errors.append(f"{folder}/{paper.get('id')}: missing {missing}")
            if paper.get("core_counted"):
                count += 1
                if paper.get("reading_depth") not in ("methods_results", "theory_methods"):
                    errors.append(f"Core without sufficient depth: {paper.get('id')}")
                for key in ("primary_url", "fulltext_url", "read_sections", "evidence_locator"):
                    if not paper.get(key):
                        errors.append(f"Missing core evidence: {paper.get('id')}/{key}")
                title = re.sub(r"\W+", "", paper["title"]).casefold()
                if title in seen:
                    errors.append(f"Duplicate core title: {paper['id']} and {seen[title]}")
                seen[title] = paper["id"]
                identifier_text = str(paper.get("doi_or_arxiv") or "")
                identifiers = re.findall(r"10\.\d{4,9}/[^\s;]+", identifier_text, re.I)
                identifiers += ["arxiv:" + value for value in re.findall(r"arxiv:\s*(\d{4}\.\d{4,5})", identifier_text, re.I)]
                for value in identifiers:
                    value = value.rstrip(".,)").casefold()
                    if value in seen_identifiers:
                        errors.append(f"Duplicate core identifier {value}: {paper['id']} and {seen_identifiers[value]}")
                    seen_identifiers[value] = paper["id"]
            registry.append({"field": folder, **paper})
        counts[folder] = count
        if count < 7:
            errors.append(f"{folder}: only {count} core papers")
        if (path / "REPORT.md").is_file():
            report = (path / "REPORT.md").read_text(encoding="utf-8-sig")
            if re.search(r"中间稿|待补齐", report[:350]):
                errors.append(f"Unfinished report marker: {folder}")
            reports.append(f"# {folder}：{field}\n\n" + report)
            if (path / "CROSS_CHECK.md").is_file():
                reports.append(f"# {folder}：补充交叉复核\n\n" + (path / "CROSS_CHECK.md").read_text(encoding="utf-8-sig"))
    manifest = {"date": "2026-10-19", "core_counts": counts,
                "total_core": sum(counts.values()), "errors": errors,
                "scope": "Metadata/depth declaration validation; not independent verification of every claim."}
    (ROOT / "audit_reading_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    (ROOT / "all_papers.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 五领域文献阅读索引", "", "2026-10-19。Luna/max分领域阅读，主代理复核关键设计断言。不是逐页精读全部文献或穷尽SOTA的声明。经典/代表性依据方法关联性，未编造被引次数。", "",
             "| 领域 | 核心篇数 |", "|---|---:|"]
    for folder, field in FIELDS.items():
        lines.append(f"| {field} | {counts[folder]} |")
    lines += ["", f"共 {sum(counts.values())} 篇核心条目；题名及已登记DOI/arXiv去重通过。缺标识符或异名仍需结合题录人工复核。", ""]
    for folder, field in FIELDS.items():
        lines += [f"## {field}", "", "| 论文与原始来源 | 年份/发表 | 已读范围 | 状态 |", "|---|---|---|---|"]
        for p in (item for item in registry if item["field"] == folder):
            esc = lambda x: str(x).replace("|", "/").replace("\n", " ")
            lines.append(f"| [{esc(p['title'])}]({p['primary_url']}) | {esc(p['year'])} / {esc(p['venue'])} | {esc(p['read_sections'])} | {'核心' if p['core_counted'] else '补充，不计数'} |")
        lines += [""]
    (ROOT / "03_PAPER_INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    intro = "# 五领域Luna/max报告合订本\n\n本文件汇集各代理报告，方便一次查看。来源版本、阅读范围和未决事项以各篇卡片为准；主代理独立审计见02_COORDINATOR_AUDIT.md，设计裁决见05_DESIGN_REASSESSMENT.md。\n\n"
    (ROOT / "04_ALL_AGENT_REPORTS.md").write_text(intro + "\n\n---\n\n".join(reports), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
