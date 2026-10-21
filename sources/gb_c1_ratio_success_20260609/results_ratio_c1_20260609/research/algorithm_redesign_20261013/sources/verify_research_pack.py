"""Audit coverage and source preservation; generate a navigable reading index."""
from pathlib import Path
import hashlib
import json
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]

CORE = "CGBC GBmooer MGBCC GFSAF GBC-CNN GBGCL SAGHL MUGBP MGCF-Net".split()
RELATIONS = [
    ("MFLVC", "mflvc"), ("MANGO", "mango"), ("RAV", "rav"),
    ("MCSF", "mcsf"), ("Medusa", "medusa"), ("AIRMVC", "airmvc"),
    ("PROTOCOL", "protocol"), ("CPMN", "cpmn"), ("MGNR", "mgnr"),
    ("CAL", "cal"), ("AF-UMC", "af_umc"), ("NGCE", "ngce"),
    ("SegGBC", "seggbc"), ("STCMC-UR", "stcmc_ur"),
    ("RAC-DMVC", "rac_dmvc"), ("RGCL", "rgcl"), ("GCFAggMVC", "gcfaggmvc"),
]
NOISE = "AGPCS-OIC BNO-GB GBC-GL GBSVR Rank-N-Contrast SupCon GBS VPGB GBRAIN G-GBC SNSCL NR-GNN RNCGLN CR-GNN PLReMix SymNCE ProCon".split()
entries = [(n, f"reviews/core/{n}.md") for n in CORE]
entries += [(n, f"reviews/relations/{s}.md") for n, s in RELATIONS]
entries += [(n, f"reviews/noise/{'GB_RAIN' if n == 'GBRAIN' else n}.md") for n in NOISE]
entries += [("GRACE", "reviews/GRACE.md")]
entries += [(n, "reviews/noise/foundations.md") for n in ("SCE", "Co-teaching", "ELR")]

def norm(value):
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[\s_\-–—]+", "", value)

raw = (ROOT / "sources/gpt_frontier.txt").read_text(encoding="utf-8-sig")
parts = re.split(r"===== PDF PAGE (\d+) =====", raw)
pages = [(int(parts[i]), norm(parts[i + 1])) for i in range(1, len(parts), 2)]
aliases = {"SupCon": ["SupCon", "Supervised Contrastive Learning"]}
coverage = []
for name, relative in entries:
    path = ROOT / relative
    note = path.read_text(encoding="utf-8-sig") if path.exists() else ""
    mentions = [p for p, body in pages if any(norm(a) in body for a in aliases.get(name, [name]))]
    coverage.append({"paper": name, "note": relative, "exists": path.is_file(),
                     "nonempty": bool(note.strip()), "pdf_pages": mentions,
                     "has_source_link": "https://" in note or "http://" in note})

manifest = json.loads((ROOT / "sources/input_manifest.json").read_text(encoding="utf-8-sig"))
checks = []
for key in ("pdf", "zip"):
    path = Path(manifest[key])
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    checks.append({"path": str(path), "unchanged": actual == manifest[key + "_sha256"]})
local_audit = json.loads((ROOT / "sources/local_evidence_audit.json").read_text(encoding="utf-8-sig"))
for item in local_audit["inspected_files"]:
    path = Path(item["path"])
    checks.append({"path": str(path), "unchanged": hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]})

missing = [c["paper"] for c in coverage if not c["nonempty"] or not c["has_source_link"]]
report = {"named_papers": len(coverage), "distinct_note_files": len({c["note"] for c in coverage}),
          "coverage": coverage, "missing_or_unsourced": missing, "source_checks": checks,
          "boundary": "Presence/link/hash audit only; not independent validation of every paper claim or external URL availability."}
(ROOT / "sources/literature_coverage.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

lines = [
    "# 阅读索引与研究入口", "", "更新：2026-10-13。",
    "", "本轮整理 GPT 交流文档中的 47 个具名论文/方法条目，写入 45 份文献卡；SCE、Co-teaching、ELR 共用基础卡。CE 等一般概念不计论文，参考论文再引用的所有文献不递归扩展。覆盖数不是全文精读数。",
    "", "## 建议阅读顺序", "",
    "1. [项目与旧实验复盘](01_项目与旧实验复盘.md)：最终稿真正做了什么，哪些旧结论需要纠正。",
    "2. [文献核查与机制比较](02_文献核查与机制比较.md)：现有方法覆盖范围、可迁移机制及证据边界。",
    "3. [算法重设计建议](03_算法重设计建议.md)：推荐候选的公式、两条部署路线、对照及停止条件。",
    "4. [研究记忆与接续](99_研究记忆与接续.md)：后续窗口必须继承的事实和限制。",
    "", "## 阅读深度与缺口", "",
    "逐篇卡区分全文方法/实验核读、部分方法或原文预览、题录/摘要、二手检索线索。数值是否逐表复核另行说明；取得全文不代表已复现论文。",
    "", "- [核心粒球组记忆](reviews/core/00_core_memory.md)：9 篇；SAGHL、GBGCL、MUGBP 尚缺全文核读。",
    "- [关系与多视图组记忆](reviews/relations/00_relations_memory.md)：17 篇；MANGO、MCSF、NGCE、CAL、SegGBC 仍有全文或实验表缺口。",
    "- [噪声学习组记忆](reviews/noise/00_noise_memory.md)：17 篇及 3 篇基础方法；CR-GNN、NR-GNN、GBRAIN 等尚缺全文，AGPCS-OIC 仅二手索引线索，不能作为已核技术证据。",
    "", "没有把未取得的全文伪装为已经精读。新颖性判断因此是条件性判断，尤其需要继续核对 SAGHL/GBGCL。",
    "", "## 逐篇目录", "", "页码为 GPT 交流 PDF 中缩写/名称的检索位置，可能有缩写包含关系；不是原论文页码。", "",
    "| 条目 | GPT PDF 检索页 | 笔记 |", "|---|---|---|",
]
for row in coverage:
    lines.append(f"| {row['paper']} | {', '.join(map(str, row['pdf_pages'])) or '需人工对照'} | [阅读]({row['note']}) |")
lines += ["", "## 原始证据与验证", "",
          "- `sources/gpt_frontier.txt`：49 页逐页提取；`gpt_frontier_links.json`：55 个链接标注（含重复）。",
          "- `paper_snapshot/`：投稿稿件 ZIP 的阅读副本；原始 ZIP 未修改。",
          "- `sources/local_evidence_audit.json`：40 份 C1 配置签名与 Stage A 历史 CSV 重算。",
          "- `sources/literature_coverage.json`：条目覆盖、来源链接存在性和原始输入/所审源码哈希比对。",
          "- 本轮仅阅读、检索、审计和写研究文档，没有启动新模型训练。",
          "", "所有未来实验仍应先读取原始文献图与实验历史，不能以本索引代替配置、raw log 和逐 seed 结果。", ""]
(ROOT / "00_阅读索引.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"named_papers": len(coverage), "distinct_note_files": report["distinct_note_files"],
                  "missing_or_unsourced": missing, "all_source_hashes_unchanged": all(c["unchanged"] for c in checks)}, ensure_ascii=True))
if missing or not all(c["unchanged"] for c in checks):
    raise SystemExit(1)
