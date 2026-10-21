"""Download explicitly listed public papers; extract page-marked text, not run code."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json

import fitz
import requests

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "RMCNC": "https://xlearning-lab.com/assets/2024-TKDE-Robust-Multi-View-Clustering-with-Noisy-Correspondence.pdf",
    "CANDY": "https://xlearning-lab.com/assets/2024-NeurIPS-Robust-Contrastive-Multi-view-Clustering-against-Dual-Noisy-Correspondence.pdf",
    "ROLL": "https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_ROLL_Robust_Noisy_Pseudo-label_Learning_for_Multi-View_Clustering_with_Noisy_CVPR_2025_paper.pdf",
    "CorreGen": "https://openreview.net/pdf?id=a4S1nQay3b",
    "SURE": "https://xlearning-lab.com/assets/2022-TPAMI-Robust-Multi-view-Clustering-with-Incomplete-Information.pdf",
    "DIVIDE": "https://arxiv.org/pdf/2308.11164",
    "MvCLN": "https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Partially_View-Aligned_Representation_Learning_With_Noise-Robust_Contrastive_Loss_CVPR_2021_paper.pdf",
    "PVC": "https://papers.nips.cc/paper_files/paper/2020/file/1e591403ff232de0f0f139ac51d99295-Paper.pdf",
    "CPSPAN": "https://openaccess.thecvf.com/content/CVPR2023/papers/Jin_Deep_Incomplete_Multi-View_Clustering_With_Cross-View_Partial_Sample_and_Prototype_CVPR_2023_paper.pdf",
    "MGBCC": "https://arxiv.org/pdf/2412.13550",
    "HOARD": "https://ojs.aaai.org/index.php/AAAI/article/view/40136/44097",
    "GFSAF": "https://ojs.aaai.org/index.php/AAAI/article/download/39556/43517",
    "NCR": "https://papers.nips.cc/paper_files/paper/2021/file/f5e62af885293cf4d511ceef31e61c80-Paper.pdf",
    "DECL": "https://xlearning-lab.com/assets/2022-ACMMM-Deep-Evidential-Learning-with-Noisy-Correspondence-for-Cross-modal-Retrieval.pdf",
    "BiCro": "https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_BiCro_Noisy_Correspondence_Rectification_for_Multi-Modality_Data_via_Bi-Directional_Cross-Modal_CVPR_2023_paper.pdf",
}


def fetch(name, url):
    entry = {"id": name, "requested_url": url}
    pdf_path = ROOT / "sources" / f"{name}.pdf"
    try:
        if pdf_path.exists():
            data = pdf_path.read_bytes()
        else:
            response = requests.get(url, timeout=(15, 45))
            response.raise_for_status()
            entry["resolved_url"] = response.url
            data = response.content
            if not data.startswith(b"%PDF"):
                raise ValueError("Response is not a PDF")
            pdf_path.write_bytes(data)
        with fitz.open(stream=data, filetype="pdf") as doc:
            pages = [f"\n===== PDF PAGE {i + 1} =====\n{p.get_text(sort=True)}"
                     for i, p in enumerate(doc)]
        (ROOT / "sources" / f"{name}.txt").write_text("".join(pages), encoding="utf-8")
        entry.update(status="ok", pages=len(pages), bytes=len(data),
                     sha256=hashlib.sha256(data).hexdigest())
    except Exception as exc:
        entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return entry


if __name__ == "__main__":
    (ROOT / "sources").mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(fetch, key, value) for key, value in SOURCES.items()]
        for future in as_completed(jobs):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=True), flush=True)
    (ROOT / "source_manifest.json").write_text(
        json.dumps(sorted(results, key=lambda x: x["id"]), indent=2, ensure_ascii=False),
        encoding="utf-8")
