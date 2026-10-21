"""Preserve first acquisition failures; fetch verified alternative primary links."""
import collect_sources as base

base.SOURCES = {
    'Polynormer_official_2024': 'https://proceedings.iclr.cc/paper_files/paper/2024/file/4ff87ac8f2f86bd471e25bed6eb69d77-Paper-Conference.pdf',
    'HiGCN_official_2024': 'https://ojs.aaai.org/index.php/AAAI/article/download/29160/30194',
    'critical_benchmark_2023': 'https://arxiv.org/pdf/2302.11640',
    'NT_main': 'https://raw.githubusercontent.com/cf020031308/MoNT/main/main.py',
    'NT_reproduce': 'https://raw.githubusercontent.com/cf020031308/MoNT/main/reproduce.sh',
    'Polynormer_dataset': 'https://raw.githubusercontent.com/cornell-zhang/Polynormer/master/dataset.py',
    'Polynormer_main': 'https://raw.githubusercontent.com/cornell-zhang/Polynormer/master/main.py',
}
first = base.ROOT / 'source_manifest.json'
first.replace(base.ROOT / 'source_manifest_initial.json')
try:
    base.main()
finally:
    if first.exists():
        first.replace(base.ROOT / 'source_manifest_followup.json')
    (base.ROOT / 'source_manifest_initial.json').replace(first)
