"""Archive supplemental literature without changing earlier acquisition records."""
import collect_sources as base

base.ROOT = base.ROOT / 'supplement'
base.ROOT.mkdir(exist_ok=True)
base.SOURCES = {
    'HarmonyGNNs_official_2026': 'https://proceedings.iclr.cc/paper_files/paper/2026/file/c33176723974e9b50ca0297d769903c8-Paper-Conference.pdf',
    'SpaM_2026': 'https://arxiv.org/html/2601.01207v1',
    'GraphPFN_current_2026': 'https://arxiv.org/html/2509.21489',
}

if __name__ == '__main__':
    base.main()
