# GEAF bridge diagnostic report

Scope: offline hard-correction diagnostic. Clean labels are diagnostic only.

Raw rows: 120
Gate passed settings: 6

## Gate decisions

| dataset | confidence | gate | hard_net | ncr | cdr | hard_net_delta | positive seeds | reason |
|---|---:|---|---:|---:|---:|---:|---:|---|
| amazoncom | 0.00 | FAIL | 0.3682 | 0.5763 | 0.2081 | 0.7800 | 5 | cdr above threshold |
| amazoncom | 0.60 | FAIL | 0.1604 | 0.1906 | 0.0302 | 0.1749 | 5 | cdr above threshold |
| amazoncom | 0.70 | PASS | 0.0574 | 0.0646 | 0.0072 | 0.0589 | 5 | passed |
| amazonpho | 0.00 | FAIL | 0.4264 | 0.5952 | 0.1687 | 0.8849 | 5 | cdr above threshold |
| amazonpho | 0.60 | PASS | 0.1963 | 0.2220 | 0.0257 | 0.2206 | 5 | passed |
| amazonpho | 0.70 | PASS | 0.0739 | 0.0809 | 0.0070 | 0.0770 | 5 | passed |
| dblp | 0.00 | FAIL | 0.5054 | 0.6836 | 0.1782 | 0.6785 | 5 | cdr above threshold |
| dblp | 0.60 | PASS | 0.1955 | 0.2209 | 0.0254 | 0.2079 | 5 | passed |
| dblp | 0.70 | PASS | 0.0685 | 0.0761 | 0.0076 | 0.0709 | 5 | passed |
| pubmed | 0.00 | FAIL | 0.4429 | 0.6489 | 0.2060 | 0.6840 | 5 | cdr above threshold |
| pubmed | 0.60 | FAIL | 0.1886 | 0.2246 | 0.0360 | 0.2109 | 5 | cdr above threshold |
| pubmed | 0.70 | PASS | 0.0642 | 0.0736 | 0.0094 | 0.0680 | 5 | passed |

## Aggregate snapshot

- amazoncom confidence=0.00: hard_net=0.3682, ncr=0.5763, cdr=0.2081, balls=549.0, target_size=38.0
- amazoncom confidence=0.60: hard_net=0.1604, ncr=0.1906, cdr=0.0302, balls=549.0, target_size=38.0
- amazoncom confidence=0.70: hard_net=0.0574, ncr=0.0646, cdr=0.0072, balls=549.0, target_size=38.0
- amazonpho confidence=0.00: hard_net=0.4264, ncr=0.5952, cdr=0.1687, balls=375.0, target_size=31.0
- amazonpho confidence=0.60: hard_net=0.1963, ncr=0.2220, cdr=0.0257, balls=375.0, target_size=31.0
- amazonpho confidence=0.70: hard_net=0.0739, ncr=0.0809, cdr=0.0070, balls=375.0, target_size=31.0
- dblp confidence=0.00: hard_net=0.5054, ncr=0.6836, cdr=0.1782, balls=408.0, target_size=67.0
- dblp confidence=0.60: hard_net=0.1955, ncr=0.2209, cdr=0.0254, balls=408.0, target_size=67.0
- dblp confidence=0.70: hard_net=0.0685, ncr=0.0761, cdr=0.0076, balls=408.0, target_size=67.0
- pubmed confidence=0.00: hard_net=0.4429, ncr=0.6489, cdr=0.2060, balls=367.0, target_size=82.0
- pubmed confidence=0.60: hard_net=0.1886, ncr=0.2246, cdr=0.0360, balls=367.0, target_size=82.0
- pubmed confidence=0.70: hard_net=0.0642, ncr=0.0736, cdr=0.0094, balls=367.0, target_size=82.0
