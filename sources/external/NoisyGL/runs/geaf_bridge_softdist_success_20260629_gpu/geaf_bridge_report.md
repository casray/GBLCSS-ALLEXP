# GEAF bridge diagnostic report

Scope: offline hard-correction diagnostic. Clean labels are diagnostic only.

Raw rows: 90
Gate passed settings: 2

## Gate decisions

| dataset | confidence | gate | hard_net | ncr | cdr | hard_net_delta | positive seeds | reason |
|---|---:|---|---:|---:|---:|---:|---:|---|
| citeseer | 0.60 | FAIL | 0.0521 | 0.1008 | 0.0487 | 0.0932 | 5 | cdr above threshold |
| citeseer | 0.70 | FAIL | 0.0172 | 0.0329 | 0.0157 | 0.0244 | 4 | hard_net below threshold; ncr below threshold |
| citeseer | 0.80 | FAIL | 0.0078 | 0.0113 | 0.0035 | 0.0098 | 5 | hard_net below threshold; ncr below threshold |
| coauthorcs | 0.60 | PASS | 0.2263 | 0.2405 | 0.0142 | 0.2376 | 5 | passed |
| coauthorcs | 0.70 | PASS | 0.0890 | 0.0935 | 0.0045 | 0.0901 | 5 | passed |
| coauthorcs | 0.80 | FAIL | 0.0338 | 0.0353 | 0.0015 | 0.0339 | 5 | ncr below threshold |
| cora | 0.60 | FAIL | 0.1286 | 0.1645 | 0.0359 | 0.1656 | 5 | cdr above threshold |
| cora | 0.70 | FAIL | 0.0393 | 0.0453 | 0.0061 | 0.0391 | 5 | ncr below threshold |
| cora | 0.80 | FAIL | 0.0114 | 0.0132 | 0.0019 | 0.0114 | 5 | hard_net below threshold; ncr below threshold |

## Aggregate snapshot

- citeseer confidence=0.60: hard_net=0.0521, ncr=0.1008, cdr=0.0487, balls=212.0, target_size=24.0
- citeseer confidence=0.70: hard_net=0.0172, ncr=0.0329, cdr=0.0157, balls=212.0, target_size=24.0
- citeseer confidence=0.80: hard_net=0.0078, ncr=0.0113, cdr=0.0035, balls=212.0, target_size=24.0
- coauthorcs confidence=0.60: hard_net=0.2263, ncr=0.2405, cdr=0.0142, balls=805.0, target_size=35.0
- coauthorcs confidence=0.70: hard_net=0.0890, ncr=0.0935, cdr=0.0045, balls=805.0, target_size=35.0
- coauthorcs confidence=0.80: hard_net=0.0338, ncr=0.0353, cdr=0.0015, balls=805.0, target_size=35.0
- cora confidence=0.60: hard_net=0.1286, ncr=0.1645, cdr=0.0359, balls=189.0, target_size=20.0
- cora confidence=0.70: hard_net=0.0393, ncr=0.0453, cdr=0.0061, balls=189.0, target_size=20.0
- cora confidence=0.80: hard_net=0.0114, ncr=0.0132, cdr=0.0019, balls=189.0, target_size=20.0
