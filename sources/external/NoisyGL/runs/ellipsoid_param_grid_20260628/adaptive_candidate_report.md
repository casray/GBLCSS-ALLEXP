# Ellipsoid SoftDist adaptive parameter grid report

Scope: offline target diagnostic only. Clean labels are used only for diagnosis and gate checks.
This report is not downstream accuracy evidence and should not be used as a paper claim by itself.

Raw rows: 1600
Error rows: 0

## Dataset conclusions

- amazoncom: ball-count-sensitive; best=max_balls=64, split_improvement=0.001, passed=True, mass_delta=10.9144, cdr_delta=0.0007, ncr_delta=0.0007, balls=64.0; max_balls changed the gate decision or clean-mass delta materially
- amazonpho: ball-count-sensitive; best=max_balls=64, split_improvement=0.05, passed=True, mass_delta=7.9166, cdr_delta=-0.0018, ncr_delta=-0.0000, balls=64.0; max_balls changed the gate decision or clean-mass delta materially
- dblp: collapse-fixed; best=max_balls=128, split_improvement=0.001, passed=True, mass_delta=2.9623, cdr_delta=0.0044, ncr_delta=-0.0102, balls=128.0; a collapsed setting failed and a lower-threshold setting recovered usable balls
- pubmed: ball-count-sensitive; best=max_balls=512, split_improvement=0.01, passed=True, mass_delta=1.8385, cdr_delta=0.0015, ncr_delta=0.0141, balls=376.2; max_balls changed the gate decision or clean-mass delta materially

## Direct answers

- DBLP split adaptation: yes. Lower split_improvement recovers balls and passes the gate; best=max_balls=128, split=0.001.
- PubMed max_balls impact: yes. Gate and clean-mass delta change with max_balls; best=max_balls=512, split=0.01.
- amazonpho 64/256 cap: no evidence that the cap is too low; best offline setting uses max_balls=64 and higher caps are weaker.
- amazoncom 64/256 cap: no evidence that the cap is too low; best offline setting uses max_balls=64 and higher caps are weaker.

## Worth downstream

- amazoncom: max_balls=64, split_improvement=0.001, mass_delta=10.9144, cdr_delta=0.0007, ncr_delta=0.0007, positive_seeds=5, fallback=0.0000, balls=64.0
- amazoncom: max_balls=64, split_improvement=0.005, mass_delta=10.9144, cdr_delta=0.0007, ncr_delta=0.0007, positive_seeds=5, fallback=0.0000, balls=64.0
- amazoncom: max_balls=64, split_improvement=0.02, mass_delta=10.9144, cdr_delta=0.0007, ncr_delta=0.0007, positive_seeds=5, fallback=0.0000, balls=64.0
- amazoncom: max_balls=64, split_improvement=0.05, mass_delta=10.9144, cdr_delta=0.0007, ncr_delta=0.0007, positive_seeds=5, fallback=0.0000, balls=64.0
- amazoncom: max_balls=64, split_improvement=0.01, mass_delta=10.7577, cdr_delta=0.0007, ncr_delta=0.0007, positive_seeds=5, fallback=0.0000, balls=64.0
- amazoncom: max_balls=128, split_improvement=0.001, mass_delta=6.7195, cdr_delta=0.0018, ncr_delta=0.0009, positive_seeds=5, fallback=0.0000, balls=128.0
- amazoncom: max_balls=128, split_improvement=0.005, mass_delta=6.7195, cdr_delta=0.0018, ncr_delta=0.0009, positive_seeds=5, fallback=0.0000, balls=128.0
- amazoncom: max_balls=128, split_improvement=0.01, mass_delta=6.7195, cdr_delta=0.0018, ncr_delta=0.0009, positive_seeds=5, fallback=0.0000, balls=128.0
- amazoncom: max_balls=128, split_improvement=0.02, mass_delta=6.7195, cdr_delta=0.0018, ncr_delta=0.0009, positive_seeds=5, fallback=0.0000, balls=128.0
- amazoncom: max_balls=128, split_improvement=0.05, mass_delta=6.7195, cdr_delta=0.0018, ncr_delta=0.0009, positive_seeds=5, fallback=0.0000, balls=128.0
- amazoncom: max_balls=256, split_improvement=0.02, mass_delta=3.0378, cdr_delta=0.0034, ncr_delta=-0.0024, positive_seeds=4, fallback=0.0000, balls=232.6
- amazoncom: max_balls=256, split_improvement=0.001, mass_delta=2.9812, cdr_delta=0.0033, ncr_delta=-0.0026, positive_seeds=4, fallback=0.0000, balls=234.6
- amazoncom: max_balls=256, split_improvement=0.005, mass_delta=2.9812, cdr_delta=0.0033, ncr_delta=-0.0026, positive_seeds=4, fallback=0.0000, balls=234.6
- amazoncom: max_balls=256, split_improvement=0.01, mass_delta=2.9812, cdr_delta=0.0033, ncr_delta=-0.0026, positive_seeds=4, fallback=0.0000, balls=234.6
- amazoncom: max_balls=256, split_improvement=0.05, mass_delta=2.9812, cdr_delta=0.0033, ncr_delta=-0.0026, positive_seeds=4, fallback=0.0000, balls=234.6
- amazoncom: max_balls=512, split_improvement=0.001, mass_delta=2.9812, cdr_delta=0.0033, ncr_delta=-0.0026, positive_seeds=4, fallback=0.0000, balls=234.6
- amazoncom: max_balls=512, split_improvement=0.005, mass_delta=2.9812, cdr_delta=0.0033, ncr_delta=-0.0026, positive_seeds=4, fallback=0.0000, balls=234.6
- amazoncom: max_balls=512, split_improvement=0.01, mass_delta=2.9812, cdr_delta=0.0033, ncr_delta=-0.0026, positive_seeds=4, fallback=0.0000, balls=234.6
- amazoncom: max_balls=512, split_improvement=0.02, mass_delta=2.9812, cdr_delta=0.0033, ncr_delta=-0.0026, positive_seeds=4, fallback=0.0000, balls=234.6
- amazoncom: max_balls=512, split_improvement=0.05, mass_delta=2.9812, cdr_delta=0.0033, ncr_delta=-0.0026, positive_seeds=4, fallback=0.0000, balls=234.6
- amazonpho: max_balls=64, split_improvement=0.05, mass_delta=7.9166, cdr_delta=-0.0018, ncr_delta=-0.0000, positive_seeds=5, fallback=0.0000, balls=64.0
- amazonpho: max_balls=64, split_improvement=0.005, mass_delta=7.8240, cdr_delta=-0.0018, ncr_delta=-0.0000, positive_seeds=5, fallback=0.0000, balls=64.0
- amazonpho: max_balls=64, split_improvement=0.01, mass_delta=7.8240, cdr_delta=-0.0018, ncr_delta=-0.0000, positive_seeds=5, fallback=0.0000, balls=64.0
- amazonpho: max_balls=64, split_improvement=0.02, mass_delta=7.8240, cdr_delta=-0.0018, ncr_delta=-0.0000, positive_seeds=5, fallback=0.0000, balls=64.0
- amazonpho: max_balls=64, split_improvement=0.001, mass_delta=7.7704, cdr_delta=-0.0018, ncr_delta=-0.0000, positive_seeds=5, fallback=0.0000, balls=64.0
- amazonpho: max_balls=128, split_improvement=0.05, mass_delta=4.2983, cdr_delta=-0.0018, ncr_delta=-0.0012, positive_seeds=5, fallback=0.0000, balls=126.2
- amazonpho: max_balls=128, split_improvement=0.001, mass_delta=4.2909, cdr_delta=-0.0018, ncr_delta=-0.0009, positive_seeds=5, fallback=0.0000, balls=126.2
- amazonpho: max_balls=256, split_improvement=0.001, mass_delta=4.2206, cdr_delta=-0.0027, ncr_delta=0.0014, positive_seeds=5, fallback=0.0000, balls=132.4
- amazonpho: max_balls=128, split_improvement=0.005, mass_delta=4.1440, cdr_delta=-0.0023, ncr_delta=0.0004, positive_seeds=5, fallback=0.0000, balls=127.2
- amazonpho: max_balls=128, split_improvement=0.01, mass_delta=4.1440, cdr_delta=-0.0023, ncr_delta=0.0004, positive_seeds=5, fallback=0.0000, balls=127.2
- amazonpho: max_balls=512, split_improvement=0.005, mass_delta=4.0978, cdr_delta=-0.0025, ncr_delta=0.0017, positive_seeds=5, fallback=0.0000, balls=130.8
- amazonpho: max_balls=256, split_improvement=0.005, mass_delta=4.0799, cdr_delta=-0.0025, ncr_delta=0.0014, positive_seeds=5, fallback=0.0000, balls=131.4
- amazonpho: max_balls=256, split_improvement=0.01, mass_delta=4.0799, cdr_delta=-0.0025, ncr_delta=0.0014, positive_seeds=5, fallback=0.0000, balls=131.4
- amazonpho: max_balls=256, split_improvement=0.02, mass_delta=4.0799, cdr_delta=-0.0025, ncr_delta=0.0014, positive_seeds=5, fallback=0.0000, balls=131.4
- amazonpho: max_balls=256, split_improvement=0.05, mass_delta=4.0799, cdr_delta=-0.0025, ncr_delta=0.0014, positive_seeds=5, fallback=0.0000, balls=131.4
- amazonpho: max_balls=512, split_improvement=0.001, mass_delta=4.0799, cdr_delta=-0.0025, ncr_delta=0.0014, positive_seeds=5, fallback=0.0000, balls=131.4
- amazonpho: max_balls=512, split_improvement=0.01, mass_delta=4.0799, cdr_delta=-0.0025, ncr_delta=0.0014, positive_seeds=5, fallback=0.0000, balls=131.4
- amazonpho: max_balls=512, split_improvement=0.02, mass_delta=4.0799, cdr_delta=-0.0025, ncr_delta=0.0014, positive_seeds=5, fallback=0.0000, balls=131.4
- amazonpho: max_balls=512, split_improvement=0.05, mass_delta=3.9686, cdr_delta=-0.0034, ncr_delta=0.0037, positive_seeds=5, fallback=0.0000, balls=130.4
- amazonpho: max_balls=128, split_improvement=0.02, mass_delta=3.9408, cdr_delta=-0.0021, ncr_delta=0.0010, positive_seeds=5, fallback=0.0000, balls=126.2
- dblp: max_balls=128, split_improvement=0.001, mass_delta=2.9623, cdr_delta=0.0044, ncr_delta=-0.0102, positive_seeds=5, fallback=0.0000, balls=128.0
- dblp: max_balls=128, split_improvement=0.005, mass_delta=2.9623, cdr_delta=0.0044, ncr_delta=-0.0102, positive_seeds=5, fallback=0.0000, balls=128.0
- dblp: max_balls=128, split_improvement=0.01, mass_delta=2.9623, cdr_delta=0.0044, ncr_delta=-0.0102, positive_seeds=5, fallback=0.0000, balls=128.0
- dblp: max_balls=128, split_improvement=0.02, mass_delta=2.9623, cdr_delta=0.0044, ncr_delta=-0.0102, positive_seeds=5, fallback=0.0000, balls=128.0
- dblp: max_balls=64, split_improvement=0.001, mass_delta=2.8001, cdr_delta=0.0036, ncr_delta=-0.0084, positive_seeds=5, fallback=0.0000, balls=64.0
- dblp: max_balls=64, split_improvement=0.005, mass_delta=2.8001, cdr_delta=0.0036, ncr_delta=-0.0084, positive_seeds=5, fallback=0.0000, balls=64.0
- dblp: max_balls=64, split_improvement=0.01, mass_delta=2.8001, cdr_delta=0.0036, ncr_delta=-0.0084, positive_seeds=5, fallback=0.0000, balls=64.0
- dblp: max_balls=64, split_improvement=0.02, mass_delta=2.8001, cdr_delta=0.0036, ncr_delta=-0.0084, positive_seeds=5, fallback=0.0000, balls=64.0
- dblp: max_balls=256, split_improvement=0.01, mass_delta=0.9350, cdr_delta=0.0041, ncr_delta=-0.0022, positive_seeds=4, fallback=0.0000, balls=256.0
- dblp: max_balls=512, split_improvement=0.01, mass_delta=0.4278, cdr_delta=0.0072, ncr_delta=-0.0110, positive_seeds=4, fallback=0.0000, balls=338.4
- pubmed: max_balls=512, split_improvement=0.01, mass_delta=1.8385, cdr_delta=0.0015, ncr_delta=0.0141, positive_seeds=4, fallback=0.0000, balls=376.2
- pubmed: max_balls=256, split_improvement=0.005, mass_delta=1.5936, cdr_delta=0.0056, ncr_delta=-0.0033, positive_seeds=5, fallback=0.0000, balls=256.0
- pubmed: max_balls=256, split_improvement=0.05, mass_delta=1.2955, cdr_delta=0.0072, ncr_delta=0.0002, positive_seeds=4, fallback=0.0000, balls=256.0
- pubmed: max_balls=128, split_improvement=0.005, mass_delta=0.7427, cdr_delta=0.0037, ncr_delta=-0.0009, positive_seeds=4, fallback=0.0000, balls=128.0
- pubmed: max_balls=128, split_improvement=0.02, mass_delta=0.7366, cdr_delta=0.0034, ncr_delta=0.0003, positive_seeds=4, fallback=0.0000, balls=128.0
- pubmed: max_balls=128, split_improvement=0.05, mass_delta=0.5830, cdr_delta=0.0029, ncr_delta=0.0002, positive_seeds=4, fallback=0.0000, balls=128.0
- pubmed: max_balls=256, split_improvement=0.01, mass_delta=0.3719, cdr_delta=0.0068, ncr_delta=-0.0052, positive_seeds=4, fallback=0.0000, balls=256.0

## Gate table

| dataset | max_balls | split | gate | mass_delta | cdr_delta | ncr_delta | positive_seeds | beats_shuffled | fallback | balls | reason |
|---|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---|
| amazoncom | 64 | 0.001 | PASS | 10.9144 | 0.0007 | 0.0007 | 5 | True | 0.0000 | 64.0 | pass |
| amazoncom | 64 | 0.005 | PASS | 10.9144 | 0.0007 | 0.0007 | 5 | True | 0.0000 | 64.0 | pass |
| amazoncom | 64 | 0.01 | PASS | 10.7577 | 0.0007 | 0.0007 | 5 | True | 0.0000 | 64.0 | pass |
| amazoncom | 64 | 0.02 | PASS | 10.9144 | 0.0007 | 0.0007 | 5 | True | 0.0000 | 64.0 | pass |
| amazoncom | 64 | 0.05 | PASS | 10.9144 | 0.0007 | 0.0007 | 5 | True | 0.0000 | 64.0 | pass |
| amazoncom | 128 | 0.001 | PASS | 6.7195 | 0.0018 | 0.0009 | 5 | True | 0.0000 | 128.0 | pass |
| amazoncom | 128 | 0.005 | PASS | 6.7195 | 0.0018 | 0.0009 | 5 | True | 0.0000 | 128.0 | pass |
| amazoncom | 128 | 0.01 | PASS | 6.7195 | 0.0018 | 0.0009 | 5 | True | 0.0000 | 128.0 | pass |
| amazoncom | 128 | 0.02 | PASS | 6.7195 | 0.0018 | 0.0009 | 5 | True | 0.0000 | 128.0 | pass |
| amazoncom | 128 | 0.05 | PASS | 6.7195 | 0.0018 | 0.0009 | 5 | True | 0.0000 | 128.0 | pass |
| amazoncom | 256 | 0.001 | PASS | 2.9812 | 0.0033 | -0.0026 | 4 | True | 0.0000 | 234.6 | pass |
| amazoncom | 256 | 0.005 | PASS | 2.9812 | 0.0033 | -0.0026 | 4 | True | 0.0000 | 234.6 | pass |
| amazoncom | 256 | 0.01 | PASS | 2.9812 | 0.0033 | -0.0026 | 4 | True | 0.0000 | 234.6 | pass |
| amazoncom | 256 | 0.02 | PASS | 3.0378 | 0.0034 | -0.0024 | 4 | True | 0.0000 | 232.6 | pass |
| amazoncom | 256 | 0.05 | PASS | 2.9812 | 0.0033 | -0.0026 | 4 | True | 0.0000 | 234.6 | pass |
| amazoncom | 512 | 0.001 | PASS | 2.9812 | 0.0033 | -0.0026 | 4 | True | 0.0000 | 234.6 | pass |
| amazoncom | 512 | 0.005 | PASS | 2.9812 | 0.0033 | -0.0026 | 4 | True | 0.0000 | 234.6 | pass |
| amazoncom | 512 | 0.01 | PASS | 2.9812 | 0.0033 | -0.0026 | 4 | True | 0.0000 | 234.6 | pass |
| amazoncom | 512 | 0.02 | PASS | 2.9812 | 0.0033 | -0.0026 | 4 | True | 0.0000 | 234.6 | pass |
| amazoncom | 512 | 0.05 | PASS | 2.9812 | 0.0033 | -0.0026 | 4 | True | 0.0000 | 234.6 | pass |
| amazonpho | 64 | 0.001 | PASS | 7.7704 | -0.0018 | -0.0000 | 5 | True | 0.0000 | 64.0 | pass |
| amazonpho | 64 | 0.005 | PASS | 7.8240 | -0.0018 | -0.0000 | 5 | True | 0.0000 | 64.0 | pass |
| amazonpho | 64 | 0.01 | PASS | 7.8240 | -0.0018 | -0.0000 | 5 | True | 0.0000 | 64.0 | pass |
| amazonpho | 64 | 0.02 | PASS | 7.8240 | -0.0018 | -0.0000 | 5 | True | 0.0000 | 64.0 | pass |
| amazonpho | 64 | 0.05 | PASS | 7.9166 | -0.0018 | -0.0000 | 5 | True | 0.0000 | 64.0 | pass |
| amazonpho | 128 | 0.001 | PASS | 4.2909 | -0.0018 | -0.0009 | 5 | True | 0.0000 | 126.2 | pass |
| amazonpho | 128 | 0.005 | PASS | 4.1440 | -0.0023 | 0.0004 | 5 | True | 0.0000 | 127.2 | pass |
| amazonpho | 128 | 0.01 | PASS | 4.1440 | -0.0023 | 0.0004 | 5 | True | 0.0000 | 127.2 | pass |
| amazonpho | 128 | 0.02 | PASS | 3.9408 | -0.0021 | 0.0010 | 5 | True | 0.0000 | 126.2 | pass |
| amazonpho | 128 | 0.05 | PASS | 4.2983 | -0.0018 | -0.0012 | 5 | True | 0.0000 | 126.2 | pass |
| amazonpho | 256 | 0.001 | PASS | 4.2206 | -0.0027 | 0.0014 | 5 | True | 0.0000 | 132.4 | pass |
| amazonpho | 256 | 0.005 | PASS | 4.0799 | -0.0025 | 0.0014 | 5 | True | 0.0000 | 131.4 | pass |
| amazonpho | 256 | 0.01 | PASS | 4.0799 | -0.0025 | 0.0014 | 5 | True | 0.0000 | 131.4 | pass |
| amazonpho | 256 | 0.02 | PASS | 4.0799 | -0.0025 | 0.0014 | 5 | True | 0.0000 | 131.4 | pass |
| amazonpho | 256 | 0.05 | PASS | 4.0799 | -0.0025 | 0.0014 | 5 | True | 0.0000 | 131.4 | pass |
| amazonpho | 512 | 0.001 | PASS | 4.0799 | -0.0025 | 0.0014 | 5 | True | 0.0000 | 131.4 | pass |
| amazonpho | 512 | 0.005 | PASS | 4.0978 | -0.0025 | 0.0017 | 5 | True | 0.0000 | 130.8 | pass |
| amazonpho | 512 | 0.01 | PASS | 4.0799 | -0.0025 | 0.0014 | 5 | True | 0.0000 | 131.4 | pass |
| amazonpho | 512 | 0.02 | PASS | 4.0799 | -0.0025 | 0.0014 | 5 | True | 0.0000 | 131.4 | pass |
| amazonpho | 512 | 0.05 | PASS | 3.9686 | -0.0034 | 0.0037 | 5 | True | 0.0000 | 130.4 | pass |
| dblp | 64 | 0.001 | PASS | 2.8001 | 0.0036 | -0.0084 | 5 | True | 0.0000 | 64.0 | pass |
| dblp | 64 | 0.005 | PASS | 2.8001 | 0.0036 | -0.0084 | 5 | True | 0.0000 | 64.0 | pass |
| dblp | 64 | 0.01 | PASS | 2.8001 | 0.0036 | -0.0084 | 5 | True | 0.0000 | 64.0 | pass |
| dblp | 64 | 0.02 | PASS | 2.8001 | 0.0036 | -0.0084 | 5 | True | 0.0000 | 64.0 | pass |
| dblp | 64 | 0.05 | FAIL | -22.2654 | 0.0000 | 0.0000 | 0 | False | 0.0000 | 1.0 | no diagnostic gain; not enough positive seeds; not better than shuffled |
| dblp | 128 | 0.001 | PASS | 2.9623 | 0.0044 | -0.0102 | 5 | True | 0.0000 | 128.0 | pass |
| dblp | 128 | 0.005 | PASS | 2.9623 | 0.0044 | -0.0102 | 5 | True | 0.0000 | 128.0 | pass |
| dblp | 128 | 0.01 | PASS | 2.9623 | 0.0044 | -0.0102 | 5 | True | 0.0000 | 128.0 | pass |
| dblp | 128 | 0.02 | PASS | 2.9623 | 0.0044 | -0.0102 | 5 | True | 0.0000 | 128.0 | pass |
| dblp | 128 | 0.05 | FAIL | -22.2654 | 0.0000 | 0.0000 | 0 | False | 0.0000 | 1.0 | no diagnostic gain; not enough positive seeds; not better than shuffled |
| dblp | 256 | 0.001 | FAIL | 0.3825 | 0.0045 | -0.0065 | 3 | True | 0.0000 | 256.0 | not enough positive seeds |
| dblp | 256 | 0.005 | FAIL | 0.6030 | 0.0049 | -0.0070 | 3 | True | 0.0000 | 256.0 | not enough positive seeds |
| dblp | 256 | 0.01 | PASS | 0.9350 | 0.0041 | -0.0022 | 4 | True | 0.0000 | 256.0 | pass |
| dblp | 256 | 0.02 | FAIL | 0.3936 | 0.0059 | -0.0112 | 3 | True | 0.0000 | 256.0 | not enough positive seeds |
| dblp | 256 | 0.05 | FAIL | -22.2654 | 0.0000 | 0.0000 | 0 | False | 0.0000 | 1.0 | no diagnostic gain; not enough positive seeds; not better than shuffled |
| dblp | 512 | 0.001 | FAIL | 0.1446 | 0.0090 | -0.0155 | 3 | True | 0.0000 | 337.2 | not enough positive seeds |
| dblp | 512 | 0.005 | FAIL | 0.1988 | 0.0088 | -0.0156 | 3 | True | 0.0000 | 338.4 | not enough positive seeds |
| dblp | 512 | 0.01 | PASS | 0.4278 | 0.0072 | -0.0110 | 4 | True | 0.0000 | 338.4 | pass |
| dblp | 512 | 0.02 | FAIL | 0.0185 | 0.0084 | -0.0144 | 3 | True | 0.0000 | 338.8 | not enough positive seeds |
| dblp | 512 | 0.05 | FAIL | -22.2654 | 0.0000 | 0.0000 | 0 | False | 0.0000 | 1.0 | no diagnostic gain; not enough positive seeds; not better than shuffled |
| pubmed | 64 | 0.001 | FAIL | 0.7827 | 0.0002 | -0.0003 | 2 | False | 0.0000 | 64.0 | not enough positive seeds; not better than shuffled |
| pubmed | 64 | 0.005 | FAIL | 0.8803 | 0.0003 | -0.0003 | 2 | False | 0.0000 | 64.0 | not enough positive seeds; not better than shuffled |
| pubmed | 64 | 0.01 | FAIL | 0.7986 | 0.0002 | -0.0003 | 2 | False | 0.0000 | 64.0 | not enough positive seeds; not better than shuffled |
| pubmed | 64 | 0.02 | FAIL | 0.7679 | -0.0009 | 0.0032 | 2 | False | 0.0000 | 64.0 | not enough positive seeds; not better than shuffled |
| pubmed | 64 | 0.05 | FAIL | 0.7247 | 0.0002 | -0.0005 | 2 | False | 0.0000 | 64.0 | not enough positive seeds; not better than shuffled |
| pubmed | 128 | 0.001 | FAIL | 0.2407 | 0.0020 | 0.0020 | 3 | True | 0.0000 | 128.0 | not enough positive seeds |
| pubmed | 128 | 0.005 | PASS | 0.7427 | 0.0037 | -0.0009 | 4 | True | 0.0000 | 128.0 | pass |
| pubmed | 128 | 0.01 | FAIL | 0.2660 | 0.0031 | -0.0001 | 3 | True | 0.0000 | 128.0 | not enough positive seeds |
| pubmed | 128 | 0.02 | PASS | 0.7366 | 0.0034 | 0.0003 | 4 | True | 0.0000 | 128.0 | pass |
| pubmed | 128 | 0.05 | PASS | 0.5830 | 0.0029 | 0.0002 | 4 | True | 0.0000 | 128.0 | pass |
| pubmed | 256 | 0.001 | FAIL | 0.0466 | 0.0075 | -0.0060 | 3 | True | 0.0000 | 256.0 | not enough positive seeds |
| pubmed | 256 | 0.005 | PASS | 1.5936 | 0.0056 | -0.0033 | 5 | True | 0.0000 | 256.0 | pass |
| pubmed | 256 | 0.01 | PASS | 0.3719 | 0.0068 | -0.0052 | 4 | True | 0.0000 | 256.0 | pass |
| pubmed | 256 | 0.02 | FAIL | 0.3150 | 0.0049 | -0.0005 | 3 | True | 0.0000 | 256.0 | not enough positive seeds |
| pubmed | 256 | 0.05 | PASS | 1.2955 | 0.0072 | 0.0002 | 4 | True | 0.0000 | 256.0 | pass |
| pubmed | 512 | 0.001 | FAIL | 0.0419 | 0.0003 | 0.0156 | 3 | True | 0.0000 | 375.4 | not enough positive seeds |
| pubmed | 512 | 0.005 | FAIL | 0.2823 | 0.0015 | 0.0113 | 3 | True | 0.0000 | 375.2 | not enough positive seeds |
| pubmed | 512 | 0.01 | PASS | 1.8385 | 0.0015 | 0.0141 | 4 | True | 0.0000 | 376.2 | pass |
| pubmed | 512 | 0.02 | FAIL | 0.4041 | 0.0019 | 0.0129 | 3 | True | 0.0000 | 375.6 | not enough positive seeds |
| pubmed | 512 | 0.05 | FAIL | -1.2057 | 0.0012 | 0.0080 | 2 | True | 0.0000 | 377.2 | no diagnostic gain; not enough positive seeds |

## Aggregate snapshot

- amazoncom / frozengrace_gb_gcn_softdist_xh / max_balls=64 / split=0.001: n=5, balls=64.0, clean_mass=-209.0473, cdr=0.2459, ncr=0.7157
- amazoncom / frozengrace_gb_gcn_softdist_xh / max_balls=64 / split=0.005: n=5, balls=64.0, clean_mass=-209.0473, cdr=0.2459, ncr=0.7157
- amazoncom / frozengrace_gb_gcn_softdist_xh / max_balls=64 / split=0.01: n=5, balls=64.0, clean_mass=-208.9332, cdr=0.2458, ncr=0.7166
- amazoncom / frozengrace_gb_gcn_softdist_xh / max_balls=64 / split=0.02: n=5, balls=64.0, clean_mass=-209.0473, cdr=0.2459, ncr=0.7157
- amazoncom / frozengrace_gb_gcn_softdist_xh / max_balls=64 / split=0.05: n=5, balls=64.0, clean_mass=-209.0473, cdr=0.2459, ncr=0.7157
- amazoncom / frozengrace_gb_gcn_softdist_xh / max_balls=128 / split=0.001: n=5, balls=128.0, clean_mass=-184.3926, cdr=0.2202, ncr=0.7226
- amazoncom / frozengrace_gb_gcn_softdist_xh / max_balls=128 / split=0.005: n=5, balls=128.0, clean_mass=-184.3926, cdr=0.2202, ncr=0.7226
- amazoncom / frozengrace_gb_gcn_softdist_xh / max_balls=128 / split=0.01: n=5, balls=128.0, clean_mass=-184.3926, cdr=0.2202, ncr=0.7226
- amazoncom / frozengrace_gb_gcn_softdist_xh / max_balls=128 / split=0.02: n=5, balls=128.0, clean_mass=-184.3926, cdr=0.2202, ncr=0.7226
- amazoncom / frozengrace_gb_gcn_softdist_xh / max_balls=128 / split=0.05: n=5, balls=128.0, clean_mass=-184.3926, cdr=0.2202, ncr=0.7226
- amazoncom / frozengrace_gb_gcn_softdist_xh / max_balls=256 / split=0.001: n=5, balls=234.6, clean_mass=-168.9713, cdr=0.2108, ncr=0.7055
- amazoncom / frozengrace_gb_gcn_softdist_xh / max_balls=256 / split=0.005: n=5, balls=234.6, clean_mass=-168.9713, cdr=0.2108, ncr=0.7055
- amazoncom / frozengrace_gb_gcn_softdist_xh / max_balls=256 / split=0.01: n=5, balls=234.6, clean_mass=-168.9713, cdr=0.2108, ncr=0.7055
- amazoncom / frozengrace_gb_gcn_softdist_xh / max_balls=256 / split=0.02: n=5, balls=232.6, clean_mass=-168.7312, cdr=0.2102, ncr=0.7072
- amazoncom / frozengrace_gb_gcn_softdist_xh / max_balls=256 / split=0.05: n=5, balls=234.6, clean_mass=-168.9713, cdr=0.2108, ncr=0.7055
- amazoncom / frozengrace_gb_gcn_softdist_xh / max_balls=512 / split=0.001: n=5, balls=234.6, clean_mass=-168.9713, cdr=0.2108, ncr=0.7055
- amazoncom / frozengrace_gb_gcn_softdist_xh / max_balls=512 / split=0.005: n=5, balls=234.6, clean_mass=-168.9713, cdr=0.2108, ncr=0.7055
- amazoncom / frozengrace_gb_gcn_softdist_xh / max_balls=512 / split=0.01: n=5, balls=234.6, clean_mass=-168.9713, cdr=0.2108, ncr=0.7055
- amazoncom / frozengrace_gb_gcn_softdist_xh / max_balls=512 / split=0.02: n=5, balls=234.6, clean_mass=-168.9713, cdr=0.2108, ncr=0.7055
- amazoncom / frozengrace_gb_gcn_softdist_xh / max_balls=512 / split=0.05: n=5, balls=234.6, clean_mass=-168.9713, cdr=0.2108, ncr=0.7055
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=64 / split=0.001: n=5, balls=64.0, clean_mass=-198.1329, cdr=0.2466, ncr=0.7164
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=64 / split=0.005: n=5, balls=64.0, clean_mass=-198.1329, cdr=0.2466, ncr=0.7164
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=64 / split=0.01: n=5, balls=64.0, clean_mass=-198.1756, cdr=0.2465, ncr=0.7174
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=64 / split=0.02: n=5, balls=64.0, clean_mass=-198.1329, cdr=0.2466, ncr=0.7164
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=64 / split=0.05: n=5, balls=64.0, clean_mass=-198.1329, cdr=0.2466, ncr=0.7164
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=128 / split=0.001: n=5, balls=128.0, clean_mass=-177.6731, cdr=0.2220, ncr=0.7235
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=128 / split=0.005: n=5, balls=128.0, clean_mass=-177.6731, cdr=0.2220, ncr=0.7235
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=128 / split=0.01: n=5, balls=128.0, clean_mass=-177.6731, cdr=0.2220, ncr=0.7235
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=128 / split=0.02: n=5, balls=128.0, clean_mass=-177.6731, cdr=0.2220, ncr=0.7235
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=128 / split=0.05: n=5, balls=128.0, clean_mass=-177.6731, cdr=0.2220, ncr=0.7235
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=256 / split=0.001: n=5, balls=234.6, clean_mass=-165.9901, cdr=0.2141, ncr=0.7030
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=256 / split=0.005: n=5, balls=234.6, clean_mass=-165.9901, cdr=0.2141, ncr=0.7030
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=256 / split=0.01: n=5, balls=234.6, clean_mass=-165.9901, cdr=0.2141, ncr=0.7030
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=256 / split=0.02: n=5, balls=232.6, clean_mass=-165.6935, cdr=0.2136, ncr=0.7048
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=256 / split=0.05: n=5, balls=234.6, clean_mass=-165.9901, cdr=0.2141, ncr=0.7030
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=512 / split=0.001: n=5, balls=234.6, clean_mass=-165.9901, cdr=0.2141, ncr=0.7030
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=512 / split=0.005: n=5, balls=234.6, clean_mass=-165.9901, cdr=0.2141, ncr=0.7030
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=512 / split=0.01: n=5, balls=234.6, clean_mass=-165.9901, cdr=0.2141, ncr=0.7030
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=512 / split=0.02: n=5, balls=234.6, clean_mass=-165.9901, cdr=0.2141, ncr=0.7030
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=512 / split=0.05: n=5, balls=234.6, clean_mass=-165.9901, cdr=0.2141, ncr=0.7030
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=64 / split=0.001: n=5, balls=64.0, clean_mass=-224.6498, cdr=0.6298, ncr=0.3326
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=64 / split=0.005: n=5, balls=64.0, clean_mass=-224.6498, cdr=0.6298, ncr=0.3326
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=64 / split=0.01: n=5, balls=64.0, clean_mass=-225.6342, cdr=0.6303, ncr=0.3267
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=64 / split=0.02: n=5, balls=64.0, clean_mass=-224.6498, cdr=0.6298, ncr=0.3326
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=64 / split=0.05: n=5, balls=64.0, clean_mass=-224.6498, cdr=0.6298, ncr=0.3326
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=128 / split=0.001: n=5, balls=128.0, clean_mass=-237.3074, cdr=0.6184, ncr=0.2971
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=128 / split=0.005: n=5, balls=128.0, clean_mass=-237.3074, cdr=0.6184, ncr=0.2971
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=128 / split=0.01: n=5, balls=128.0, clean_mass=-237.3074, cdr=0.6184, ncr=0.2971
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=128 / split=0.02: n=5, balls=128.0, clean_mass=-237.3074, cdr=0.6184, ncr=0.2971
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=128 / split=0.05: n=5, balls=128.0, clean_mass=-237.3074, cdr=0.6184, ncr=0.2971
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=256 / split=0.001: n=5, balls=234.6, clean_mass=-262.8901, cdr=0.6085, ncr=0.2320
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=256 / split=0.005: n=5, balls=234.6, clean_mass=-262.8901, cdr=0.6085, ncr=0.2320
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=256 / split=0.01: n=5, balls=234.6, clean_mass=-262.8901, cdr=0.6085, ncr=0.2320
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=256 / split=0.02: n=5, balls=232.6, clean_mass=-263.9687, cdr=0.6091, ncr=0.2312
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=256 / split=0.05: n=5, balls=234.6, clean_mass=-262.8901, cdr=0.6085, ncr=0.2320
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=512 / split=0.001: n=5, balls=234.6, clean_mass=-262.8901, cdr=0.6085, ncr=0.2320
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=512 / split=0.005: n=5, balls=234.6, clean_mass=-262.8901, cdr=0.6085, ncr=0.2320
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=512 / split=0.01: n=5, balls=234.6, clean_mass=-262.8901, cdr=0.6085, ncr=0.2320
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=512 / split=0.02: n=5, balls=234.6, clean_mass=-262.8901, cdr=0.6085, ncr=0.2320
- amazoncom / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=512 / split=0.05: n=5, balls=234.6, clean_mass=-262.8901, cdr=0.6085, ncr=0.2320
- amazoncom / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=64 / split=0.001: n=5, balls=64.0, clean_mass=-220.9183, cdr=0.6270, ncr=0.3367
- amazoncom / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=64 / split=0.005: n=5, balls=64.0, clean_mass=-220.9183, cdr=0.6270, ncr=0.3367
- amazoncom / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=64 / split=0.01: n=5, balls=64.0, clean_mass=-221.3970, cdr=0.6270, ncr=0.3323
- amazoncom / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=64 / split=0.02: n=5, balls=64.0, clean_mass=-220.9183, cdr=0.6270, ncr=0.3367
- amazoncom / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=64 / split=0.05: n=5, balls=64.0, clean_mass=-220.9183, cdr=0.6270, ncr=0.3367
- amazoncom / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=128 / split=0.001: n=5, balls=128.0, clean_mass=-232.6442, cdr=0.6187, ncr=0.2950
- amazoncom / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=128 / split=0.005: n=5, balls=128.0, clean_mass=-232.6442, cdr=0.6187, ncr=0.2950
- amazoncom / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=128 / split=0.01: n=5, balls=128.0, clean_mass=-232.6442, cdr=0.6187, ncr=0.2950
- amazoncom / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=128 / split=0.02: n=5, balls=128.0, clean_mass=-232.6442, cdr=0.6187, ncr=0.2950
- amazoncom / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=128 / split=0.05: n=5, balls=128.0, clean_mass=-232.6442, cdr=0.6187, ncr=0.2950
- amazoncom / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=256 / split=0.001: n=5, balls=234.6, clean_mass=-254.6452, cdr=0.6063, ncr=0.2307
- amazoncom / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=256 / split=0.005: n=5, balls=234.6, clean_mass=-254.6452, cdr=0.6063, ncr=0.2307
- amazoncom / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=256 / split=0.01: n=5, balls=234.6, clean_mass=-254.6452, cdr=0.6063, ncr=0.2307
- amazoncom / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=256 / split=0.02: n=5, balls=232.6, clean_mass=-255.6357, cdr=0.6057, ncr=0.2283
- amazoncom / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=256 / split=0.05: n=5, balls=234.6, clean_mass=-254.6452, cdr=0.6063, ncr=0.2307
- amazoncom / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=512 / split=0.001: n=5, balls=234.6, clean_mass=-254.6452, cdr=0.6063, ncr=0.2307
- amazoncom / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=512 / split=0.005: n=5, balls=234.6, clean_mass=-254.6452, cdr=0.6063, ncr=0.2307
- amazoncom / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=512 / split=0.01: n=5, balls=234.6, clean_mass=-254.6452, cdr=0.6063, ncr=0.2307
- amazoncom / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=512 / split=0.02: n=5, balls=234.6, clean_mass=-254.6452, cdr=0.6063, ncr=0.2307
- amazoncom / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=512 / split=0.05: n=5, balls=234.6, clean_mass=-254.6452, cdr=0.6063, ncr=0.2307
- amazonpho / frozengrace_gb_gcn_softdist_xh / max_balls=64 / split=0.001: n=5, balls=64.0, clean_mass=-69.1022, cdr=0.1460, ncr=0.8357
- amazonpho / frozengrace_gb_gcn_softdist_xh / max_balls=64 / split=0.005: n=5, balls=64.0, clean_mass=-69.0862, cdr=0.1469, ncr=0.8341
- amazonpho / frozengrace_gb_gcn_softdist_xh / max_balls=64 / split=0.01: n=5, balls=64.0, clean_mass=-69.0862, cdr=0.1469, ncr=0.8341
- amazonpho / frozengrace_gb_gcn_softdist_xh / max_balls=64 / split=0.02: n=5, balls=64.0, clean_mass=-69.0862, cdr=0.1469, ncr=0.8341
- amazonpho / frozengrace_gb_gcn_softdist_xh / max_balls=64 / split=0.05: n=5, balls=64.0, clean_mass=-69.4973, cdr=0.1478, ncr=0.8338
- amazonpho / frozengrace_gb_gcn_softdist_xh / max_balls=128 / split=0.001: n=5, balls=126.2, clean_mass=-61.1985, cdr=0.1448, ncr=0.8055
- amazonpho / frozengrace_gb_gcn_softdist_xh / max_balls=128 / split=0.005: n=5, balls=127.2, clean_mass=-61.0808, cdr=0.1459, ncr=0.8036
- amazonpho / frozengrace_gb_gcn_softdist_xh / max_balls=128 / split=0.01: n=5, balls=127.2, clean_mass=-61.0808, cdr=0.1459, ncr=0.8036
- amazonpho / frozengrace_gb_gcn_softdist_xh / max_balls=128 / split=0.02: n=5, balls=126.2, clean_mass=-61.5758, cdr=0.1466, ncr=0.8014
- amazonpho / frozengrace_gb_gcn_softdist_xh / max_balls=128 / split=0.05: n=5, balls=126.2, clean_mass=-61.0702, cdr=0.1451, ncr=0.8055
- amazonpho / frozengrace_gb_gcn_softdist_xh / max_balls=256 / split=0.001: n=5, balls=132.4, clean_mass=-61.0452, cdr=0.1468, ncr=0.8004
- amazonpho / frozengrace_gb_gcn_softdist_xh / max_balls=256 / split=0.005: n=5, balls=131.4, clean_mass=-60.9836, cdr=0.1462, ncr=0.8020
- amazonpho / frozengrace_gb_gcn_softdist_xh / max_balls=256 / split=0.01: n=5, balls=131.4, clean_mass=-60.9836, cdr=0.1462, ncr=0.8020
- amazonpho / frozengrace_gb_gcn_softdist_xh / max_balls=256 / split=0.02: n=5, balls=131.4, clean_mass=-60.9836, cdr=0.1462, ncr=0.8020
- amazonpho / frozengrace_gb_gcn_softdist_xh / max_balls=256 / split=0.05: n=5, balls=131.4, clean_mass=-60.9836, cdr=0.1462, ncr=0.8020
- amazonpho / frozengrace_gb_gcn_softdist_xh / max_balls=512 / split=0.001: n=5, balls=131.4, clean_mass=-60.9836, cdr=0.1462, ncr=0.8020
- amazonpho / frozengrace_gb_gcn_softdist_xh / max_balls=512 / split=0.005: n=5, balls=130.8, clean_mass=-61.1118, cdr=0.1459, ncr=0.8020
- amazonpho / frozengrace_gb_gcn_softdist_xh / max_balls=512 / split=0.01: n=5, balls=131.4, clean_mass=-60.9836, cdr=0.1462, ncr=0.8020
- amazonpho / frozengrace_gb_gcn_softdist_xh / max_balls=512 / split=0.02: n=5, balls=131.4, clean_mass=-60.9836, cdr=0.1462, ncr=0.8020
- amazonpho / frozengrace_gb_gcn_softdist_xh / max_balls=512 / split=0.05: n=5, balls=130.4, clean_mass=-61.2200, cdr=0.1457, ncr=0.8033
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=64 / split=0.001: n=5, balls=64.0, clean_mass=-61.3318, cdr=0.1442, ncr=0.8357
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=64 / split=0.005: n=5, balls=64.0, clean_mass=-61.2622, cdr=0.1451, ncr=0.8341
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=64 / split=0.01: n=5, balls=64.0, clean_mass=-61.2622, cdr=0.1451, ncr=0.8341
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=64 / split=0.02: n=5, balls=64.0, clean_mass=-61.2622, cdr=0.1451, ncr=0.8341
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=64 / split=0.05: n=5, balls=64.0, clean_mass=-61.5807, cdr=0.1460, ncr=0.8338
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=128 / split=0.001: n=5, balls=126.2, clean_mass=-56.9076, cdr=0.1430, ncr=0.8046
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=128 / split=0.005: n=5, balls=127.2, clean_mass=-56.9368, cdr=0.1437, ncr=0.8040
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=128 / split=0.01: n=5, balls=127.2, clean_mass=-56.9368, cdr=0.1437, ncr=0.8040
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=128 / split=0.02: n=5, balls=126.2, clean_mass=-57.6349, cdr=0.1446, ncr=0.8024
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=128 / split=0.05: n=5, balls=126.2, clean_mass=-56.7719, cdr=0.1432, ncr=0.8043
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=256 / split=0.001: n=5, balls=132.4, clean_mass=-56.8246, cdr=0.1441, ncr=0.8018
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=256 / split=0.005: n=5, balls=131.4, clean_mass=-56.9037, cdr=0.1437, ncr=0.8034
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=256 / split=0.01: n=5, balls=131.4, clean_mass=-56.9037, cdr=0.1437, ncr=0.8034
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=256 / split=0.02: n=5, balls=131.4, clean_mass=-56.9037, cdr=0.1437, ncr=0.8034
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=256 / split=0.05: n=5, balls=131.4, clean_mass=-56.9037, cdr=0.1437, ncr=0.8034
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=512 / split=0.001: n=5, balls=131.4, clean_mass=-56.9037, cdr=0.1437, ncr=0.8034
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=512 / split=0.005: n=5, balls=130.8, clean_mass=-57.0139, cdr=0.1434, ncr=0.8037
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=512 / split=0.01: n=5, balls=131.4, clean_mass=-56.9037, cdr=0.1437, ncr=0.8034
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=512 / split=0.02: n=5, balls=131.4, clean_mass=-56.9037, cdr=0.1437, ncr=0.8034
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=512 / split=0.05: n=5, balls=130.4, clean_mass=-57.2514, cdr=0.1424, ncr=0.8070
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=64 / split=0.001: n=5, balls=64.0, clean_mass=-94.9942, cdr=0.7139, ncr=0.1886
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=64 / split=0.005: n=5, balls=64.0, clean_mass=-94.5150, cdr=0.7130, ncr=0.1863
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=64 / split=0.01: n=5, balls=64.0, clean_mass=-94.5150, cdr=0.7130, ncr=0.1863
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=64 / split=0.02: n=5, balls=64.0, clean_mass=-94.5150, cdr=0.7130, ncr=0.1863
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=64 / split=0.05: n=5, balls=64.0, clean_mass=-95.4508, cdr=0.7144, ncr=0.1835
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=128 / split=0.001: n=5, balls=126.2, clean_mass=-127.3703, cdr=0.6710, ncr=0.1476
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=128 / split=0.005: n=5, balls=127.2, clean_mass=-127.3884, cdr=0.6707, ncr=0.1544
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=128 / split=0.01: n=5, balls=127.2, clean_mass=-127.3884, cdr=0.6707, ncr=0.1544
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=128 / split=0.02: n=5, balls=126.2, clean_mass=-126.4069, cdr=0.6712, ncr=0.1553
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=128 / split=0.05: n=5, balls=126.2, clean_mass=-126.6040, cdr=0.6720, ncr=0.1528
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=256 / split=0.001: n=5, balls=132.4, clean_mass=-129.3358, cdr=0.6678, ncr=0.1574
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=256 / split=0.005: n=5, balls=131.4, clean_mass=-129.0866, cdr=0.6689, ncr=0.1590
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=256 / split=0.01: n=5, balls=131.4, clean_mass=-129.0866, cdr=0.6689, ncr=0.1590
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=256 / split=0.02: n=5, balls=131.4, clean_mass=-129.0866, cdr=0.6689, ncr=0.1590
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=256 / split=0.05: n=5, balls=131.4, clean_mass=-129.0866, cdr=0.6689, ncr=0.1590
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=512 / split=0.001: n=5, balls=131.4, clean_mass=-129.0866, cdr=0.6689, ncr=0.1590
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=512 / split=0.005: n=5, balls=130.8, clean_mass=-129.8702, cdr=0.6668, ncr=0.1566
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=512 / split=0.01: n=5, balls=131.4, clean_mass=-129.0866, cdr=0.6689, ncr=0.1590
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=512 / split=0.02: n=5, balls=131.4, clean_mass=-129.0866, cdr=0.6689, ncr=0.1590
- amazonpho / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=512 / split=0.05: n=5, balls=130.4, clean_mass=-129.4504, cdr=0.6695, ncr=0.1593
- amazonpho / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=64 / split=0.001: n=5, balls=64.0, clean_mass=-92.7489, cdr=0.7107, ncr=0.1868
- amazonpho / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=64 / split=0.005: n=5, balls=64.0, clean_mass=-92.7861, cdr=0.7087, ncr=0.1865
- amazonpho / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=64 / split=0.01: n=5, balls=64.0, clean_mass=-92.7861, cdr=0.7087, ncr=0.1865
- amazonpho / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=64 / split=0.02: n=5, balls=64.0, clean_mass=-92.7861, cdr=0.7087, ncr=0.1865
- amazonpho / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=64 / split=0.05: n=5, balls=64.0, clean_mass=-93.1618, cdr=0.7103, ncr=0.1865
- amazonpho / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=128 / split=0.001: n=5, balls=126.2, clean_mass=-121.0841, cdr=0.6679, ncr=0.1506
- amazonpho / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=128 / split=0.005: n=5, balls=127.2, clean_mass=-120.7123, cdr=0.6689, ncr=0.1510
- amazonpho / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=128 / split=0.01: n=5, balls=127.2, clean_mass=-120.7123, cdr=0.6689, ncr=0.1510
- amazonpho / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=128 / split=0.02: n=5, balls=126.2, clean_mass=-119.7716, cdr=0.6730, ncr=0.1473
- amazonpho / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=128 / split=0.05: n=5, balls=126.2, clean_mass=-120.3259, cdr=0.6698, ncr=0.1523
- amazonpho / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=256 / split=0.001: n=5, balls=132.4, clean_mass=-122.8272, cdr=0.6693, ncr=0.1473
- amazonpho / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=256 / split=0.005: n=5, balls=131.4, clean_mass=-122.5749, cdr=0.6693, ncr=0.1499
- amazonpho / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=256 / split=0.01: n=5, balls=131.4, clean_mass=-122.5749, cdr=0.6693, ncr=0.1499
- amazonpho / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=256 / split=0.02: n=5, balls=131.4, clean_mass=-122.5749, cdr=0.6693, ncr=0.1499
- amazonpho / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=256 / split=0.05: n=5, balls=131.4, clean_mass=-122.5749, cdr=0.6693, ncr=0.1499
- amazonpho / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=512 / split=0.001: n=5, balls=131.4, clean_mass=-122.5749, cdr=0.6693, ncr=0.1499
- amazonpho / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=512 / split=0.005: n=5, balls=130.8, clean_mass=-123.5270, cdr=0.6673, ncr=0.1513
- amazonpho / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=512 / split=0.01: n=5, balls=131.4, clean_mass=-122.5749, cdr=0.6693, ncr=0.1499
- amazonpho / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=512 / split=0.02: n=5, balls=131.4, clean_mass=-122.5749, cdr=0.6693, ncr=0.1499
- amazonpho / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=512 / split=0.05: n=5, balls=130.4, clean_mass=-123.0543, cdr=0.6660, ncr=0.1522
- dblp / frozengrace_gb_gcn_softdist_xh / max_balls=64 / split=0.001: n=5, balls=64.0, clean_mass=-111.4517, cdr=0.1783, ncr=0.8101
- dblp / frozengrace_gb_gcn_softdist_xh / max_balls=64 / split=0.005: n=5, balls=64.0, clean_mass=-111.4517, cdr=0.1783, ncr=0.8101
- dblp / frozengrace_gb_gcn_softdist_xh / max_balls=64 / split=0.01: n=5, balls=64.0, clean_mass=-111.4517, cdr=0.1783, ncr=0.8101
- dblp / frozengrace_gb_gcn_softdist_xh / max_balls=64 / split=0.02: n=5, balls=64.0, clean_mass=-111.4517, cdr=0.1783, ncr=0.8101
- dblp / frozengrace_gb_gcn_softdist_xh / max_balls=64 / split=0.05: n=5, balls=1.0, clean_mass=-133.0269, cdr=0.5539, ncr=0.4434
- dblp / frozengrace_gb_gcn_softdist_xh / max_balls=128 / split=0.001: n=5, balls=128.0, clean_mass=-110.7707, cdr=0.1705, ncr=0.7950
- dblp / frozengrace_gb_gcn_softdist_xh / max_balls=128 / split=0.005: n=5, balls=128.0, clean_mass=-110.7707, cdr=0.1705, ncr=0.7950
- dblp / frozengrace_gb_gcn_softdist_xh / max_balls=128 / split=0.01: n=5, balls=128.0, clean_mass=-110.7707, cdr=0.1705, ncr=0.7950
- dblp / frozengrace_gb_gcn_softdist_xh / max_balls=128 / split=0.02: n=5, balls=128.0, clean_mass=-110.7707, cdr=0.1705, ncr=0.7950
- dblp / frozengrace_gb_gcn_softdist_xh / max_balls=128 / split=0.05: n=5, balls=1.0, clean_mass=-133.0269, cdr=0.5539, ncr=0.4434
- dblp / frozengrace_gb_gcn_softdist_xh / max_balls=256 / split=0.001: n=5, balls=256.0, clean_mass=-109.7170, cdr=0.1726, ncr=0.7461
- dblp / frozengrace_gb_gcn_softdist_xh / max_balls=256 / split=0.005: n=5, balls=256.0, clean_mass=-109.7170, cdr=0.1726, ncr=0.7461
- dblp / frozengrace_gb_gcn_softdist_xh / max_balls=256 / split=0.01: n=5, balls=256.0, clean_mass=-110.0491, cdr=0.1735, ncr=0.7413
- dblp / frozengrace_gb_gcn_softdist_xh / max_balls=256 / split=0.02: n=5, balls=256.0, clean_mass=-109.7170, cdr=0.1726, ncr=0.7461
- dblp / frozengrace_gb_gcn_softdist_xh / max_balls=256 / split=0.05: n=5, balls=1.0, clean_mass=-133.0269, cdr=0.5539, ncr=0.4434
- dblp / frozengrace_gb_gcn_softdist_xh / max_balls=512 / split=0.001: n=5, balls=337.2, clean_mass=-110.7296, cdr=0.1751, ncr=0.7138
- dblp / frozengrace_gb_gcn_softdist_xh / max_balls=512 / split=0.005: n=5, balls=338.4, clean_mass=-110.9470, cdr=0.1744, ncr=0.7142
- dblp / frozengrace_gb_gcn_softdist_xh / max_balls=512 / split=0.01: n=5, balls=337.6, clean_mass=-111.1760, cdr=0.1760, ncr=0.7096
- dblp / frozengrace_gb_gcn_softdist_xh / max_balls=512 / split=0.02: n=5, balls=337.6, clean_mass=-111.1760, cdr=0.1760, ncr=0.7096
- dblp / frozengrace_gb_gcn_softdist_xh / max_balls=512 / split=0.05: n=5, balls=1.0, clean_mass=-133.0269, cdr=0.5539, ncr=0.4434
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=64 / split=0.001: n=5, balls=64.0, clean_mass=-108.6516, cdr=0.1819, ncr=0.8017
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=64 / split=0.005: n=5, balls=64.0, clean_mass=-108.6516, cdr=0.1819, ncr=0.8017
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=64 / split=0.01: n=5, balls=64.0, clean_mass=-108.6516, cdr=0.1819, ncr=0.8017
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=64 / split=0.02: n=5, balls=64.0, clean_mass=-108.6516, cdr=0.1819, ncr=0.8017
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=64 / split=0.05: n=5, balls=1.0, clean_mass=-155.2923, cdr=0.5539, ncr=0.4434
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=128 / split=0.001: n=5, balls=128.0, clean_mass=-107.8084, cdr=0.1749, ncr=0.7848
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=128 / split=0.005: n=5, balls=128.0, clean_mass=-107.8084, cdr=0.1749, ncr=0.7848
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=128 / split=0.01: n=5, balls=128.0, clean_mass=-107.8084, cdr=0.1749, ncr=0.7848
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=128 / split=0.02: n=5, balls=128.0, clean_mass=-107.8084, cdr=0.1749, ncr=0.7848
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=128 / split=0.05: n=5, balls=1.0, clean_mass=-155.2923, cdr=0.5539, ncr=0.4434
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=256 / split=0.001: n=5, balls=256.0, clean_mass=-109.3345, cdr=0.1771, ncr=0.7396
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=256 / split=0.005: n=5, balls=256.0, clean_mass=-109.1140, cdr=0.1776, ncr=0.7391
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=256 / split=0.01: n=5, balls=256.0, clean_mass=-109.1140, cdr=0.1776, ncr=0.7391
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=256 / split=0.02: n=5, balls=256.0, clean_mass=-109.3234, cdr=0.1785, ncr=0.7349
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=256 / split=0.05: n=5, balls=1.0, clean_mass=-155.2923, cdr=0.5539, ncr=0.4434
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=512 / split=0.001: n=5, balls=337.2, clean_mass=-110.5850, cdr=0.1841, ncr=0.6983
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=512 / split=0.005: n=5, balls=338.4, clean_mass=-110.7482, cdr=0.1832, ncr=0.6986
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=512 / split=0.01: n=5, balls=338.4, clean_mass=-110.7482, cdr=0.1832, ncr=0.6986
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=512 / split=0.02: n=5, balls=338.8, clean_mass=-111.1575, cdr=0.1844, ncr=0.6952
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=512 / split=0.05: n=5, balls=1.0, clean_mass=-155.2923, cdr=0.5539, ncr=0.4434
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=64 / split=0.001: n=5, balls=64.0, clean_mass=-167.6673, cdr=0.5459, ncr=0.4054
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=64 / split=0.005: n=5, balls=64.0, clean_mass=-167.6716, cdr=0.5464, ncr=0.4047
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=64 / split=0.01: n=5, balls=64.0, clean_mass=-167.6716, cdr=0.5464, ncr=0.4047
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=64 / split=0.02: n=5, balls=64.0, clean_mass=-169.2473, cdr=0.5464, ncr=0.3996
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=64 / split=0.05: n=5, balls=1.0, clean_mass=-155.2923, cdr=0.5539, ncr=0.4434
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=128 / split=0.001: n=5, balls=128.0, clean_mass=-184.2233, cdr=0.5415, ncr=0.3689
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=128 / split=0.005: n=5, balls=128.0, clean_mass=-183.2137, cdr=0.5414, ncr=0.3724
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=128 / split=0.01: n=5, balls=128.0, clean_mass=-183.1285, cdr=0.5398, ncr=0.3751
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=128 / split=0.02: n=5, balls=128.0, clean_mass=-182.4403, cdr=0.5409, ncr=0.3737
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=128 / split=0.05: n=5, balls=1.0, clean_mass=-155.2923, cdr=0.5539, ncr=0.4434
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=256 / split=0.001: n=5, balls=256.0, clean_mass=-218.5912, cdr=0.5192, ncr=0.3243
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=256 / split=0.005: n=5, balls=256.0, clean_mass=-218.5191, cdr=0.5197, ncr=0.3215
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=256 / split=0.01: n=5, balls=256.0, clean_mass=-218.5191, cdr=0.5197, ncr=0.3215
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=256 / split=0.02: n=5, balls=256.0, clean_mass=-218.6731, cdr=0.5204, ncr=0.3242
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=256 / split=0.05: n=5, balls=1.0, clean_mass=-155.2923, cdr=0.5539, ncr=0.4434
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=512 / split=0.001: n=5, balls=338.4, clean_mass=-232.8764, cdr=0.5152, ncr=0.2859
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=512 / split=0.005: n=5, balls=338.4, clean_mass=-232.8764, cdr=0.5152, ncr=0.2859
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=512 / split=0.01: n=5, balls=337.4, clean_mass=-233.2675, cdr=0.5143, ncr=0.2894
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=512 / split=0.02: n=5, balls=337.2, clean_mass=-233.9116, cdr=0.5145, ncr=0.2896
- dblp / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=512 / split=0.05: n=5, balls=1.0, clean_mass=-155.2923, cdr=0.5539, ncr=0.4434
- dblp / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=64 / split=0.001: n=5, balls=64.0, clean_mass=-157.5184, cdr=0.5464, ncr=0.4040
- dblp / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=64 / split=0.005: n=5, balls=64.0, clean_mass=-157.5184, cdr=0.5464, ncr=0.4040
- dblp / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=64 / split=0.01: n=5, balls=64.0, clean_mass=-159.2203, cdr=0.5471, ncr=0.3958
- dblp / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=64 / split=0.02: n=5, balls=64.0, clean_mass=-157.5184, cdr=0.5464, ncr=0.4040
- dblp / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=64 / split=0.05: n=5, balls=1.0, clean_mass=-133.0269, cdr=0.5539, ncr=0.4434
- dblp / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=128 / split=0.001: n=5, balls=128.0, clean_mass=-175.8877, cdr=0.5372, ncr=0.3786
- dblp / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=128 / split=0.005: n=5, balls=128.0, clean_mass=-176.3774, cdr=0.5370, ncr=0.3783
- dblp / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=128 / split=0.01: n=5, balls=128.0, clean_mass=-174.5589, cdr=0.5360, ncr=0.3837
- dblp / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=128 / split=0.02: n=5, balls=128.0, clean_mass=-174.5589, cdr=0.5360, ncr=0.3837
- dblp / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=128 / split=0.05: n=5, balls=1.0, clean_mass=-133.0269, cdr=0.5539, ncr=0.4434
- dblp / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=256 / split=0.001: n=5, balls=256.0, clean_mass=-216.1090, cdr=0.5114, ncr=0.3340
- dblp / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=256 / split=0.005: n=5, balls=256.0, clean_mass=-216.1090, cdr=0.5114, ncr=0.3340
- dblp / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=256 / split=0.01: n=5, balls=256.0, clean_mass=-216.4719, cdr=0.5113, ncr=0.3327
- dblp / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=256 / split=0.02: n=5, balls=256.0, clean_mass=-216.1090, cdr=0.5114, ncr=0.3340
- dblp / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=256 / split=0.05: n=5, balls=1.0, clean_mass=-133.0269, cdr=0.5539, ncr=0.4434
- dblp / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=512 / split=0.001: n=5, balls=337.2, clean_mass=-232.6020, cdr=0.4956, ncr=0.3144
- dblp / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=512 / split=0.005: n=5, balls=337.2, clean_mass=-232.6020, cdr=0.4956, ncr=0.3144
- dblp / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=512 / split=0.01: n=5, balls=337.2, clean_mass=-232.6020, cdr=0.4956, ncr=0.3144
- dblp / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=512 / split=0.02: n=5, balls=338.4, clean_mass=-231.5824, cdr=0.4951, ncr=0.3109
- dblp / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=512 / split=0.05: n=5, balls=1.0, clean_mass=-133.0269, cdr=0.5539, ncr=0.4434
- pubmed / frozengrace_gb_gcn_softdist_xh / max_balls=64 / split=0.001: n=5, balls=64.0, clean_mass=-116.5825, cdr=0.1910, ncr=0.7850
- pubmed / frozengrace_gb_gcn_softdist_xh / max_balls=64 / split=0.005: n=5, balls=64.0, clean_mass=-116.6642, cdr=0.1909, ncr=0.7850
- pubmed / frozengrace_gb_gcn_softdist_xh / max_balls=64 / split=0.01: n=5, balls=64.0, clean_mass=-116.5825, cdr=0.1910, ncr=0.7850
- pubmed / frozengrace_gb_gcn_softdist_xh / max_balls=64 / split=0.02: n=5, balls=64.0, clean_mass=-116.4024, cdr=0.1913, ncr=0.7830
- pubmed / frozengrace_gb_gcn_softdist_xh / max_balls=64 / split=0.05: n=5, balls=64.0, clean_mass=-116.6146, cdr=0.1910, ncr=0.7851
- pubmed / frozengrace_gb_gcn_softdist_xh / max_balls=128 / split=0.001: n=5, balls=128.0, clean_mass=-112.5821, cdr=0.1891, ncr=0.7615
- pubmed / frozengrace_gb_gcn_softdist_xh / max_balls=128 / split=0.005: n=5, balls=128.0, clean_mass=-113.0589, cdr=0.1885, ncr=0.7622
- pubmed / frozengrace_gb_gcn_softdist_xh / max_balls=128 / split=0.01: n=5, balls=128.0, clean_mass=-112.5821, cdr=0.1891, ncr=0.7615
- pubmed / frozengrace_gb_gcn_softdist_xh / max_balls=128 / split=0.02: n=5, balls=128.0, clean_mass=-112.7357, cdr=0.1886, ncr=0.7613
- pubmed / frozengrace_gb_gcn_softdist_xh / max_balls=128 / split=0.05: n=5, balls=128.0, clean_mass=-112.5821, cdr=0.1891, ncr=0.7615
- pubmed / frozengrace_gb_gcn_softdist_xh / max_balls=256 / split=0.001: n=5, balls=256.0, clean_mass=-110.1876, cdr=0.1876, ncr=0.7087
- pubmed / frozengrace_gb_gcn_softdist_xh / max_balls=256 / split=0.005: n=5, balls=256.0, clean_mass=-110.3396, cdr=0.1885, ncr=0.7071
- pubmed / frozengrace_gb_gcn_softdist_xh / max_balls=256 / split=0.01: n=5, balls=256.0, clean_mass=-109.9674, cdr=0.1889, ncr=0.7079
- pubmed / frozengrace_gb_gcn_softdist_xh / max_balls=256 / split=0.02: n=5, balls=256.0, clean_mass=-110.3396, cdr=0.1885, ncr=0.7071
- pubmed / frozengrace_gb_gcn_softdist_xh / max_balls=256 / split=0.05: n=5, balls=256.0, clean_mass=-110.8947, cdr=0.1880, ncr=0.7027
- pubmed / frozengrace_gb_gcn_softdist_xh / max_balls=512 / split=0.001: n=5, balls=376.2, clean_mass=-108.8623, cdr=0.2090, ncr=0.6230
- pubmed / frozengrace_gb_gcn_softdist_xh / max_balls=512 / split=0.005: n=5, balls=376.0, clean_mass=-108.7889, cdr=0.2111, ncr=0.6206
- pubmed / frozengrace_gb_gcn_softdist_xh / max_balls=512 / split=0.01: n=5, balls=377.2, clean_mass=-108.5723, cdr=0.2095, ncr=0.6210
- pubmed / frozengrace_gb_gcn_softdist_xh / max_balls=512 / split=0.02: n=5, balls=377.8, clean_mass=-108.6498, cdr=0.2089, ncr=0.6250
- pubmed / frozengrace_gb_gcn_softdist_xh / max_balls=512 / split=0.05: n=5, balls=376.2, clean_mass=-107.9006, cdr=0.2087, ncr=0.6288
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=64 / split=0.001: n=5, balls=64.0, clean_mass=-115.7998, cdr=0.1912, ncr=0.7847
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=64 / split=0.005: n=5, balls=64.0, clean_mass=-115.7840, cdr=0.1912, ncr=0.7847
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=64 / split=0.01: n=5, balls=64.0, clean_mass=-115.7840, cdr=0.1912, ncr=0.7847
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=64 / split=0.02: n=5, balls=64.0, clean_mass=-115.6345, cdr=0.1903, ncr=0.7861
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=64 / split=0.05: n=5, balls=64.0, clean_mass=-115.8898, cdr=0.1912, ncr=0.7846
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=128 / split=0.001: n=5, balls=128.0, clean_mass=-112.3414, cdr=0.1911, ncr=0.7634
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=128 / split=0.005: n=5, balls=128.0, clean_mass=-112.3161, cdr=0.1922, ncr=0.7614
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=128 / split=0.01: n=5, balls=128.0, clean_mass=-112.3161, cdr=0.1922, ncr=0.7614
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=128 / split=0.02: n=5, balls=128.0, clean_mass=-111.9991, cdr=0.1920, ncr=0.7616
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=128 / split=0.05: n=5, balls=128.0, clean_mass=-111.9991, cdr=0.1920, ncr=0.7616
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=256 / split=0.001: n=5, balls=256.0, clean_mass=-110.1410, cdr=0.1952, ncr=0.7027
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=256 / split=0.005: n=5, balls=256.0, clean_mass=-108.7460, cdr=0.1941, ncr=0.7038
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=256 / split=0.01: n=5, balls=256.0, clean_mass=-109.5955, cdr=0.1957, ncr=0.7027
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=256 / split=0.02: n=5, balls=256.0, clean_mass=-110.0246, cdr=0.1933, ncr=0.7066
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=256 / split=0.05: n=5, balls=256.0, clean_mass=-109.5992, cdr=0.1952, ncr=0.7030
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=512 / split=0.001: n=5, balls=375.4, clean_mass=-108.8204, cdr=0.2094, ncr=0.6386
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=512 / split=0.005: n=5, balls=375.2, clean_mass=-108.5066, cdr=0.2125, ncr=0.6319
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=512 / split=0.01: n=5, balls=376.2, clean_mass=-106.7338, cdr=0.2110, ncr=0.6351
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=512 / split=0.02: n=5, balls=375.6, clean_mass=-108.2457, cdr=0.2109, ncr=0.6379
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid / max_balls=512 / split=0.05: n=5, balls=377.2, clean_mass=-109.1063, cdr=0.2099, ncr=0.6368
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=64 / split=0.001: n=5, balls=64.0, clean_mass=-106.8326, cdr=0.5679, ncr=0.3543
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=64 / split=0.005: n=5, balls=64.0, clean_mass=-106.2897, cdr=0.5695, ncr=0.3539
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=64 / split=0.01: n=5, balls=64.0, clean_mass=-106.5641, cdr=0.5678, ncr=0.3561
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=64 / split=0.02: n=5, balls=64.0, clean_mass=-107.5600, cdr=0.5664, ncr=0.3531
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=64 / split=0.05: n=5, balls=64.0, clean_mass=-106.1131, cdr=0.5679, ncr=0.3552
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=128 / split=0.001: n=5, balls=128.0, clean_mass=-129.5238, cdr=0.5530, ncr=0.3383
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=128 / split=0.005: n=5, balls=128.0, clean_mass=-131.1615, cdr=0.5535, ncr=0.3370
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=128 / split=0.01: n=5, balls=128.0, clean_mass=-133.3696, cdr=0.5516, ncr=0.3365
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=128 / split=0.02: n=5, balls=128.0, clean_mass=-130.7644, cdr=0.5548, ncr=0.3322
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=128 / split=0.05: n=5, balls=128.0, clean_mass=-130.7604, cdr=0.5556, ncr=0.3315
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=256 / split=0.001: n=5, balls=256.0, clean_mass=-173.4547, cdr=0.5173, ncr=0.3097
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=256 / split=0.005: n=5, balls=256.0, clean_mass=-179.0536, cdr=0.5136, ncr=0.3065
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=256 / split=0.01: n=5, balls=256.0, clean_mass=-176.5176, cdr=0.5144, ncr=0.3106
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=256 / split=0.02: n=5, balls=256.0, clean_mass=-173.4547, cdr=0.5173, ncr=0.3097
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=256 / split=0.05: n=5, balls=256.0, clean_mass=-178.5994, cdr=0.5132, ncr=0.3063
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=512 / split=0.001: n=5, balls=375.4, clean_mass=-204.6755, cdr=0.4998, ncr=0.2739
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=512 / split=0.005: n=5, balls=375.4, clean_mass=-205.6522, cdr=0.4972, ncr=0.2807
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=512 / split=0.01: n=5, balls=377.0, clean_mass=-207.3798, cdr=0.4975, ncr=0.2741
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=512 / split=0.02: n=5, balls=377.2, clean_mass=-206.1978, cdr=0.5006, ncr=0.2761
- pubmed / frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled / max_balls=512 / split=0.05: n=5, balls=377.2, clean_mass=-206.1978, cdr=0.5006, ncr=0.2761
- pubmed / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=64 / split=0.001: n=5, balls=64.0, clean_mass=-104.4295, cdr=0.5657, ncr=0.3522
- pubmed / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=64 / split=0.005: n=5, balls=64.0, clean_mass=-102.9708, cdr=0.5674, ncr=0.3529
- pubmed / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=64 / split=0.01: n=5, balls=64.0, clean_mass=-103.8385, cdr=0.5674, ncr=0.3540
- pubmed / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=64 / split=0.02: n=5, balls=64.0, clean_mass=-102.3390, cdr=0.5697, ncr=0.3485
- pubmed / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=64 / split=0.05: n=5, balls=64.0, clean_mass=-103.8385, cdr=0.5674, ncr=0.3540
- pubmed / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=128 / split=0.001: n=5, balls=128.0, clean_mass=-126.7864, cdr=0.5511, ncr=0.3270
- pubmed / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=128 / split=0.005: n=5, balls=128.0, clean_mass=-126.7864, cdr=0.5511, ncr=0.3270
- pubmed / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=128 / split=0.01: n=5, balls=128.0, clean_mass=-131.1220, cdr=0.5478, ncr=0.3238
- pubmed / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=128 / split=0.02: n=5, balls=128.0, clean_mass=-129.4453, cdr=0.5505, ncr=0.3246
- pubmed / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=128 / split=0.05: n=5, balls=128.0, clean_mass=-128.4263, cdr=0.5497, ncr=0.3265
- pubmed / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=256 / split=0.001: n=5, balls=256.0, clean_mass=-174.0217, cdr=0.5128, ncr=0.2936
- pubmed / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=256 / split=0.005: n=5, balls=256.0, clean_mass=-175.6315, cdr=0.5095, ncr=0.2918
- pubmed / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=256 / split=0.01: n=5, balls=256.0, clean_mass=-174.2785, cdr=0.5166, ncr=0.2938
- pubmed / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=256 / split=0.02: n=5, balls=256.0, clean_mass=-175.6315, cdr=0.5095, ncr=0.2918
- pubmed / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=256 / split=0.05: n=5, balls=256.0, clean_mass=-175.6315, cdr=0.5095, ncr=0.2918
- pubmed / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=512 / split=0.001: n=5, balls=376.6, clean_mass=-203.4987, cdr=0.4935, ncr=0.2505
- pubmed / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=512 / split=0.005: n=5, balls=376.0, clean_mass=-206.1937, cdr=0.4862, ncr=0.2567
- pubmed / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=512 / split=0.01: n=5, balls=375.8, clean_mass=-205.2942, cdr=0.4891, ncr=0.2593
- pubmed / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=512 / split=0.02: n=5, balls=377.0, clean_mass=-205.6114, cdr=0.4914, ncr=0.2536
- pubmed / frozengrace_gb_gcn_softdist_xh_shuffled / max_balls=512 / split=0.05: n=5, balls=375.8, clean_mass=-205.0614, cdr=0.4902, ncr=0.2565
