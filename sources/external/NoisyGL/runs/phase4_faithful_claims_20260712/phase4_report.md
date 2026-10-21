# Phase 4 faithful-claim audit

## Scope

- Full 25-setting ranking excludes historical ANL/CT/FINE and DivideMix-lite proxies.
- Seven identity-preserving baselines are ranked across all settings.
- Faithful ANL-CE, GMM+CT, and FINE reruns are additionally ranked for uniform 0.4 only.
- Every strongest-baseline comparison requires five paired seeds.

Numeric wins: **24/25**; numeric wins with at least 4/5 positive paired seeds: **23/25**.

| Dataset | Noise | Strongest formal baseline | LCSS | Baseline | Delta | Direction |
|---|---|---|---:|---:|---:|---:|
| cora | uniform0.2 | JoCoR | 62.94+-1.60 | 60.12+-2.25 | +2.83pp | 4/5 |
| cora | uniform0.4 | Small-loss | 55.59+-1.50 | 51.95+-1.97 | +3.64pp | 4/5 |
| cora | uniform0.6 | JoCoR | 41.04+-2.37 | 36.92+-2.14 | +4.13pp | 5/5 |
| cora | pair0.4 | Small-loss | 50.29+-2.76 | 46.99+-2.77 | +3.30pp | 5/5 |
| cora | instance0.4 | JoCoR | 55.67+-0.81 | 52.26+-1.01 | +3.41pp | 5/5 |
| citeseer | uniform0.2 | JoCoR | 63.38+-1.20 | 59.36+-1.43 | +4.03pp | 5/5 |
| citeseer | uniform0.4 | JoCoR | 53.83+-2.75 | 49.91+-3.37 | +3.93pp | 5/5 |
| citeseer | uniform0.6 | JoCoR | 40.17+-2.75 | 39.07+-2.47 | +1.10pp | 4/5 |
| citeseer | pair0.4 | JoCoR | 50.09+-2.40 | 48.63+-2.35 | +1.45pp | 5/5 |
| citeseer | instance0.4 | JoCoR | 54.62+-2.25 | 50.86+-2.45 | +3.76pp | 5/5 |
| corafull | uniform0.2 | JoCoR | 43.51+-0.41 | 43.33+-0.70 | +0.17pp | 3/5 |
| corafull | uniform0.4 | JoCoR | 37.33+-0.81 | 35.70+-0.32 | +1.63pp | 5/5 |
| corafull | uniform0.6 | LogitClip | 27.32+-0.75 | 24.62+-0.72 | +2.70pp | 5/5 |
| corafull | pair0.4 | JoCoR | 34.41+-1.30 | 31.45+-0.80 | +2.96pp | 5/5 |
| corafull | instance0.4 | JoCoR | 37.08+-0.69 | 35.17+-0.51 | +1.91pp | 5/5 |
| coauthorcs | uniform0.2 | JoCoR | 91.56+-0.25 | 91.81+-0.25 | -0.25pp | 0/5 |
| coauthorcs | uniform0.4 | JoCoR | 90.62+-0.57 | 89.38+-0.61 | +1.23pp | 5/5 |
| coauthorcs | uniform0.6 | JoCoR | 87.17+-0.62 | 82.58+-0.41 | +4.58pp | 5/5 |
| coauthorcs | pair0.4 | LogitClip | 85.96+-1.41 | 81.03+-0.57 | +4.93pp | 5/5 |
| coauthorcs | instance0.4 | LogitClip | 90.50+-0.49 | 89.20+-0.76 | +1.30pp | 5/5 |
| amazonpho | uniform0.2 | JoCoR | 86.20+-1.46 | 85.60+-1.26 | +0.60pp | 4/5 |
| amazonpho | uniform0.4 | JoCoR | 81.64+-0.93 | 79.57+-0.53 | +2.07pp | 5/5 |
| amazonpho | uniform0.6 | JoCoR | 70.78+-3.19 | 66.48+-4.43 | +4.30pp | 5/5 |
| amazonpho | pair0.4 | LogitClip | 72.51+-4.02 | 68.88+-3.40 | +3.63pp | 4/5 |
| amazonpho | instance0.4 | JoCoR | 79.59+-1.16 | 74.80+-1.79 | +4.79pp | 5/5 |

## Claim decision

The manuscript may report the numeric-win count above only with the explicit formal-baseline scope. Historical proxy rows remain useful as mechanism explorations but must not enter the published-method count.
