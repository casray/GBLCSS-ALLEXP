# MLP beta=0.5 Main-Table Report

## Material Passport

- Artifact type: experiment result
- Protocol: train-only label noise, explicit percent split, ROCm `cuda:0`
- Backbone: X-only 2-layer MLP for all compared noise-robust strategies
- Main method: SoftDist-MLP beta=0.5
- FINE/DivideMix note: MLP-adapted strategy baselines, not original image-domain reproduction
- Verification status: ANALYZED

## Completeness

- Expected rows: 850
- Observed aggregate cells: 170
- Missing rows: 0

## Main Label-Noise Table

| Dataset | Noise | Rate | CE | Small-loss | Co-teaching | JoCoR | ELR | SCE | FINE | DivideMix-lite | beta=0.5 | Shuffled |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cora | uniform | 0.2 | 54.96+/-2.22 | 59.84+/-1.29 | 58.62+/-1.97 | 60.12+/-2.25 | 52.95+/-2.38 | 53.57+/-2.17 | 54.85+/-1.98 | 55.19+/-2.30 | 62.94+/-1.60 | 53.70+/-2.06 |
| cora | uniform | 0.4 | 44.24+/-2.62 | 51.95+/-1.97 | 46.83+/-1.78 | 51.72+/-1.30 | 44.28+/-1.38 | 43.60+/-1.86 | 43.54+/-1.48 | 44.79+/-2.05 | 55.59+/-1.50 | 43.63+/-1.62 |
| cora | uniform | 0.6 | 32.51+/-3.38 | 36.72+/-1.09 | 33.29+/-3.48 | 36.92+/-2.14 | 32.82+/-3.51 | 32.66+/-3.78 | 31.77+/-3.04 | 32.30+/-3.61 | 41.04+/-2.37 | 31.70+/-2.69 |
| cora | pair | 0.4 | 42.83+/-2.19 | 46.99+/-2.77 | 44.26+/-2.30 | 45.91+/-2.38 | 41.83+/-1.75 | 41.40+/-2.04 | 41.72+/-1.11 | 43.34+/-2.34 | 50.29+/-2.76 | 39.38+/-1.28 |
| cora | instance | 0.4 | 45.33+/-2.22 | 51.82+/-0.94 | 47.35+/-2.24 | 52.26+/-1.01 | 45.93+/-2.26 | 43.29+/-4.35 | 44.76+/-2.74 | 44.75+/-3.03 | 55.67+/-0.81 | 44.03+/-1.75 |
| citeseer | uniform | 0.2 | 55.10+/-1.17 | 59.26+/-1.94 | 57.05+/-0.82 | 59.36+/-1.43 | 53.62+/-1.48 | 52.12+/-3.28 | 55.86+/-1.06 | 55.81+/-1.54 | 63.38+/-1.20 | 54.53+/-1.83 |
| citeseer | uniform | 0.4 | 43.50+/-3.02 | 49.70+/-3.07 | 46.24+/-2.89 | 49.91+/-3.37 | 43.17+/-3.95 | 42.09+/-3.47 | 43.43+/-3.43 | 43.93+/-2.91 | 53.83+/-2.75 | 42.35+/-2.74 |
| citeseer | uniform | 0.6 | 33.35+/-1.36 | 37.73+/-2.86 | 34.19+/-2.92 | 39.07+/-2.47 | 30.80+/-2.70 | 30.59+/-2.83 | 32.23+/-2.69 | 31.99+/-1.94 | 40.17+/-2.75 | 31.28+/-1.31 |
| citeseer | pair | 0.4 | 45.15+/-1.67 | 48.16+/-2.56 | 46.25+/-2.29 | 48.63+/-2.35 | 42.69+/-2.43 | 42.22+/-1.56 | 44.24+/-2.20 | 45.27+/-1.84 | 50.09+/-2.40 | 45.04+/-1.73 |
| citeseer | instance | 0.4 | 44.33+/-2.04 | 50.75+/-3.33 | 46.81+/-1.68 | 50.86+/-2.45 | 41.94+/-3.27 | 40.92+/-4.23 | 44.88+/-2.40 | 44.19+/-2.86 | 54.62+/-2.25 | 42.81+/-2.34 |
| pubmed | uniform | 0.2 | 74.63+/-1.49 | 81.88+/-0.85 | 81.12+/-0.55 | 81.70+/-0.71 | 76.71+/-0.53 | 79.45+/-0.62 | 74.78+/-1.13 | 83.25+/-0.72 | 82.44+/-0.42 | 75.31+/-0.97 |
| pubmed | uniform | 0.4 | 63.77+/-4.40 | 77.30+/-1.29 | 75.77+/-0.99 | 77.84+/-1.10 | 64.03+/-2.35 | 68.37+/-1.27 | 61.55+/-5.80 | 78.28+/-0.54 | 71.90+/-0.88 | 62.72+/-4.49 |
| pubmed | uniform | 0.6 | 45.50+/-2.45 | 54.72+/-5.02 | 54.50+/-6.19 | 58.12+/-5.83 | 45.74+/-2.13 | 45.49+/-3.02 | 44.03+/-3.65 | 48.57+/-4.14 | 47.53+/-2.42 | 44.27+/-2.72 |
| pubmed | pair | 0.4 | 59.48+/-1.38 | 73.19+/-1.70 | 70.41+/-1.57 | 73.93+/-1.47 | 59.96+/-1.85 | 61.08+/-2.08 | 52.53+/-3.22 | 73.42+/-2.25 | 68.49+/-1.27 | 58.10+/-1.19 |
| pubmed | instance | 0.4 | 62.69+/-2.78 | 76.47+/-1.25 | 74.89+/-0.87 | 76.89+/-0.80 | 62.80+/-1.74 | 67.75+/-2.45 | 60.05+/-2.43 | 76.69+/-1.51 | 72.01+/-1.23 | 61.05+/-3.37 |

## Paired Position of beta=0.5

| Table | Dataset | Noise | Rate | Train | Baseline | Delta | Wins |
|---|---|---|---:|---:|---|---:|---:|
| labelrate | citeseer | uniform | 0.4 | 0.05 | MLP+small-loss | +5.80pp | 5/5 |
| labelrate | citeseer | uniform | 0.4 | 0.05 | MLP+JoCoR | +6.11pp | 5/5 |
| labelrate | citeseer | uniform | 0.4 | 0.05 | MLP-FINE | +7.52pp | 5/5 |
| labelrate | citeseer | uniform | 0.4 | 0.05 | SoftDist beta=0.5 shuffled | +8.31pp | 5/5 |
| labelrate | citeseer | uniform | 0.4 | 0.65 | MLP+small-loss | +1.44pp | 4/5 |
| labelrate | citeseer | uniform | 0.4 | 0.65 | MLP+JoCoR | +1.35pp | 4/5 |
| labelrate | citeseer | uniform | 0.4 | 0.65 | MLP-FINE | +5.51pp | 5/5 |
| labelrate | citeseer | uniform | 0.4 | 0.65 | SoftDist beta=0.5 shuffled | +7.32pp | 5/5 |
| labelrate | cora | uniform | 0.4 | 0.05 | MLP+small-loss | +5.61pp | 5/5 |
| labelrate | cora | uniform | 0.4 | 0.05 | MLP+JoCoR | +6.68pp | 5/5 |
| labelrate | cora | uniform | 0.4 | 0.05 | MLP-FINE | +8.61pp | 5/5 |
| labelrate | cora | uniform | 0.4 | 0.05 | SoftDist beta=0.5 shuffled | +6.57pp | 5/5 |
| labelrate | cora | uniform | 0.4 | 0.65 | MLP+small-loss | +1.24pp | 5/5 |
| labelrate | cora | uniform | 0.4 | 0.65 | MLP+JoCoR | +0.18pp | 2/5 |
| labelrate | cora | uniform | 0.4 | 0.65 | MLP-FINE | +12.02pp | 5/5 |
| labelrate | cora | uniform | 0.4 | 0.65 | SoftDist beta=0.5 shuffled | +11.49pp | 5/5 |
| main | citeseer | instance | 0.4 | 0.20 | MLP+CE | +10.29pp | 5/5 |
| main | citeseer | instance | 0.4 | 0.20 | MLP+small-loss | +3.87pp | 5/5 |
| main | citeseer | instance | 0.4 | 0.20 | MLP+co-teaching | +7.81pp | 5/5 |
| main | citeseer | instance | 0.4 | 0.20 | MLP+JoCoR | +3.76pp | 5/5 |
| main | citeseer | instance | 0.4 | 0.20 | MLP+ELR | +12.68pp | 5/5 |
| main | citeseer | instance | 0.4 | 0.20 | MLP+SCE | +13.70pp | 5/5 |
| main | citeseer | instance | 0.4 | 0.20 | MLP-FINE | +9.74pp | 5/5 |
| main | citeseer | instance | 0.4 | 0.20 | DivideMix-lite | +10.43pp | 5/5 |
| main | citeseer | instance | 0.4 | 0.20 | SoftDist beta=0.5 shuffled | +11.81pp | 5/5 |
| main | citeseer | pair | 0.4 | 0.20 | MLP+CE | +4.94pp | 5/5 |
| main | citeseer | pair | 0.4 | 0.20 | MLP+small-loss | +1.92pp | 5/5 |
| main | citeseer | pair | 0.4 | 0.20 | MLP+co-teaching | +3.83pp | 5/5 |
| main | citeseer | pair | 0.4 | 0.20 | MLP+JoCoR | +1.45pp | 5/5 |
| main | citeseer | pair | 0.4 | 0.20 | MLP+ELR | +7.40pp | 5/5 |
| main | citeseer | pair | 0.4 | 0.20 | MLP+SCE | +7.87pp | 5/5 |
| main | citeseer | pair | 0.4 | 0.20 | MLP-FINE | +5.84pp | 5/5 |
| main | citeseer | pair | 0.4 | 0.20 | DivideMix-lite | +4.82pp | 5/5 |
| main | citeseer | pair | 0.4 | 0.20 | SoftDist beta=0.5 shuffled | +5.04pp | 5/5 |
| main | citeseer | uniform | 0.2 | 0.20 | MLP+CE | +8.28pp | 5/5 |
| main | citeseer | uniform | 0.2 | 0.20 | MLP+small-loss | +4.12pp | 5/5 |
| main | citeseer | uniform | 0.2 | 0.20 | MLP+co-teaching | +6.33pp | 5/5 |
| main | citeseer | uniform | 0.2 | 0.20 | MLP+JoCoR | +4.03pp | 5/5 |
| main | citeseer | uniform | 0.2 | 0.20 | MLP+ELR | +9.77pp | 5/5 |
| main | citeseer | uniform | 0.2 | 0.20 | MLP+SCE | +11.26pp | 5/5 |
| main | citeseer | uniform | 0.2 | 0.20 | MLP-FINE | +7.53pp | 5/5 |
| main | citeseer | uniform | 0.2 | 0.20 | DivideMix-lite | +7.58pp | 5/5 |
| main | citeseer | uniform | 0.2 | 0.20 | SoftDist beta=0.5 shuffled | +8.86pp | 5/5 |
| main | citeseer | uniform | 0.4 | 0.20 | MLP+CE | +10.34pp | 5/5 |
| main | citeseer | uniform | 0.4 | 0.20 | MLP+small-loss | +4.13pp | 5/5 |
| main | citeseer | uniform | 0.4 | 0.20 | MLP+co-teaching | +7.59pp | 5/5 |
| main | citeseer | uniform | 0.4 | 0.20 | MLP+JoCoR | +3.93pp | 5/5 |
| main | citeseer | uniform | 0.4 | 0.20 | MLP+ELR | +10.66pp | 5/5 |
| main | citeseer | uniform | 0.4 | 0.20 | MLP+SCE | +11.74pp | 5/5 |
| main | citeseer | uniform | 0.4 | 0.20 | MLP-FINE | +10.40pp | 5/5 |
| main | citeseer | uniform | 0.4 | 0.20 | DivideMix-lite | +9.90pp | 5/5 |
| main | citeseer | uniform | 0.4 | 0.20 | SoftDist beta=0.5 shuffled | +11.48pp | 5/5 |
| main | citeseer | uniform | 0.6 | 0.20 | MLP+CE | +6.82pp | 5/5 |
| main | citeseer | uniform | 0.6 | 0.20 | MLP+small-loss | +2.44pp | 5/5 |
| main | citeseer | uniform | 0.6 | 0.20 | MLP+co-teaching | +5.98pp | 5/5 |
| main | citeseer | uniform | 0.6 | 0.20 | MLP+JoCoR | +1.10pp | 4/5 |
| main | citeseer | uniform | 0.6 | 0.20 | MLP+ELR | +9.37pp | 5/5 |
| main | citeseer | uniform | 0.6 | 0.20 | MLP+SCE | +9.58pp | 5/5 |
| main | citeseer | uniform | 0.6 | 0.20 | MLP-FINE | +7.94pp | 5/5 |
| main | citeseer | uniform | 0.6 | 0.20 | DivideMix-lite | +8.18pp | 5/5 |
| main | citeseer | uniform | 0.6 | 0.20 | SoftDist beta=0.5 shuffled | +8.89pp | 5/5 |
| main | cora | instance | 0.4 | 0.20 | MLP+CE | +10.34pp | 5/5 |
| main | cora | instance | 0.4 | 0.20 | MLP+small-loss | +3.85pp | 5/5 |
| main | cora | instance | 0.4 | 0.20 | MLP+co-teaching | +8.33pp | 5/5 |
| main | cora | instance | 0.4 | 0.20 | MLP+JoCoR | +3.41pp | 5/5 |
| main | cora | instance | 0.4 | 0.20 | MLP+ELR | +9.74pp | 5/5 |
| main | cora | instance | 0.4 | 0.20 | MLP+SCE | +12.38pp | 5/5 |
| main | cora | instance | 0.4 | 0.20 | MLP-FINE | +10.91pp | 5/5 |
| main | cora | instance | 0.4 | 0.20 | DivideMix-lite | +10.92pp | 5/5 |
| main | cora | instance | 0.4 | 0.20 | SoftDist beta=0.5 shuffled | +11.64pp | 5/5 |
| main | cora | pair | 0.4 | 0.20 | MLP+CE | +7.46pp | 5/5 |
| main | cora | pair | 0.4 | 0.20 | MLP+small-loss | +3.30pp | 5/5 |
| main | cora | pair | 0.4 | 0.20 | MLP+co-teaching | +6.03pp | 5/5 |
| main | cora | pair | 0.4 | 0.20 | MLP+JoCoR | +4.38pp | 5/5 |
| main | cora | pair | 0.4 | 0.20 | MLP+ELR | +8.46pp | 5/5 |
| main | cora | pair | 0.4 | 0.20 | MLP+SCE | +8.89pp | 5/5 |
| main | cora | pair | 0.4 | 0.20 | MLP-FINE | +8.57pp | 5/5 |
| main | cora | pair | 0.4 | 0.20 | DivideMix-lite | +6.96pp | 5/5 |
| main | cora | pair | 0.4 | 0.20 | SoftDist beta=0.5 shuffled | +10.91pp | 5/5 |
| main | cora | uniform | 0.2 | 0.20 | MLP+CE | +7.99pp | 5/5 |
| main | cora | uniform | 0.2 | 0.20 | MLP+small-loss | +3.10pp | 5/5 |
| main | cora | uniform | 0.2 | 0.20 | MLP+co-teaching | +4.33pp | 5/5 |
| main | cora | uniform | 0.2 | 0.20 | MLP+JoCoR | +2.83pp | 4/5 |
| main | cora | uniform | 0.2 | 0.20 | MLP+ELR | +9.99pp | 5/5 |
| main | cora | uniform | 0.2 | 0.20 | MLP+SCE | +9.37pp | 5/5 |
| main | cora | uniform | 0.2 | 0.20 | MLP-FINE | +8.09pp | 5/5 |
| main | cora | uniform | 0.2 | 0.20 | DivideMix-lite | +7.76pp | 5/5 |
| main | cora | uniform | 0.2 | 0.20 | SoftDist beta=0.5 shuffled | +9.25pp | 5/5 |
| main | cora | uniform | 0.4 | 0.20 | MLP+CE | +11.35pp | 5/5 |
| main | cora | uniform | 0.4 | 0.20 | MLP+small-loss | +3.64pp | 4/5 |
| main | cora | uniform | 0.4 | 0.20 | MLP+co-teaching | +8.76pp | 5/5 |
| main | cora | uniform | 0.4 | 0.20 | MLP+JoCoR | +3.87pp | 5/5 |
| main | cora | uniform | 0.4 | 0.20 | MLP+ELR | +11.30pp | 5/5 |
| main | cora | uniform | 0.4 | 0.20 | MLP+SCE | +11.99pp | 5/5 |
| main | cora | uniform | 0.4 | 0.20 | MLP-FINE | +12.05pp | 5/5 |
| main | cora | uniform | 0.4 | 0.20 | DivideMix-lite | +10.80pp | 5/5 |
| main | cora | uniform | 0.4 | 0.20 | SoftDist beta=0.5 shuffled | +11.96pp | 5/5 |
| main | cora | uniform | 0.6 | 0.20 | MLP+CE | +8.54pp | 5/5 |
| main | cora | uniform | 0.6 | 0.20 | MLP+small-loss | +4.33pp | 5/5 |
| main | cora | uniform | 0.6 | 0.20 | MLP+co-teaching | +7.76pp | 5/5 |
| main | cora | uniform | 0.6 | 0.20 | MLP+JoCoR | +4.13pp | 5/5 |
| main | cora | uniform | 0.6 | 0.20 | MLP+ELR | +8.22pp | 5/5 |
| main | cora | uniform | 0.6 | 0.20 | MLP+SCE | +8.38pp | 5/5 |
| main | cora | uniform | 0.6 | 0.20 | MLP-FINE | +9.28pp | 5/5 |
| main | cora | uniform | 0.6 | 0.20 | DivideMix-lite | +8.75pp | 5/5 |
| main | cora | uniform | 0.6 | 0.20 | SoftDist beta=0.5 shuffled | +9.34pp | 5/5 |
| main | pubmed | instance | 0.4 | 0.20 | MLP+CE | +9.32pp | 5/5 |
| main | pubmed | instance | 0.4 | 0.20 | MLP+small-loss | -4.46pp | 0/5 |
| main | pubmed | instance | 0.4 | 0.20 | MLP+co-teaching | -2.88pp | 0/5 |
| main | pubmed | instance | 0.4 | 0.20 | MLP+JoCoR | -4.88pp | 0/5 |
| main | pubmed | instance | 0.4 | 0.20 | MLP+ELR | +9.21pp | 5/5 |
| main | pubmed | instance | 0.4 | 0.20 | MLP+SCE | +4.26pp | 4/5 |
| main | pubmed | instance | 0.4 | 0.20 | MLP-FINE | +11.96pp | 5/5 |
| main | pubmed | instance | 0.4 | 0.20 | DivideMix-lite | -4.68pp | 0/5 |
| main | pubmed | instance | 0.4 | 0.20 | SoftDist beta=0.5 shuffled | +10.96pp | 5/5 |
| main | pubmed | pair | 0.4 | 0.20 | MLP+CE | +9.01pp | 5/5 |
| main | pubmed | pair | 0.4 | 0.20 | MLP+small-loss | -4.70pp | 0/5 |
| main | pubmed | pair | 0.4 | 0.20 | MLP+co-teaching | -1.92pp | 0/5 |
| main | pubmed | pair | 0.4 | 0.20 | MLP+JoCoR | -5.44pp | 0/5 |
| main | pubmed | pair | 0.4 | 0.20 | MLP+ELR | +8.53pp | 5/5 |
| main | pubmed | pair | 0.4 | 0.20 | MLP+SCE | +7.41pp | 5/5 |
| main | pubmed | pair | 0.4 | 0.20 | MLP-FINE | +15.96pp | 5/5 |
| main | pubmed | pair | 0.4 | 0.20 | DivideMix-lite | -4.93pp | 0/5 |
| main | pubmed | pair | 0.4 | 0.20 | SoftDist beta=0.5 shuffled | +10.39pp | 5/5 |
| main | pubmed | uniform | 0.2 | 0.20 | MLP+CE | +7.80pp | 5/5 |
| main | pubmed | uniform | 0.2 | 0.20 | MLP+small-loss | +0.55pp | 4/5 |
| main | pubmed | uniform | 0.2 | 0.20 | MLP+co-teaching | +1.32pp | 5/5 |
| main | pubmed | uniform | 0.2 | 0.20 | MLP+JoCoR | +0.74pp | 5/5 |
| main | pubmed | uniform | 0.2 | 0.20 | MLP+ELR | +5.72pp | 5/5 |
| main | pubmed | uniform | 0.2 | 0.20 | MLP+SCE | +2.98pp | 5/5 |
| main | pubmed | uniform | 0.2 | 0.20 | MLP-FINE | +7.66pp | 5/5 |
| main | pubmed | uniform | 0.2 | 0.20 | DivideMix-lite | -0.82pp | 1/5 |
| main | pubmed | uniform | 0.2 | 0.20 | SoftDist beta=0.5 shuffled | +7.12pp | 5/5 |
| main | pubmed | uniform | 0.4 | 0.20 | MLP+CE | +8.13pp | 5/5 |
| main | pubmed | uniform | 0.4 | 0.20 | MLP+small-loss | -5.40pp | 0/5 |
| main | pubmed | uniform | 0.4 | 0.20 | MLP+co-teaching | -3.87pp | 0/5 |
| main | pubmed | uniform | 0.4 | 0.20 | MLP+JoCoR | -5.93pp | 0/5 |
| main | pubmed | uniform | 0.4 | 0.20 | MLP+ELR | +7.87pp | 5/5 |
| main | pubmed | uniform | 0.4 | 0.20 | MLP+SCE | +3.53pp | 5/5 |
| main | pubmed | uniform | 0.4 | 0.20 | MLP-FINE | +10.36pp | 5/5 |
| main | pubmed | uniform | 0.4 | 0.20 | DivideMix-lite | -6.38pp | 0/5 |
| main | pubmed | uniform | 0.4 | 0.20 | SoftDist beta=0.5 shuffled | +9.18pp | 5/5 |
| main | pubmed | uniform | 0.6 | 0.20 | MLP+CE | +2.03pp | 5/5 |
| main | pubmed | uniform | 0.6 | 0.20 | MLP+small-loss | -7.19pp | 0/5 |
| main | pubmed | uniform | 0.6 | 0.20 | MLP+co-teaching | -6.97pp | 0/5 |
| main | pubmed | uniform | 0.6 | 0.20 | MLP+JoCoR | -10.58pp | 0/5 |
| main | pubmed | uniform | 0.6 | 0.20 | MLP+ELR | +1.79pp | 5/5 |
| main | pubmed | uniform | 0.6 | 0.20 | MLP+SCE | +2.04pp | 4/5 |
| main | pubmed | uniform | 0.6 | 0.20 | MLP-FINE | +3.50pp | 5/5 |
| main | pubmed | uniform | 0.6 | 0.20 | DivideMix-lite | -1.04pp | 2/5 |
| main | pubmed | uniform | 0.6 | 0.20 | SoftDist beta=0.5 shuffled | +3.26pp | 5/5 |

## Label-Rate Endpoint Table

| Dataset | Train | Small-loss | JoCoR | FINE | beta=0.5 | Shuffled |
|---|---:|---:|---:|---:|---:|---:|
| cora | 0.05 | 38.09+/-5.45 | 37.02+/-5.39 | 35.09+/-5.11 | 43.70+/-5.81 | 37.13+/-4.58 |
| cora | 0.65 | 62.78+/-2.14 | 63.84+/-2.95 | 51.99+/-4.05 | 64.02+/-2.30 | 52.53+/-3.26 |
| citeseer | 0.05 | 37.54+/-2.96 | 37.23+/-2.82 | 35.82+/-2.93 | 43.34+/-3.51 | 35.03+/-3.24 |
| citeseer | 0.65 | 61.18+/-1.78 | 61.28+/-1.39 | 57.11+/-1.84 | 62.62+/-0.82 | 55.31+/-3.29 |

## Compact Verdict Aids

- beta=0.5 vs MLP+small-loss: 11/15 main settings mean-positive with >=3/5 seeds.
- beta=0.5 vs MLP+co-teaching: 11/15 main settings mean-positive with >=3/5 seeds.
- beta=0.5 vs MLP+JoCoR: 11/15 main settings mean-positive with >=3/5 seeds.
- beta=0.5 vs MLP+ELR: 15/15 main settings mean-positive with >=3/5 seeds.
- beta=0.5 vs MLP+SCE: 15/15 main settings mean-positive with >=3/5 seeds.
- beta=0.5 vs MLP-FINE: 15/15 main settings mean-positive with >=3/5 seeds.
- beta=0.5 vs DivideMix-lite: 10/15 main settings mean-positive with >=3/5 seeds.
- beta=0.5 vs SoftDist beta=0.5 shuffled: 15/15 main settings mean-positive with >=3/5 seeds.
