# MLP-18 ellipsoid feasibility diagnostic report

Scope: offline diagnostic only. Target construction uses frozen GRACE H, noisy train labels, and train mask. Clean labels are diagnostic-only.

## Selected geometry settings

- amazonpho: mb256_min32_split0.01; geometry_score=0.1897; target_l1=0.0074; fallback=0.0000
- amazonpho: mb256_min32_split0.001; geometry_score=0.1896; target_l1=0.0073; fallback=0.0000
- amazonpho: mb256_min32_split0.05; geometry_score=0.1896; target_l1=0.0073; fallback=0.0000
- amazonpho: mb256_min32_split0.005; geometry_score=0.1892; target_l1=0.0073; fallback=0.0000
- amazonpho: mb256_min32_split0.02; geometry_score=0.1892; target_l1=0.0073; fallback=0.0000
- amazonpho: mb128_min32_split0.001; geometry_score=0.1880; target_l1=0.0073; fallback=0.0000
- amazonpho: mb128_min32_split0.005; geometry_score=0.1880; target_l1=0.0073; fallback=0.0000
- amazonpho: mb128_min32_split0.01; geometry_score=0.1880; target_l1=0.0073; fallback=0.0000
- amazonpho: mb128_min32_split0.02; geometry_score=0.1880; target_l1=0.0073; fallback=0.0000
- amazonpho: mb128_min32_split0.05; geometry_score=0.1880; target_l1=0.0073; fallback=0.0000
- amazonpho: mb256_min64_split0.02; geometry_score=0.1759; target_l1=0.0073; fallback=0.0000
- amazonpho: mb128_min64_split0.001; geometry_score=0.1759; target_l1=0.0073; fallback=0.0000
- amazonpho: mb128_min64_split0.005; geometry_score=0.1759; target_l1=0.0073; fallback=0.0000
- amazonpho: mb256_min64_split0.05; geometry_score=0.1759; target_l1=0.0073; fallback=0.0000
- amazonpho: mb64_min64_split0.02; geometry_score=0.1756; target_l1=0.0072; fallback=0.0000
- amazonpho: mb128_min64_split0.01; geometry_score=0.1756; target_l1=0.0073; fallback=0.0000
- amazonpho: mb64_min64_split0.001; geometry_score=0.1754; target_l1=0.0072; fallback=0.0000
- amazonpho: mb64_min64_split0.01; geometry_score=0.1754; target_l1=0.0072; fallback=0.0000
- amazonpho: mb64_min64_split0.05; geometry_score=0.1754; target_l1=0.0072; fallback=0.0000
- amazonpho: mb128_min64_split0.02; geometry_score=0.1753; target_l1=0.0073; fallback=0.0000
- coauthorcs: mb64_min32_split0.001; geometry_score=0.1277; target_l1=0.0023; fallback=0.0000
- coauthorcs: mb64_min32_split0.005; geometry_score=0.1277; target_l1=0.0023; fallback=0.0000
- coauthorcs: mb64_min32_split0.01; geometry_score=0.1277; target_l1=0.0023; fallback=0.0000
- coauthorcs: mb64_min32_split0.02; geometry_score=0.1277; target_l1=0.0023; fallback=0.0000
- coauthorcs: mb64_min32_split0.05; geometry_score=0.1277; target_l1=0.0023; fallback=0.0000
- coauthorcs: mb64_min64_split0.001; geometry_score=0.1275; target_l1=0.0023; fallback=0.0000
- coauthorcs: mb64_min64_split0.005; geometry_score=0.1275; target_l1=0.0023; fallback=0.0000
- coauthorcs: mb64_min64_split0.01; geometry_score=0.1275; target_l1=0.0023; fallback=0.0000
- coauthorcs: mb64_min64_split0.02; geometry_score=0.1275; target_l1=0.0023; fallback=0.0000
- coauthorcs: mb64_min64_split0.05; geometry_score=0.1275; target_l1=0.0023; fallback=0.0000
- coauthorcs: mb64_min16_split0.001; geometry_score=0.1249; target_l1=0.0024; fallback=0.0000
- coauthorcs: mb64_min16_split0.005; geometry_score=0.1249; target_l1=0.0024; fallback=0.0000
- coauthorcs: mb64_min16_split0.01; geometry_score=0.1249; target_l1=0.0024; fallback=0.0000
- coauthorcs: mb64_min16_split0.02; geometry_score=0.1249; target_l1=0.0024; fallback=0.0000
- coauthorcs: mb64_min16_split0.05; geometry_score=0.1249; target_l1=0.0024; fallback=0.0000
- coauthorcs: mb32_min16_split0.01; geometry_score=0.1127; target_l1=0.0025; fallback=0.0000
- coauthorcs: mb32_min16_split0.02; geometry_score=0.1127; target_l1=0.0025; fallback=0.0000
- coauthorcs: mb32_min16_split0.001; geometry_score=0.1125; target_l1=0.0025; fallback=0.0000
- coauthorcs: mb32_min16_split0.005; geometry_score=0.1125; target_l1=0.0025; fallback=0.0000
- coauthorcs: mb32_min16_split0.05; geometry_score=0.1125; target_l1=0.0025; fallback=0.0000

## Offline candidate gate

Passed candidates: 0 / 0

## Selected downstream candidates

- None. Stop before downstream unless gates are intentionally revised.
