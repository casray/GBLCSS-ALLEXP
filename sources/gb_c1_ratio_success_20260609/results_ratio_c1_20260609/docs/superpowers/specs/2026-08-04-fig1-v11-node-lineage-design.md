# Fig. 1 v11 Node-Lineage Design

## Goal

Create `v11` from the restored `v10` without modifying `v10`. Stage 2's final granular-ball partition is the immutable visual anchor. Earlier panels must show the same colored node population and class composition.

## Scope

- Stage 1 final frozen representation: use the same colored node population as the final Stage 2 partition.
- Stage 2 initialization: reuse that population inside one dashed parent ball.
- Stage 2 recursive bi-partition and further splitting: preserve the same total population and class counts while changing only positions and ball membership.
- Stage 2 final partition: unchanged.
- Ignore white hollow non-voting nodes in Stages 1 and 2.
- Stages 3 and 4: unchanged.

## Invariants

1. `v10` remains byte-for-byte unchanged.
2. The colored-node count and per-class counts are identical in the Stage 1 final representation and all four Stage 2 frames.
3. Every colored node in Stage 2 frames two and three lies inside one displayed dashed ball. Child-ball sizes and node positions are deliberately uneven rather than uniformly divided.
4. No white hollow nodes appear in Stage 1's final representation or Stage 2.
5. Existing typography, palette, panel geometry, arrows, formulas, and final-partition layout are preserved.

## Deliverables

- `figures/fig1_gblcss_overview_drawio_v11.drawio`
- `figures/fig1_gblcss_overview_drawio_v11.png`
- `figures/fig1_gblcss_overview_drawio_v11.svg`
- `figures/fig1_gblcss_overview_drawio_v11.pdf`
