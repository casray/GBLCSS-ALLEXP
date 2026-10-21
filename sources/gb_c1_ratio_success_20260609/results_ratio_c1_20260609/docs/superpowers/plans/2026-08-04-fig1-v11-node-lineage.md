# Fig. 1 v11 Node-Lineage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Stage 1/2 node lineage from the unchanged Stage 2 final partition while preserving `v10`.

**Architecture:** Parse the existing draw.io XML, identify the colored-node cells in the Stage 2 final frame, and use their style distribution as the canonical population. Generate deterministic positions for the Stage 1 final embedding and Stage 2 frames one through three, then validate node counts and export publication artifacts.

**Tech Stack:** PowerShell, draw.io XML, draw.io Desktop CLI.

---

### Task 1: Inspect and classify existing node cells

**Files:**
- Read: `figures/fig1_gblcss_overview_drawio_v10.drawio`
- Create: `.drawio-tmp/fig1_gblcss_overview_v11/node_inventory.csv`

- [ ] Extract ellipse cells, geometry, styles, and panel membership.
- [ ] Identify the immutable Stage 2 final-frame colored-node population.
- [ ] Record its total and per-class counts.

### Task 2: Generate v11 without touching v10

**Files:**
- Create: `figures/fig1_gblcss_overview_drawio_v11.drawio`
- Create: `.drawio-tmp/fig1_gblcss_overview_v11/build_v11.ps1`

- [ ] Copy `v10` to `v11`.
- [ ] Remove white hollow nodes from Stage 1's final embedding and Stage 2.
- [ ] Rebuild Stage 1's final embedding from the anchor population.
- [ ] Rebuild Stage 2 frames one through three from the same population, using uneven child sizes and deterministic random layouts in frames two and three.
- [ ] Leave Stage 2 frame four and Stages 3/4 unchanged.

### Task 3: Validate invariants

**Files:**
- Create: `.drawio-tmp/fig1_gblcss_overview_v11/validation.json`

- [ ] Verify `v10` hash is unchanged.
- [ ] Verify the five class counts match across the target frames.
- [ ] Verify no hollow nodes remain in Stage 1 final embedding or Stage 2.
- [ ] Parse the generated XML successfully.

### Task 4: Export and visually review

**Files:**
- Create: `figures/fig1_gblcss_overview_drawio_v11.png`
- Create: `figures/fig1_gblcss_overview_drawio_v11.svg`
- Create: `figures/fig1_gblcss_overview_drawio_v11.pdf`

- [ ] Export PNG at 2x scale plus SVG and cropped PDF.
- [ ] Inspect the PNG for clipping, overlaps, inconsistent density, and source mismatch.
- [ ] Apply at most two focused correction rounds if blockers remain.
