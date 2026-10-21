# Systematic nearest-work search protocol

## 1. Protocol status

- Freeze date: 2026-07-27
- Search cutoff: 2026-07-27
- Target manuscript: `paper_final_zh_v30_information_sciences.md`
- Target journal: *Information Sciences*
- Purpose: close the manuscript's P0 nearest-work verification gap without making an exclusivity or priority claim.
- Amendment rule: any query added after screening begins must be recorded as an amendment with its reason; existing queries and criteria must not be deleted.

## 2. Review question

Which published or publicly available computer-science studies combine, or partially combine, the following three conditions?

1. node labels are noisy or label-noise robustness is evaluated;
2. graph structure or a graph-derived representation is available during training;
3. the deployed node classifier is graph-free, graph-less, or strictly X-only.

The review also asks how the closest studies differ from GB-LCSS in supervision construction, grouping, use of clean labels, teacher resources, and inference inputs.

## 3. Scope

- Date range: 2018-01-01 through 2026-07-27.
- Earlier sources: include only when they define a foundational method used by the manuscript, such as granular-ball classification.
- Languages: English for the reproducible primary search; Chinese sources may be added only as a separately labelled supplementary search.
- Document types:
  - include peer-reviewed journal and conference papers;
  - include a preprint only when no verified published version is found;
  - include benchmark/dataset papers when they define the evaluation protocol;
  - exclude blogs, marketing pages, theses without a peer-reviewed version, and forum posts as scientific evidence.
- Task scope: node classification. Graph classification, link prediction, graph-level noisy labels, and structure-noise-only studies are excluded unless they directly establish a method distinction used in the manuscript.

## 4. Data sources

1. OpenAlex API and web index.
2. Semantic Scholar API and web index.
3. Crossref/DOI metadata.
4. DBLP and official conference proceedings.
5. OpenReview, arXiv, and publisher article pages as original-source locators.
6. General Web Search for discovery and citation chaining; discovery results must be verified against an original or authoritative source before inclusion.

Google Scholar is not treated as a reproducible API source in this protocol because the available interface does not provide stable, auditable result counts. It may be used manually by the author as a later supplementary search.

## 5. Frozen search strings

The following strings are searched as exact conceptual queries; APIs may require URL encoding but not semantic rewriting.

| ID | Search string |
|---|---|
| Q1 | `("label noise" OR "noisy labels") AND ("graph-free" OR "graph-less" OR graphless) AND ("node classification" OR GNN)` |
| Q2 | `("label noise" OR "noisy labels") AND ("GNN-to-MLP" OR "GNN to MLP") AND (distillation OR student)` |
| Q3 | `(robust OR noise) AND ("graph-less neural network" OR "graph-free MLP") AND label` |
| Q4 | `("noisy labels" OR "label noise") AND ("self-distillation" OR "knowledge distillation") AND graph AND MLP` |
| Q5 | `("label noise" OR "noisy label") AND graph AND MLP AND ("inference without graph" OR "without message passing")` |
| Q6 | `("granular ball" OR "granular-ball") AND graph AND ("noisy label" OR robustness)` |
| Q7 | `("local consensus" OR "group-level soft target") AND ("noisy label" OR "label correction") AND graph` |
| Q8 | `("Graph-MLP" OR GSDN OR GLNN OR "graph-less neural networks") AND ("label noise" OR "noisy labels")` |
| Q9 | `("robust node classification" OR "graph label noise") AND (graphless OR "graph-free" OR "X-only")` |
| Q10 | `("noisy graph learning" OR NoisyGL) AND (MLP OR graphless OR "graph-free")` |

## 6. Inclusion criteria

A record is included in the nearest-work matrix when all of the following hold:

1. its existence and bibliographic metadata are verified through DOI, official proceedings, OpenReview, arXiv, or an authoritative index;
2. it addresses node classification or a directly reusable supervision-construction mechanism;
3. it covers at least two of the three review conditions, or is a foundational granular-ball/noisy-label method necessary to delimit the paper's mechanism;
4. the available abstract or full text is sufficient to code the comparison axes without guessing.

## 7. Exclusion criteria

Exclude and record the reason when a work:

- addresses only graph structure noise, feature noise, or adversarial attacks without label-noise evidence;
- requires graph structure, graph-derived embeddings, or message passing at deployed inference and is not needed as a graph-dependent contrast;
- studies graph-level classification rather than node classification without a transferable supervision mechanism;
- is a duplicate preprint/published version;
- cannot be verified against an authoritative source;
- is only a secondary summary, blog, forum post, or generated bibliography entry.

## 8. Two-pass screening and coding

### Pass 1: title and abstract

Code each retrieved record as `INCLUDE_CANDIDATE`, `EXCLUDE`, or `UNCERTAIN`, with a one-line reason.

### Pass 2: original-source verification

For each candidate, verify title, authors, year, venue, DOI or stable identifier, and code:

- noisy node labels used/evaluated;
- graph available during training;
- inference reads adjacency;
- inference reads graph-derived representation;
- deployed classifier is strictly X-only;
- grouping is label-independent;
- group-level soft target is used;
- clean labels are used for training, selection, or only evaluation;
- full-node teacher prediction/distillation is used;
- granular-ball role;
- evidence locator and verification status.

## 9. Claim boundary

Even a zero-hit search does not prove global non-existence. The strongest allowed conclusion is:

> Under the documented search strategy and cutoff date, the identified nearest studies cover overlapping subsets of noisy-label learning, training-time graph use, and graph-free inference; GB-LCSS is positioned by its specific supervision-construction and resource-flow combination.

The following remain prohibited without stronger evidence:

- "This is the first method..."
- "No prior work..."
- "The problem has not been studied..."
- "GB-LCSS is uniquely capable of..."

## 10. Required outputs

1. Search log with source, query ID, date, returned count where available, and retrieval notes.
2. Screening log with inclusion/exclusion reason.
3. Verified nearest-work matrix with authoritative locators.
4. Search limitations and distributional-skew advisory.
5. Manuscript changes, if any, separated from the Phase 2 search and verification artifacts.
