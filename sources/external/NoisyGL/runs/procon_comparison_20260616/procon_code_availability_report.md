# Phase 26: ProCon Code Availability Check

Date: 2026-06-16

## Purpose

Phase 26 was planned as a same-protocol comparison with ProCon (IJCAI 2025), because its peer-informed noisy-label mechanism is close to our local semantic-unit motivation. The plan required an official or clearly attributable implementation before attempting protocol alignment.

## Sources Checked

- IJCAI proceedings page: <https://www.ijcai.org/proceedings/2025/623>
- DOI/PDF links exposed from the IJCAI proceedings page.
- Public web queries for:
  - `"Leveraging Peer-Informed Label Consistency" GitHub`
  - `"ProCon" "Peer-Informed Label Consistency" GitHub`
  - `"Peer-Informed Label Consistency" code`
  - `"ProCon" "Label Consistency" graph noisy labels`

## Finding

No official code repository or author-linked implementation was found from the IJCAI page or public search results during this execution. The IJCAI page provides the paper metadata and abstract, but does not expose a code link.

This is an availability finding, not a claim that no code exists anywhere. It only means we do not currently have a trustworthy implementation source suitable for a same-protocol reproduction.

## Mechanism Summary For Related Work

According to the IJCAI abstract, ProCon constructs semantic peers using both feature similarity and graph adjacency, estimates a consistency distribution, uses a Gaussian mixture model to identify clean samples, and refines prototypes iteratively. This makes it a close related-work point for local semantic consistency under label noise, but its protocol and implementation cannot be aligned here without official code.

## Decision

- Do not self-reimplement ProCon as if it were an external baseline.
- Do not include ProCon numbers in the main result table.
- Mention ProCon in related work as a 2025 peer-informed label-consistency method.
- If the paper uses any ProCon result from the original paper, label it clearly as a reference result, not a same-protocol result.

