# GB-LCSS LaTeX sources (CAS single-column template)

This package uses the `cas-sc` class and supporting files copied from Elsevier CAS bundle version 2.4. The manuscript content, tables, figures, and numerical results are carried over from the prior finalized source; only the document class/front matter and non-content layout controls were rebuilt for the supplied template.

## Files

- `main/`: complete manuscript source and figures.
- `supplementary/`: supplementary source and tables.

## Build

Compile `main/main.tex` with a TeX distribution that includes the packages used by the CAS class. BibTeX is required for the main manuscript. Compile `supplementary/supplementary.tex` separately; it has no external bibliography.
