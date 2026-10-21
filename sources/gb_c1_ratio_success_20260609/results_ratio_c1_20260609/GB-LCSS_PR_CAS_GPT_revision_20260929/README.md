# GB-LCSS Pattern Recognition submission sources

This package uses the Elsevier CAS single-column class (`cas-sc`) from template bundle v2.4. The main manuscript uses the CAS `review` option for the review copy; the separate supplementary material remains single-spaced. The two TeX entry files are the supplied revised versions used to generate the included PDFs.

## Contents

- `main/`: main manuscript source, figures, tables, bibliography, CAS class files, and compiled PDF.
- `supplementary/`: supplementary source, tables, CAS class files, and compiled PDF.

## Build

Compile `main/main.tex` from the `main/` directory with a LaTeX engine and BibTeX. Compile `supplementary/supplementary.tex` separately from the `supplementary/` directory. The supplementary file has no external bibliography.
