ENTRY
  { author title journal booktitle year volume number pages eprint archivePrefix rawreference }
  {} {}
FUNCTION {begin.bib}
{ "\begin{thebibliography}{99}" write$ newline$
  "\raggedright" write$ newline$
}
FUNCTION {output.entry}
{ newline$
  "\bibitem{" write$ cite$ write$ "}" write$ newline$
  rawreference write$ newline$
}
FUNCTION {article} { output.entry }
FUNCTION {inproceedings} { output.entry }
FUNCTION {misc} { output.entry }
FUNCTION {default.type} { output.entry }
FUNCTION {end.bib} { newline$ "\end{thebibliography}" write$ newline$ }
READ
EXECUTE {begin.bib}
ITERATE {call.type$}
EXECUTE {end.bib}
