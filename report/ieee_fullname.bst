
ENTRY { address author booktitle chapter edition editor howpublished institution journal key month note number organization pages publisher school series title type volume year } {} { label }
INTEGERS { output.state before.all mid.sentence after.sentence after.block }
FUNCTION {init.state.consts} { #0 'before.all := #1 'mid.sentence := #2 'after.sentence := #3 'after.block := }
STRINGS { s t }
FUNCTION {output.nonnull} { 's := output.state mid.sentence = { ", " * write$ } { output.state after.block = { add.period$ write$ newline$ "\newblock " write$ } { output.state before.all = 'skip$ { add.period$ " " * write$ } if$ } if$ mid.sentence 'output.state := } if$ s }
FUNCTION {output} { duplicate$ empty$ 'pop$ 'output.nonnull if$ }
FUNCTION {output.check} { 't := duplicate$ empty$ { pop$ "empty " t * " in " * cite$ * warning$ } 'output.nonnull if$ }
FUNCTION {fin.entry} { add.period$ write$ newline$ }
FUNCTION {new.block} { output.state before.all = 'skip$ { after.block 'output.state := } if$ }
FUNCTION {format.date} { year empty$ { "" } { year } if$ }
FUNCTION {format.names} { 's := #1 'nameptr := s num.names$ 'numnames := numnames 'namesleft := { namesleft #0 > } { s nameptr "{f.~}{vv~}{ll}{, jj}" format.name$ 't := nameptr #1 > { namesleft #1 > { ", " * t * } { numnames #2 > { "," * } 'skip$ if$ t "others" = { " et~al." * } { " and " * t * } if$ } if$ } 't if$ nameptr #1 + 'nameptr := namesleft #1 - 'namesleft := } while$ }
FUNCTION {format.title} { title empty$ { "" } { "``" title "t" change.case$ * ",''" * } if$ }
FUNCTION {format.journal} { journal empty$ { "" } { "{\em " journal * "}" * } if$ }
FUNCTION {format.booktitle} { booktitle empty$ { "" } { "In {\em " booktitle * "}" * } if$ }
FUNCTION {format.vol.num.pages} { volume empty$ { "" } { "vol.~" volume * } if$ number empty$ 'skip$ { "(" number * ")" * * } if$ pages empty$ 'skip$ { ", pp.~" * pages * } if$ }
FUNCTION {article} { output.state before.all := format.names output new.block format.title "title" output.check new.block format.journal output format.vol.num.pages output format.date "year" output.check fin.entry }
FUNCTION {inproceedings} { output.state before.all := format.names output new.block format.title "title" output.check new.block format.booktitle output format.vol.num.pages output format.date "year" output.check fin.entry }
FUNCTION {misc} { output.state before.all := format.names output new.block format.title output new.block format.date output fin.entry }
FUNCTION {software} { output.state before.all := format.names output new.block format.title output new.block format.date output fin.entry }
READ
FUNCTION {sortify} { purify$ "l" change.case$ }
FUNCTION {sort.format.names} { 's := #1 'nameptr := "" s num.names$ 'numnames := numnames 'namesleft := { namesleft #0 > } { nameptr #1 > { "   " * } 'skip$ if$ s nameptr "{ll{ }}{  f{ }}{  jj{ }}{  vv{ }}" format.name$ 't := nameptr numnames = t "others" = and { "et al" * } { t sortify * } if$ nameptr #1 + 'nameptr := namesleft #1 - 'namesleft := } while$ }
FUNCTION {presort} { sort.format.names "    " * title field.or.null sortify * #1 entry.max$ substring$ 'sort.key$ := }
ITERATE {presort}
SORT
STRINGS { longest.label }
INTEGERS { number.label longest.label.width }
FUNCTION {initialize.longest.label} { "" 'longest.label := #1 'number.label := #0 'longest.label.width := }
FUNCTION {longest.label.pass} { number.label int.to.str$ 'label := number.label #1 + 'number.label := label width$ longest.label.width > { label 'longest.label := label width$ 'longest.label.width := } 'skip$ if$ }
EXECUTE {initialize.longest.label}
ITERATE {longest.label.pass}
FUNCTION {begin.bib} { preamble$ empty$ 'skip$ { preamble$ write$ newline$ } if$ "\begin{thebibliography}{"  longest.label  * "}" * write$ newline$ }
EXECUTE {begin.bib}
FUNCTION {write.bibitem} { newline$ "\bibitem{" write$ cite$ write$ "}" write$ newline$ "" }
ITERATE {write.bibitem}
ITERATE {call.type$}
FUNCTION {end.bib} { newline$ "\end{thebibliography}" write$ newline$ }
EXECUTE {end.bib}
