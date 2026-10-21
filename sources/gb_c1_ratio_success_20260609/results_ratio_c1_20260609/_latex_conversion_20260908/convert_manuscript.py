import copy
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / 'GB-LCSS_Elsevier_LaTeX_Template'
DOC = json.loads((HERE / 'source.json').read_text(encoding='utf-8'))
BLOCKS = DOC['blocks']
AUDIT = {'tables': [], 'equations': [], 'figures': [], 'references': [], 'blocks': []}


def plain(x):
    if isinstance(x, list):
        return ''.join(plain(v) for v in x)
    if not isinstance(x, dict):
        return ''
    t, c = x['t'], x.get('c')
    if t == 'Str':
        return c
    if t in ('Space', 'SoftBreak', 'LineBreak'):
        return ' '
    if t in ('Para', 'Plain', 'Strong', 'Emph', 'SmallCaps', 'Superscript', 'Subscript', 'MetaInlines'):
        return plain(c)
    if t == 'Math':
        return '$' + c[1] + '$'
    if t in ('Span', 'Link'):
        return plain(c[1])
    if t == 'Header':
        return plain(c[2])
    if t == 'Code':
        return c[1]
    return ''


SPECIAL = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
           '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
           '\\': r'\textbackslash{}', '\u200b': '', '\u00a0': '~',
           '\u2013': '--', '\u2014': '---', '\u2212': r'\ensuremath{-}',
           '\u00b1': r'\ensuremath{\pm}', '\u00d7': r'\ensuremath{\times}',
           '\u03c1': r'\ensuremath{\rho}', '\u03b1': r'\ensuremath{\alpha}',
           '\u03b2': r'\ensuremath{\beta}', '\u03c4': r'\ensuremath{\tau}',
           '\u2264': r'\ensuremath{\leq}', '\u2265': r'\ensuremath{\geq}',
           '\u2019': "'", '\u2018': '`', '\u201c': '``', '\u201d': "''",
           '\u00b2': r'\textsuperscript{2}', '\u00b3': r'\textsuperscript{3}'}
ACCENTS = {'\u0301': "'", '\u0300': '`', '\u0308': '"', '\u0302': '^',
           '\u0303': '~', '\u030c': 'v', '\u0327': 'c', '\u030a': 'r'}


def esc(s):
    out = []
    for ch in s:
        if ch in SPECIAL:
            out.append(SPECIAL[ch])
        elif ord(ch) > 127:
            parts = unicodedata.normalize('NFD', ch)
            if len(parts) == 2 and parts[1] in ACCENTS:
                out.append('\\' + ACCENTS[parts[1]] + '{' + parts[0] + '}')
            else:
                raise ValueError('Unhandled Unicode character: ' + repr(ch))
        else:
            out.append(ch)
    return ''.join(out)


def math(s):
    s = s.replace('^{}', '').replace(r'\hspace{0pt}', '')
    operators = {'logsoftmax': r'\log\operatorname{softmax}',
                 'softmax': r'\operatorname{softmax}', 'max': r'\max', 'BS': r'\operatorname{BS}'}
    s = re.sub(r'(?<![A-Za-z\\])(logsoftmax|softmax|max|BS)(?![A-Za-z])',
               lambda m: operators[m[0]], s)
    s = s.replace(r'D_{split}', r'D_{\mathrm{split}}')
    for label in ('LCSS', 'KD', 'tar'):
        s = s.replace('{' + label + '}', r'{\mathrm{' + label + '}}')
    s = s.replace('seed + 1,000,003', r'\text{seed} + 1{,}000{,}003')
    s = s.replace(r'\ mod\ ', r'\bmod ')
    if r'\left\{ \begin{matrix}' in s:
        s = s.replace(r'\left\{ \begin{matrix}', r'\begin{cases}')
        s = s.replace(r'\end{matrix} \right.\ ', r'\end{cases}')
    return s.strip()


def citation_sub(s):
    def cite(m):
        keys = []
        for part in m[1].split(','):
            bounds = re.split(r'--|-', part.strip())
            if len(bounds) == 1:
                nums = [int(bounds[0])]
            else:
                nums = range(int(bounds[0]), int(bounds[1]) + 1)
            keys.extend('ref' + str(n) for n in nums)
        return r'\cite{' + ','.join(keys) + '}'
    return re.sub(r'\[((?:[1-9]\d?)(?:(?:--|-|,\s*)[1-9]\d?)*)\]', cite, s)


def inlines(xs, bold=True, cites=True):
    out = []
    for x in xs:
        t, c = x['t'], x.get('c')
        if t == 'Str':
            out.append(esc(c))
        elif t in ('Space', 'SoftBreak'):
            out.append(' ')
        elif t == 'LineBreak':
            out.append(r'\newline ')
        elif t == 'Math':
            out.append(r'\(' + math(c[1]) + r'\)')
        elif t == 'Strong':
            value = inlines(c, bold, cites)
            out.append(r'\textbf{' + value + '}' if bold else value)
        elif t in ('Emph', 'SmallCaps', 'Superscript', 'Subscript', 'Strikeout'):
            cmd = {'Emph': 'emph', 'SmallCaps': 'textsc', 'Superscript': 'textsuperscript',
                   'Subscript': 'textsubscript', 'Strikeout': 'sout'}[t]
            out.append('\\' + cmd + '{' + inlines(c, bold, cites) + '}')
        elif t == 'Span':
            out.append(inlines(c[1], bold, cites))
        elif t == 'Link':
            url = c[2][0]
            label = inlines(c[1], bold, cites)
            out.append(r'\url{' + url.replace('%', r'\%') + '}' if plain(c[1]) == url
                       else r'\href{' + url.replace('%', r'\%') + '}{' + label + '}')
        elif t == 'Code':
            out.append(r'\texttt{' + esc(c[1]) + '}')
        elif t == 'Quoted':
            out.append('``' + inlines(c[1], bold, cites) + "''")
        else:
            raise ValueError('Unhandled inline type: ' + t)
    value = ''.join(out)
    return citation_sub(value) if cites else value


def blocks_tex(bs, bold=True, cites=True):
    out = []
    for b in bs:
        t, c = b['t'], b.get('c')
        if t in ('Para', 'Plain'):
            out.append(inlines(c, bold, cites))
        elif t in ('BulletList', 'OrderedList'):
            env = 'itemize' if t == 'BulletList' else 'enumerate'
            items = c if t == 'BulletList' else c[1]
            out.append('\\begin{' + env + '}\n' + '\n'.join(r'\item ' + blocks_tex(item, bold, cites)
                       for item in items) + '\n\\end{' + env + '}')
        else:
            raise ValueError('Unhandled nested block: ' + t)
    return '\n\n'.join(out)


def caption(bs, prefix):
    value = blocks_tex(bs, bold=False)
    return re.sub(r'^' + prefix + r'\s+[A-H]?\.?\d+\.?\s*', '', value).strip()


TABLE_NAMES = ['1', '2', '3', '4a', '4b', 'A.1', 'A.2', 'A.3', 'A.4', 'A.5',
               'B.1', 'C.1', 'D.1', 'D.2', 'D.3', 'F.1', 'G.1', 'G.2', 'G.3',
               'H.1', 'H.2', 'H.3']
WIDTHS = {
    '1': [0.19, .12, .15, .12, .10, .15, .17],
    '2': [.28, .144, .144, .144, .144, .144],
    '3': [.15, .24, .122, .122, .122, .122, .122],
    '4a': [.30, .14, .14, .14, .14, .14],
    '4b': [.30, .14, .14, .14, .14, .14],
    'B.1': [.13, .13, .265, .10, .16, .10, .115],
    'C.1': [.25, .16, .40, .19],
    'D.1': [.16, .135, .135, .165, .135, .135, .135],
    'D.2': [.16, .16, .16, .18, .16, .18],
    'D.3': [.16, .17, .17, .17, .165, .165],
    'F.1': [.16, .16, .17, .17, .17, .17],
    'G.1': [.18, .22, .24, .36],
    'G.2': [.20, .44, .36],
    'G.3': [.15, .20, .10, .10, .075, .12, .135, .12],
    'H.1': [.21, .115, .15, .11, .09, .15, .175],
    'H.2': [.16, .14, .13, .14, .115, .09, .14, .125],
    'H.3': [.15, .15, .15, .15, .13, .10, .17],
}


def get_rows(b):
    c = b['c']
    return c[3][1] + [r for body in c[4] for r in body[2] + body[3]] + c[5][1]


def table_tex(b, name, cap):
    rows = get_rows(b)
    nc = len(b['c'][2])
    widths = WIDTHS.get(name, [.24] + [.76 / (nc - 1)] * (nc - 1))
    assert len(widths) == nc
    widths = [w / sum(widths) for w in widths]
    spec = []
    prose = name in ('G.1', 'G.2', 'G.3')
    for j, w in enumerate(widths):
        align = r'\raggedright' if j == 0 or prose or (name == '3' and j == 1) or (name == 'B.1' and j == 2) or (name == 'C.1' and j < 3) else r'\centering'
        spec.append('>{' + align + r'\arraybackslash}p{\dimexpr ' + f'{w:.6f}' + r'\TableWidth\relax}')
    encoded, checks = [], []
    for i, row in enumerate(rows):
        cells = []
        assert len(row[1]) == nc, (name, i)
        for cell in row[1]:
            assert cell[2:4] == [1, 1], (name, cell[2:4])
            value = blocks_tex(cell[4])
            if i == 0:
                value = value.replace(r'\ensuremath{-}', r' \ensuremath{-} ')
                value = value.replace('Temperature', r'Tempe\-rature')
            cells.append(value)
            checks.append({'source': plain(cell[4]), 'latex': value})
        encoded.append(' & '.join(cells) + r' \\')
    head = encoded[0]
    body = '\n'.join(encoded[1:])
    setting = r'\fontsize{9}{11}\selectfont'
    if name in ('G.1', 'G.2'):
        setting = r'\fontsize{9.5}{12}\selectfont'
    if name in ('H.2', 'G.3'):
        setting = r'\fontsize{8.5}{10.5}\selectfont'
    space = ''
    if len(rows) <= 12 and not name.startswith('4'):
        height = 115 + len(rows) * 22
        space = r'\Needspace{' + str(height) + 'pt}\n'
    common = (space + '\\begingroup\n' + setting + '\n' +
              r'\setlength{\tabcolsep}{3pt}' + '\n' +
              r'\renewcommand{\arraystretch}{1.18}' + '\n' +
              r'\setlength{\TableWidth}{\dimexpr\linewidth-' + str(2 * (nc - 1)) + r'\tabcolsep\relax}' + '\n')
    if name.startswith('4'):
        table = (common + r'\begin{tabular}{@{}' + ''.join(spec) + '@{}}\n' +
                 r'\toprule' + '\n' + head + '\n' + r'\midrule' + '\n' + body + '\n' +
                 r'\bottomrule' + '\n' + r'\end{tabular}' + '\n' + r'\endgroup')
    else:
        table = (common + r'\begin{longtable}{@{}' + ''.join(spec) + '@{}}\n' +
                 r'\caption{' + cap + r'}\label{tab:' + name + r'}\\' + '\n' +
                 r'\toprule' + '\n' + head + '\n' + r'\midrule\endfirsthead' + '\n' +
                 r'\multicolumn{' + str(nc) + r'}{l}{\small\itshape Table \thetable{} (continued)}\\' + '\n' +
                 r'\toprule' + '\n' + head + '\n' + r'\midrule\endhead' + '\n' +
                 r'\midrule\multicolumn{' + str(nc) + r'}{r}{\small\itshape Continued on the next page}\\\endfoot' + '\n' +
                 r'\bottomrule\endlastfoot' + '\n' + body + '\n' + r'\end{longtable}' + '\n' + r'\endgroup')
    AUDIT['tables'].append({'name': name, 'rows': len(rows), 'columns': nc, 'cells': checks,
                            'latex': table})
    return table


def algorithm_tex(number):
    src = (ROOT / '_docx_tools' / 'latex_algorithms' / f'algorithm{number}_en.tex').read_text(encoding='utf-8')
    content = re.search(r'\\begin\{algorithm\}\[H\].*?\\end\{algorithm\}', src, re.S)[0]
    content = content.replace(r'\begin{algorithm}[H]', r'\begin{algorithm}[htbp]')
    return content.replace(r'\KwIn', r'\label{alg:' + str(number) + '}\n' + r'\KwIn', 1)


def main():
    figures = sorted((OUT / 'figures').glob('fig[1-5]_*.png'))
    assert len(figures) == 5
    body, refs = [], []
    table_index = figure_index = 0
    pending_caption = None
    in_refs = False
    app = False
    table4_caption = ''
    abstract = blocks_tex([BLOCKS[3]])
    highlights = blocks_tex([BLOCKS[1]])
    keywords = re.sub(r'^Keywords\s*:\s*', '', plain(BLOCKS[4]))
    for i, b in enumerate(BLOCKS):
        if i < 5:
            continue
        t = b['t']
        txt = plain(b)
        if t == 'Header':
            if not txt.strip():
                continue
            if txt == 'References':
                in_refs = True
                body.append(r'\FloatBarrier' + '\n' + r'\input{references}')
                continue
            upcoming = BLOCKS[i + 1:i + 5]
            next_table = next((j for j, item in enumerate(upcoming) if item['t'] == 'Table'), None)
            if next_table is not None and not any(item['t'] == 'Header' and not re.match(r'^(?:[1-8A-H]\.\d)', plain(item)) for item in upcoming[:next_table]):
                row_count = len(get_rows(upcoming[next_table]))
                if row_count <= 12:
                    intro_chars = sum(len(plain(item)) for item in upcoming[:next_table])
                    reserve = min(650, 145 + row_count * 22 + int(intro_chars / 90) * 15)
                    body.append(r'\Needspace{' + str(reserve) + 'pt}')
            if txt == 'Supplementary material':
                in_refs = False
                app = True
                body.append(r'''\clearpage
\appendix
\section*{Supplementary material}
\addcontentsline{toc}{section}{Supplementary material}
\renewcommand{\thetable}{\Alph{section}.\arabic{table}}
\renewcommand{\theHtable}{\Alph{section}.\arabic{table}}
''')
                continue
            if txt.startswith('Appendix '):
                match = re.fullmatch(r'Appendix ([A-H])\.\s*(.+)', txt)
                assert match, txt
                body.append(r'\FloatBarrier' + '\n' + r'\setcounter{table}{0}' + '\n' +
                            r'\section{' + esc(match[2]) + r'}\label{sec:' + match[1] + '}')
            else:
                match = re.fullmatch(r'([1-8A-H](?:\.\d+)*)\s+(.+?)\s*', txt)
                assert match, txt
                number, heading = match.groups()
                depth = number.count('.')
                command = ['section', 'subsection', 'subsubsection'][depth]
                if depth == 0:
                    body.append(r'\FloatBarrier')
                body.append('\\' + command + '{' + esc(heading) + r'}\label{sec:' + number + '}')
            AUDIT['blocks'].append(i)
        elif in_refs:
            match = re.match(r'^\[(\d+)\]\s+', txt)
            assert match, txt
            number = int(match[1])
            xs = copy.deepcopy(b['c'])
            assert xs[0]['c'] == '[' + str(number) + ']'
            xs = xs[2:]
            ref = inlines(xs, cites=False)
            refs.append(r'\bibitem{ref' + str(number) + '}\n' + ref)
            AUDIT['references'].append({'number': number, 'source': txt, 'latex': ref})
            AUDIT['blocks'].append(i)
        elif t == 'Figure':
            number = figure_index + 1
            cap = caption(b['c'][1][1], r'(?:Fig\.|Figure)')
            filename = figures[figure_index].name
            body.append(r'\begin{figure}[htbp]' + '\n' + r'\centering' + '\n' +
                        r'\includegraphics[width=\linewidth]{figures/' + filename + '}\n' +
                        r'\caption{' + cap + r'}\label{fig:' + str(number) + '}\n' + r'\end{figure}')
            AUDIT['figures'].append({'number': number, 'file': filename})
            AUDIT['blocks'].append(i)
            figure_index += 1
        elif t == 'Table':
            name = TABLE_NAMES[table_index]
            cap = caption(b['c'][1][1], 'Table') if b['c'][1][1] else pending_caption
            if name == '4a':
                body.append(r'\begin{table}[htbp]' + '\n' + r'\centering' + '\n' +
                            r'\caption{' + table4_caption + r'}\label{tab:4}' + '\n' +
                            r'{\small\textbf{(a) Uniform 0.4}}\par\smallskip')
            elif name == '4b':
                body.append(r'\par\medskip{\small\textbf{(b) Uniform 0.6}}\par\smallskip')
            assert cap or name.startswith('4'), name
            body.append(table_tex(b, name, cap))
            if name == '4b':
                body.append(r'\end{table}')
            pending_caption = None
            table_index += 1
            AUDIT['blocks'].append(i)
        elif t in ('Para', 'Plain'):
            if re.match(r'^Algorithm [12]:', txt):
                number = int(txt[10])
                body.append(algorithm_tex(number))
                AUDIT['blocks'].append(i)
                continue
            if any(x['t'] == 'Image' for x in b['c']):
                continue
            if re.match(r'^Table (?:[A-H]\.\d+|4)\.?\s+(?!reports)', txt):
                pending_caption = caption([b], 'Table')
                if re.match(r'^Table 4[. ]', txt):
                    table4_caption = pending_caption
                AUDIT['blocks'].append(i)
                continue
            if txt in ('(a) Uniform 0.4', '(b) Uniform 0.6'):
                continue
            nonspace = [x for x in b['c'] if x['t'] != 'Space']
            if nonspace and nonspace[0]['t'] == 'Math':
                eqnum = re.fullmatch(r'\((\d+)\)', plain(nonspace[1:]))
                if eqnum:
                    number = int(eqnum[1])
                    formula = math(nonspace[0]['c'][1])
                    body.append(r'\begin{equation}\label{eq:' + str(number) + '}\n' +
                                formula + '\n' + r'\end{equation}')
                    AUDIT['equations'].append({'number': number, 'source': nonspace[0]['c'][1], 'latex': formula})
                    AUDIT['blocks'].append(i)
                    continue
                if len(nonspace) == 1 and nonspace[0]['c'][0]['t'] == 'DisplayMath':
                    body.append('\\[\n' + math(nonspace[0]['c'][1]) + '\n\\]')
                    AUDIT['blocks'].append(i)
                    continue
            body.append(blocks_tex([b]))
            AUDIT['blocks'].append(i)
        else:
            body.append(blocks_tex([b]))
            AUDIT['blocks'].append(i)
    assert table_index == 22 and figure_index == 5 and len(refs) == 63
    assert [e['number'] for e in AUDIT['equations']] == list(range(1, 21))
    title = inlines(DOC['meta']['title']['c'], bold=False)
    preamble = (HERE / 'preamble.tex').read_text(encoding='utf-8')
    main_tex = preamble + '\n' + r'\begin{document}' + '\n' + r'\begin{frontmatter}' + '\n'
    main_tex += r'\title{' + title + '}\n'
    main_tex += '% Author and affiliation information can be added here.\n'
    main_tex += r'\begin{abstract}' + '\n' + abstract + '\n' + r'\end{abstract}' + '\n'
    main_tex += r'\begin{keyword}' + '\n' + r' \sep '.join(esc(k.strip()) for k in keywords.split(',')) + '\n' + r'\end{keyword}' + '\n'
    main_tex += r'\end{frontmatter}' + '\n\n% The unchanged Highlights are supplied separately in highlights.tex.\n\n'
    main_tex += '\n\n'.join(body) + '\n\n' + r'\end{document}' + '\n'
    OUT.mkdir(exist_ok=True)
    (OUT / 'main.tex').write_text(main_tex, encoding='utf-8')
    (OUT / 'references.tex').write_text(r'\begin{thebibliography}{99}' + '\n' + r'\raggedright' + '\n\n' + '\n\n'.join(refs) + '\n\n' + r'\end{thebibliography}' + '\n', encoding='utf-8')
    (HERE / 'conversion_audit.json').write_text(json.dumps(AUDIT, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'tables': len(AUDIT['tables']), 'table_cells': sum(len(t['cells']) for t in AUDIT['tables']),
                      'numbered_equations': len(AUDIT['equations']), 'references': len(refs),
                      'figures': figure_index, 'main_tex': str(OUT / 'main.tex')}, indent=2))


if __name__ == '__main__':
    main()
