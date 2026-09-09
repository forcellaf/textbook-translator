"""
Normalise inline math so pandoc recognises it.

Pandoc requires an opening $ to be immediately followed by a non-space
character, and a closing $ to be immediately preceded by one. MinerU writes
spacing-heavy LaTeX, so spans like

    $n S { \\mathrm { d } } l $

are common. Pandoc does not read those as math at all: it escapes the $ and the
LaTeX commands leak into text mode, giving errors like

    ! LaTeX Error: \\mathrm allowed only in math mode.

This trims whitespace immediately inside inline delimiters. Display math ($$ on
its own line) is left completely alone, and the mathematical content is
unchanged -- LaTeX ignores that whitespace when typesetting.
"""

import re
from pathlib import Path

path = Path('translated_book.md')
lines = path.read_text(encoding='utf-8').split('\n')

INLINE = re.compile(r'(?<!\$)\$([^$\n]+?)\$(?!\$)')

changed = 0


def tidy(match):
    global changed
    inner = match.group(1)
    stripped = inner.strip()
    if stripped != inner:
        changed += 1
    return '$' + stripped + '$'


in_display = False
out_lines = []
for line in lines:
    if line.strip() == '$$':
        in_display = not in_display
        out_lines.append(line)
    elif in_display:
        out_lines.append(line)
    else:
        out_lines.append(INLINE.sub(tidy, line))

out = '\n'.join(out_lines)
path.write_text(out, encoding='utf-8')

print(f'display math balanced at EOF: {not in_display}')
print(f'inline spans trimmed        : {changed}')
print(f'$$ delimiters               : {out.count("$$")} (even: {out.count("$$") % 2 == 0})')

in_display = False
remaining = 0
for line in out.split('\n'):
    if line.strip() == '$$':
        in_display = not in_display
        continue
    if in_display:
        continue
    for m in INLINE.finditer(line):
        if m.group(1) != m.group(1).strip():
            remaining += 1
print(f'offending spans remaining   : {remaining}')
