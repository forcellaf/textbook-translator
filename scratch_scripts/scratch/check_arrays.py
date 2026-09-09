"""Find array/tabular rows with more & separators than the column spec allows.

Produces "! Extra alignment tab has been changed to \\cr" at compile time,
reported against the generated .tex rather than the markdown.
"""

import re
from pathlib import Path

t = Path('translated_book.md').read_text(encoding='utf-8')

# \begin{array}{ r l } ... \end{array}
ARRAY = re.compile(r'\\begin\{(array|tabular)\}\s*\{([^}]*)\}(.*?)\\end\{\1\}', re.DOTALL)

problems = []
for m in ARRAY.finditer(t):
    env, spec, body = m.group(1), m.group(2), m.group(3)
    # count real column letters in the spec
    cols = len(re.findall(r'[lcr]', spec))
    if cols == 0:
        continue
    # split rows on \\ (not \\\\ inside something else)
    rows = re.split(r'\\\\', body)
    for ri, row in enumerate(rows):
        # ignore & inside nested braces
        depth = 0
        tabs = 0
        for ch in row:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
            elif ch == '&' and depth == 0:
                tabs += 1
        if tabs > cols - 1:
            ln = t[:m.start()].count('\n') + 1
            problems.append((ln, env, spec.strip(), cols, ri + 1, tabs, row.strip()[:120]))

print(f'rows with too many alignment tabs: {len(problems)}\n')
for ln, env, spec, cols, ri, tabs, row in problems[:20]:
    print(f'  L{ln} {env}{{{spec}}}: {cols} cols, row {ri} has {tabs} tab(s)')
    print(f'      {row}')
