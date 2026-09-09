import re
from pathlib import Path

t = Path('translated_book.md').read_text(encoding='utf-8')
lines = t.split('\n')

# Any \$ in the file: a literal dollar sign is legitimate, but in this book
# (no currency anywhere) it is almost certainly a math delimiter that got escaped.
hits = [(i, l) for i, l in enumerate(lines, 1) if r'\$' in l]
print(f'lines containing an escaped \\$ : {len(hits)}\n')
for i, l in hits:
    idx = l.find(r'\$')
    print(f'  L{i}: ...{l[max(0, idx-70):idx+30]}...')

# Independently: prose lines with an odd number of unescaped $, which is what
# actually breaks pandoc's math parsing.
print('\n--- prose lines with an odd number of unescaped $ ---')
in_display = False
bad = 0
for i, line in enumerate(lines, 1):
    if line.strip() == '$$':
        in_display = not in_display
        continue
    if in_display:
        continue
    n = len(re.findall(r'(?<!\\)\$', line))
    if n % 2:
        bad += 1
        print(f'  L{i}: {line.strip()[:130]}')
print(f'\ntotal: {bad}')
