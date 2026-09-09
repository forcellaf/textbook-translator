"""Report every math block whose braces don't balance.

An unbalanced brace inside math produces errors like
    ! Argument of \\math@egroup has an extra }
and LaTeX reports the line in the *generated* .tex, not the markdown, so
locating it by hand is tedious. This checks all of them in one pass.
"""

import re
from pathlib import Path

lines = Path('translated_book.md').read_text(encoding='utf-8').split('\n')


def imbalance(s: str) -> int:
    """Net brace depth, ignoring escaped \\{ and \\}."""
    s = re.sub(r'\\[{}]', '', s)
    depth = 0
    low = 0
    for ch in s:
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            low = min(low, depth)
    return depth if depth else low


# --- display math -------------------------------------------------------
bad_display = []
in_block = False
start = 0
buf = []
for i, line in enumerate(lines, 1):
    if line.strip() == '$$':
        if in_block:
            body = '\n'.join(buf)
            d = imbalance(body)
            if d:
                bad_display.append((start, d, body))
            in_block = False
            buf = []
        else:
            in_block = True
            start = i
        continue
    if in_block:
        buf.append(line)

print(f'display math blocks with unbalanced braces: {len(bad_display)}')
for ln, d, body in bad_display:
    sign = f'+{d}' if d > 0 else str(d)
    print(f'  L{ln} ({sign}): {body.strip()[:150]}')

# --- inline math --------------------------------------------------------
INLINE = re.compile(r'(?<!\$)\$([^$\n]+?)\$(?!\$)')
bad_inline = []
in_block = False
for i, line in enumerate(lines, 1):
    if line.strip() == '$$':
        in_block = not in_block
        continue
    if in_block:
        continue
    for m in INLINE.finditer(line):
        d = imbalance(m.group(1))
        if d:
            bad_inline.append((i, d, m.group(1)))

print(f'\ninline math spans with unbalanced braces: {len(bad_inline)}')
for ln, d, body in bad_inline[:25]:
    sign = f'+{d}' if d > 0 else str(d)
    print(f'  L{ln} ({sign}): {body.strip()[:120]}')
