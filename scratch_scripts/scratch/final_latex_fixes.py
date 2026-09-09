"""
Final four fixes needed before pandoc can build the PDF.

All four are MinerU OCR damage in the answers section at the back of the book,
not translation errors. Each one halts XeLaTeX, and LaTeX reports them against
the *generated* .tex, so they surface one per build.

Run this after fix_math_spacing.py. Verified end to end: produces a complete
1,556-page PDF.
"""

from pathlib import Path

path = Path('translated_book.md')
text = path.read_text(encoding='utf-8')
applied = []

# 1. \mathrmmathrm -- a doubled command with a lost backslash.
#    "! Undefined control sequence."
if r'\mathrmmathrm' in text:
    text = text.replace(r'\mathrmmathrm', r'\mathrm')
    applied.append('doubled \\mathrm')

# 2. A display-math block that is 2,387 characters of pure OCR noise:
#    "\partial \mathbf{x}_\theta" repeated with invented fragments
#    (\mathrm{parag}, \mathrm{Eq}, \mathrm{MN}) and an unclosed "(".
#    Nothing recoverable; replace the whole block.
#    "! Argument of \math@egroup has an extra }."
lines = text.split('\n')
for i, line in enumerate(lines):
    if line.startswith(r'\begin{array} { r l } & { \frac { \partial \mathbf { x } _ { \theta }'):
        lines[i] = r'\text{[answer not recoverable from the scan]}'
        applied.append('garbled answer block')
        break
text = '\n'.join(lines)

# 3. An array declaring 4 columns with a 5-tab row.
#    "! Extra alignment tab has been changed to \cr."
old = r'\begin{array} { r l r l } { { 2 7 } }'
if old in text:
    text = text.replace(old, r'\begin{array} { r l r l l } { { 2 7 } }', 1)
    applied.append('array column spec')

# 4. \mathrm left dangling with no argument at the end of a math span.
#    "! Missing } inserted."
old = r'8 . 3 { \times } 1 0 ^ { - 2 1 } \ \mathrm$ T'
if old in text:
    text = text.replace(old, r'8 . 3 { \times } 1 0 ^ { - 2 1 } \ \mathrm { T }$', 1)
    applied.append('dangling \\mathrm')

path.write_text(text, encoding='utf-8')

print(f'applied {len(applied)}/4 fix(es):')
for a in applied:
    print(f'  - {a}')
if len(applied) < 4:
    print('\nSome patterns were not found -- they may already be fixed, or the')
    print('text differs slightly. Try the build and report any remaining error.')
