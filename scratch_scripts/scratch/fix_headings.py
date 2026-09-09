"""
Rebuild the heading hierarchy.

MinerU emitted every heading in the book at level 1 (#) -- chapters, sections,
subsections, worked examples, all flat. Pandoc turns each one into a \\chapter,
and every \\chapter forces a page break onto a fresh recto page. That single
defect causes both the broken outline AND most of the blank pages.

Rules, applied in order:

  #     Chapter N                    (the chapter's own title, if it follows on
                                      its own line, is merged into this heading)
  ##    12.1 Title                   numbered sections
  ##    G.1 Title                    lettered aside sections
  ##    Summary / Questions / Exercises / Problems
  ###   1. Subsection                numbered subsections
  ###   Example 12.1
  ###   anything else

One line, "Electric charge $\\rightarrow$ Electric charge", is body text that
MinerU mistakenly marked as a heading; it is demoted to a plain paragraph.
"""

import re
from pathlib import Path

path = Path('translated_book.md')
lines = path.read_text(encoding='utf-8').split('\n')

HEADING = re.compile(r'^#\s+(.*\S)\s*$')

CHAPTER = re.compile(r'^Chapter\s+\d+\s*$')
SECTION = re.compile(r'^\d+\.\d+\s+\S')
LETTER_SECTION = re.compile(r'^[A-Z]\.\d+\s+\S')
BACK_MATTER = re.compile(r'^(Summary|Questions|Exercises|Problems)\s*$', re.I)
NUMBERED_SUB = re.compile(r'^\d+\.\s+\S')
EXAMPLE = re.compile(r'^Example\s+\d', re.I)

# Not a heading at all -- body text MinerU mis-tagged.
NOT_A_HEADING = {r'Electric charge $\rightarrow$ Electric charge'}

out = []
counts = {1: 0, 2: 0, 3: 0, 'demoted': 0, 'merged': 0}
i = 0
while i < len(lines):
    m = HEADING.match(lines[i])
    if not m:
        out.append(lines[i])
        i += 1
        continue

    title = m.group(1)

    if title in NOT_A_HEADING:
        out.append(title)
        counts['demoted'] += 1
        i += 1
        continue

    if CHAPTER.match(title):
        # Look ahead: a chapter's real title often sits in the next heading,
        # separated only by a blank line. Merge it in.
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        nxt = HEADING.match(lines[j]) if j < len(lines) else None
        if nxt:
            cand = nxt.group(1)
            is_other = not (CHAPTER.match(cand) or SECTION.match(cand)
                            or LETTER_SECTION.match(cand) or BACK_MATTER.match(cand)
                            or NUMBERED_SUB.match(cand) or EXAMPLE.match(cand))
            if is_other and len(cand) < 80:
                out.append(f'# {title}: {cand}')
                counts[1] += 1
                counts['merged'] += 1
                i = j + 1
                continue
        out.append(f'# {title}')
        counts[1] += 1
    elif SECTION.match(title) or LETTER_SECTION.match(title) or BACK_MATTER.match(title):
        out.append(f'## {title}')
        counts[2] += 1
    else:
        # numbered subsections, examples, and everything else
        out.append(f'### {title}')
        counts[3] += 1
    i += 1

path.write_text('\n'.join(out), encoding='utf-8')

print(f'level 1 (chapters)   : {counts[1]}')
print(f'level 2 (sections)   : {counts[2]}')
print(f'level 3 (subsections): {counts[3]}')
print(f'chapter titles merged: {counts["merged"]}')
print(f'demoted to body text : {counts["demoted"]}')
print(f'total                : {counts[1] + counts[2] + counts[3]}')
