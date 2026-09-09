import re

t = open('/tmp/t3.tex', encoding='utf-8').read()

# \frac whose next non-space character is not {
pat = re.compile(r'\\frac\s*(?!\{)')
hits = [m.start() for m in pat.finditer(t)]
print(f'\\frac not followed by {{ : {len(hits)}')
for h in hits[:5]:
    ln = t[:h].count('\n') + 1
    print(f'  L{ln}: ...{t[max(0, h-140):h+140]}...\n')

# Same check in the markdown source
src = open('translated_book.md', encoding='utf-8').read()
hits2 = [m.start() for m in pat.finditer(src)]
print(f'\n--- in translated_book.md: {len(hits2)} ---')
for h in hits2[:5]:
    ln = src[:h].count('\n') + 1
    print(f'  L{ln}: ...{src[max(0, h-140):h+140]}...\n')
