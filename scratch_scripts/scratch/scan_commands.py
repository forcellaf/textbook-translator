"""List every \\command token used in math, flagging ones LaTeX won't know.

Cheaper than discovering them one pandoc run at a time.
"""

import re
from collections import Counter
from pathlib import Path

t = Path('translated_book.md').read_text(encoding='utf-8')

# Commands that plain LaTeX + amsmath/amssymb provide. Not exhaustive, but
# anything outside it is worth eyeballing.
KNOWN = set("""
alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa
lambda mu nu xi pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi
psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega
frac sqrt sum int oint prod lim inf sup max min log ln exp sin cos tan cot sec
csc arcsin arccos arctan sinh cosh tanh det dim ker deg gcd hom Pr
mathrm mathbf mathit mathcal mathbb mathfrak mathsf mathtt mathscr boldsymbol
pmb text textbf textit textrm mbox hbox
times div pm mp cdot cdots ldots vdots ddots dots
leq geq neq approx equiv sim simeq cong propto ll gg
leqslant geqslant
rightarrow leftarrow Rightarrow Leftarrow leftrightarrow to gets mapsto
uparrow downarrow longrightarrow longleftarrow
partial nabla infty forall exists in notin subset supset subseteq supseteq
cup cap emptyset varnothing angle perp parallel
hat bar vec dot ddot tilde widehat widetilde overline underline overrightarrow
left right big Big bigg Bigg langle rangle lfloor rfloor lceil rceil
quad qquad hspace vspace displaystyle textstyle scriptstyle scriptscriptstyle
begin end label ref caption
prime circ ast star bullet oplus otimes wedge vee neg
binom choose over atop
mathop limits nolimits substack
Vert vert lvert rvert lVert rVert
color textcolor
""".split())

# Pull command tokens from inside math only.
math_spans = re.findall(r'\$\$(.*?)\$\$', t, flags=re.DOTALL)
math_spans += re.findall(r'(?<!\$)\$([^$\n]+?)\$(?!\$)', t)
blob = '\n'.join(math_spans)

tokens = Counter(re.findall(r'\\([A-Za-z]+)', blob))
unknown = {k: v for k, v in tokens.items() if k not in KNOWN}

print(f'distinct command tokens in math: {len(tokens)}')
print(f'not in the known list          : {len(unknown)}\n')
for name, count in sorted(unknown.items(), key=lambda kv: -kv[1]):
    print(f'  {count:>4}x  \\{name}')

# Show context for the rarest ones -- those are the likely typos.
print('\n--- context for tokens appearing 1-2 times ---')
lines = t.split('\n')
for name, count in sorted(unknown.items(), key=lambda kv: kv[1]):
    if count > 2:
        break
    for i, line in enumerate(lines, 1):
        if '\\' + name in line:
            idx = line.find('\\' + name)
            print(f'  L{i} \\{name}: ...{line[max(0, idx-60):idx+60]}...')
            break
