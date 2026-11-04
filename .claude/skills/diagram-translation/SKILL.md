---
name: diagram-translation
description: How to turn one scanned textbook figure with Chinese labels into an English figure for the book - choosing between a TikZ redraw, an English overlay on the scan, or a legend, and checking the result. Preloaded by the diagram-translator agent; use it when reconstructing or relabelling a figure from data/work/<book>/kit/figure_text/.
---

# Translating the text inside a figure

The book is typeset with pdflatex, which **cannot set Chinese**. Every figure
you produce must contain no Chinese character at all, and must say exactly
what the original says, in English, with the book's own terminology.

## 1. Read the figure before choosing

List every piece of Chinese in the image: labels, axis titles, units written
in Chinese (`秒`, `米`), legends, annotations, table cells, speech bubbles,
甲/乙/丙 used as names (translate as A/B/C, or keep the book's letters if the
translated text already names them). Note what is *not* text: Latin and Greek
letters, numbers, symbols stay exactly as they are.

Take the English for each label from the task's `translated_context` first:
if the prose calls 电源 "the source", the figure says "source", not "power
supply". Keep labels short; a label is not a sentence.

## 2. Choose the method

| The image is ... | Method |
|---|---|
| a line drawing you can redraw faithfully: circuit, force/free-body diagram, ray diagram, block diagram, flow chart, graph or plot of a few curves, simple geometry | `tikz` |
| a photograph, a shaded or textured illustration, a dense drawing, a hand-drawn sketch, a map, or anything a redraw would simplify | `overlay` |
| an overlay that cannot fit (labels overlap the drawing, too many tiny labels) or a redraw you could not get right in three renders | `legend` |

When in doubt between `tikz` and `overlay`, choose `overlay`: a redraw that
moves, drops or adds a line changes the physics; an overlay cannot.

## 3. Write it

Everything goes in `<kit>/figure_text/`. `<stem>` is the image's file name
without its extension. The spec, `<stem>.json`, always:

    {"image": "<file name>", "method": "tikz|overlay|legend",
     "labels": [{"zh": "电源", "en": "source", ...}],
     "accepted": false, "notes": "one sentence for the run log"}

It is JSON: double every backslash (`"$R_{1}$"` is fine, `"\\Omega"`).
English label text is LaTeX text: math in `$...$`, and `%`, `&`, `#` are
escaped for you.

### `tikz`

Write a complete standalone document to `<stem>.tex`:

    \documentclass[border=2pt]{standalone}
    \usepackage{amsmath, amssymb}
    \usepackage{tikz}
    \usetikzlibrary{arrows.meta, calc, positioning, decorations.markings, patterns}
    \begin{document}
    \begin{tikzpicture}[x=1cm, y=1cm, >=Stealth, font=\footnotesize]
      ...
    \end{tikzpicture}
    \end{document}

- Make the drawing about `target_width_cm` wide (from the task) so that
  `\footnotesize` labels come out at book size; the build scales the PDF to
  that width.
- Work from the image: estimate coordinates by placing the image's extent on
  a grid (e.g. 0 to `target_width_cm` across) and reading positions off it.
  Keep proportions, the relative positions of every element, line styles
  (dashed, thick), arrow directions, and every non-Chinese label.
- `circuitikz` and `pgfplots` may be used (`\usepackage{circuitikz}`,
  `\usepackage{pgfplots}\pgfplotsset{compat=1.18}`) when they make the
  drawing more faithful; if one fails to compile with "File not found",
  draw with plain TikZ instead.
- Each `labels` entry still lists `zh` and `en` (no `box`), so the run log
  shows what was translated.

### `overlay`

Each label needs `box`: `[x0, y0, x1, y1]` as **fractions of the image's
width and height, origin top left**, covering the whole Chinese text (be
generous by a character's height; a sliver of a stroke left visible looks
like dirt). Optional `size`: `tiny`, `scriptsize`, `footnotesize` (default),
`small`, `normalsize`. The box is painted white and the English is centred on
it, on its own white background, so a longer English label may extend past
the box: make sure it does not cover a line that matters, or shorten it.

To estimate a box: find the label's left, right, top and bottom edges in the
image and divide by the image's width and height. Check yourself against a
second label (two labels on the same line have the same y).

### `legend`

The scan stays untouched; an English legend is set under it. Each label
needs `where`, a position a reader can find: "upper left", "on the battery",
"label beside the lower arrow", "x-axis", "y-axis". The legend cannot quote
the Chinese (pdflatex), so `where` must be unambiguous.

## 4. Render and look

    uv run python scripts/figure_text.py <kit> render <stem> --json

- If `ok` is false, fix what `errors` says and render again.
- If `ok`, Read the `preview` image and the original side by side. Check:
  no Chinese left, nothing covered that should show, every label present and
  readable, no label on the wrong element; for TikZ, every line, arrow,
  component and number of the original is there, in the same place.
- Fix and re-render; at most **three** renders per method. Then fall back:
  `tikz` -> `overlay` or `legend`; `overlay` -> `legend`.
- Only when the preview is right, set `"accepted": true` in the spec (do
  not change anything else after the last render: the build checks that the
  spec still matches what was rendered, and otherwise uses the original).
- If even the legend cannot be made right, leave `"accepted": false` and say
  why in `notes`: the book keeps the original scan and the run log says so.
