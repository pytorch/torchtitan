# pr-image tools

- HTML/CSS with inline SVG for arrows (default; boxes, code and tables are easy):
  `python3 -I ${CLAUDE_SKILL_DIR}/scripts/render_html.py <form>.html` writes `<form>.png` and `<form>_900px.png`.
  It needs Pillow and a headless Chrome on Linux: Playwright's (`npx playwright install chromium-headless-shell`)
  or `google-chrome`/`chromium` on PATH. The page can't load files or reach the network: inline every image and font.
  SVG arrowheads: `<marker markerUnits="userSpaceOnUse" ...>`, or they grow with the stroke width. In inline JS,
  don't name a global `top` (it is `window.top`; the script dies and the PNG comes out blank). SVG `<text>` drops
  leading spaces: put `style="white-space: pre"` on it for indented code. At 900 px about 110 monospace (13 px) or
  118 body (14 px) characters fit a line; render to check.
- Locally only: a `<form>.py` that writes the HTML for to-scale geometry; matplotlib for charts with real axes
  (`figsize=(9, h), dpi=200`); graphviz (`dot -Tpng -Gdpi=200`) for graphs with many nodes. GitHub Actions runs
  HTML only.
- [examples/](../examples/README.md): HTML sources of images that worked.
- CSS `svg text { fill: ...; font-size: ... }` silently overrides the SVG `fill=` and `font-size=` attributes:
  put defaults on `svg { }` instead. Inside a padded 900 px panel only ~828 px fit: size the SVG to its parent.
