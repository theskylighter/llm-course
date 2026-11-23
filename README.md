# Introduction to Large Language Models

A study site: Markdown lessons rendered by a React app, with every code example executed by a validator so the output on the page is always real.

## Layout

```
content/
  site.json          title, brand name, storage prefix
  course.json        chapters and lessons (slug, title, syllabus tag) in reading order
  lessons/           one Markdown file per lesson; index.md is the home page
  diagrams/          SVG figures, included in a lesson with <<SVG:name>>
tools/
  sitegen.py         command-line entry point (export | check)
  sitegen/           Markdown renderer, exporter, code validator
web/                 React + TypeScript + Vite app
  e2e/smoke.mjs      browser test (headless Chrome)
  src/data/lessons.json   generated from content/ (not committed)
```

## Commands

Needs Python 3.10+ and Node 18+. The Python side uses only the standard library.

```bash
python tools/sitegen.py export        # content/ -> web/src/data/lessons.json
cd web && npm install                 # first time only
npm run dev                           # http://localhost:5173
npm run build                         # typecheck + production build into web/dist
npm run e2e                           # browser tests (needs Chrome; starts its own preview server)
python tools/sitegen.py check         # run every code example and compare with its ```output block
python tools/sitegen.py check 01_     # only lessons whose file name contains "01_"
```

After editing a lesson, run `export` again (or restart `npm run dev`).

## Writing a lesson

1. Add `{ "slug", "title", "syllabus" }` to `content/course.json`. Slugs named `NN_M_name` (for example `03_2_loops`) get the number 3.2 in the sidebar.
2. Create `content/lessons/<slug>.md`. See `01_1_anatomy.md` for every supported block.
3. `python tools/sitegen.py check <slug>` if it has code, then check the page in `npm run dev`.

## Deploying (Vercel)

`vercel.json` already runs the export and the web build and serves `web/dist` with single-page-app rewrites. Connect the GitHub repo to a Vercel project and every push to `main` deploys.

The code examples use the standard library and NumPy, so install NumPy (`pip install numpy`) before running `python tools/sitegen.py check`. Math is written in LaTeX (`$...$`, `$$...$$`) and rendered with KaTeX. `cd web && npm run test` is an alias for `npm run e2e`.
