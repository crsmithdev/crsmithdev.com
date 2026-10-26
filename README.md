# crsmith.dev

The personal site of Chris Smith, built with [Astro](https://astro.build/) 7.

## Run

```
npm install
npm run dev
```

The dev server shows drafts and posts dated in the future. Production builds leave them out.

## Build

```
npm run build
```

The output is in `dist/`. `npm run preview` serves it.

## Content

| File | What it holds |
|---|---|
| `src/site.ts` | Name, role line, tagline, description, contact links |
| `src/content/sections/about.md` | The About text |
| `src/content/projects/*.md` | One project each: `title`, `link`, `weight` and the text |
| `src/content/sections/projects.md` | The paragraph after the project list |
| `src/content/experience/*.md` | One job each: `title`, `when`, `small`, `weight` and the text |
| `src/content/sections/experience.md` | The paragraphs after the job list |
| `src/content/blog/*.md` | Posts |
| `src/content/blog/<slug>/index.mdx` | Posts with components or files beside them |

`weight` sets the order. `small: true` shows a job in the compact style.

A post has `title`, `date` (with its offset, for example `"2026-09-21T21:00:00-07:00"`) and optionally `draft: true`. The text before `<!--more-->` (`{/* more */}` in MDX) is the summary for the feeds and the meta description. Without it, the summary is the first paragraphs up to 70 words.

In MDX, `Svg` inlines an SVG so that it takes the page's colours:

```mdx
import Svg from "../../../components/Svg.astro";
import chart from "./chart.svg?raw";

<Svg svg={chart} caption="What the chart shows." />
```

## Deploy

Every push to `master` builds the site and deploys it to GitHub Pages at `crsmith.dev` (`.github/workflows/pages.yml`).
