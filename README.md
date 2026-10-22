# crsmith.dev

The personal site of Chris Smith, built with [Hugo](https://gohugo.io/) 0.167.0.

## Run

```
hugo server
```

The server shows a type panel at the bottom right. The panel changes the fonts in your browser only. Copy its settings into the `:root` type variables in `assets/css/site.css` to keep them. Production builds leave the panel out.

## Build

```
hugo --gc --minify
```

The output is in `public/`.

## Content

| File | What it holds |
|---|---|
| `hugo.toml` | Name, role line, tagline, contact links |
| `content/_index.md` | The About text |
| `content/projects/*.md` | One project each: `title`, `link`, `weight` and the text |
| `content/projects/_index.md` | The paragraph after the project list |
| `content/experience/*.md` | One job each: `title`, `when`, `small`, `weight` and the text |
| `content/experience/_index.md` | The paragraphs after the job list |
| `content/blog/*.md` | Posts |

`weight` sets the order. `small = true` shows a job in the compact style.

A new post:

```
hugo new content blog/my-post.md
```

## Deploy

Every push to `master` builds the site and deploys it to GitHub Pages at `crsmith.dev` (`.github/workflows/pages.yml`).
