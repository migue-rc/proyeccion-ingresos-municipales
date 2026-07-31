# quarto-project-template

Template for project sites published under https://migue-rc.github.io/PROJECT-NAME,
sharing the hub's theme (light/dark SCSS + remote `brand-theme.css`).

## Creating a new project

1. Create the repo from this template.
2. In `_quarto.yml`, replace `PROJECT-TITLE` and `PROJECT-NAME`.
   `site-url` is required — without it Quarto does not generate `sitemap.xml`.
3. In `llms.txt`, replace the placeholders and describe the project's pages.
4. Write your notebook(s) and `make publish`.

## Search-engine / LLM discoverability (IndexNow)

`make publish` pings [IndexNow](https://www.indexnow.org) after deploying, so
Bing, Naver, Seznam, Yandex (and the AI-search indexes they feed) learn about
changed pages within minutes instead of waiting for a crawl.

**There is no key to generate for a project.** An IndexNow key authorizes an
entire *host*, and every project lives under the same host as the hub
(`migue-rc.github.io`). The hub serves one key file at its root
(`https://migue-rc.github.io/indexnow-key.txt`) that already authorizes every
project URL. `scripts/submit_indexnow.py` fetches that key at publish time,
reads this project's own `sitemap.xml`, and submits the URLs changed in the
last day. Nothing to create, nothing to store — rotating the key is a one-file
edit on the hub that every project inherits automatically.

Run `make indexnow` on its own to ping without republishing, or
`python3 scripts/submit_indexnow.py --dry-run` to preview.

## Registering the project on the hub

The hub repo (`migue-rc.github.io`) is the host root, so it owns `robots.txt`,
the top-level `llms.txt` and the IndexNow key. After publishing a new project,
over there just add a card to `projects.yml` and run `make publish` — the hub
regenerates its own `robots.txt` sitemap list and `llms.txt` project list from
`projects.yml` automatically.

Do **not** add a `robots.txt` or an IndexNow key to project repos — crawlers
read robots.txt only at the host root, and the key is host-wide, so the hub's
copies govern every project path.

## Styles

- `sketchy-light.scss` / `sketchy-dark.scss` — shared, fetched from the hub
  (`make fetch-styles`); do not edit.
- `custom.scss` / `local-style.css` — per-project overrides; safe to edit.

## Make targets

| Target | Purpose |
| --- | --- |
| `make preview` | Local preview |
| `make publish` | Fetch latest shared styles, publish to gh-pages, then ping IndexNow |
| `make fetch-styles` | Pull shared SCSS from the hub over the network |
| `make sync-styles` | Copy shared SCSS from a local hub checkout |
| `make indexnow` | Ping IndexNow with this project's changed URLs |
