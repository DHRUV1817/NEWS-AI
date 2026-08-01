# Site

The page that runs the service. Next.js, TypeScript, no CSS framework — colour
and type resolve through tokens in `styles/tokens.css`.

```bash
npm install
npm run dev          # http://localhost:3000
```

It calls the HTTP service, so start that too:

```bash
cd ../api
uv run --python 3.12 uvicorn newsninja.api:create_app --factory --port 8000
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:8000` | The service the page calls |

Copy `.env.local.example` to `.env.local` to change it. The variable is read at
build time, so a deployed site must have it set before the build runs, not
after.

## What the page reads at build time

The evaluation section renders from `../docs/evals/latest.json`, which
`python -m evals.run --report` writes. The page keeps no copy of those numbers —
a hand-maintained duplicate is how a site ends up quoting a figure no run ever
produced.

If that file is absent the section says so and names the command. CI fails if a
build publishes the fallback while the report is committed, so the page cannot
quietly claim nothing has been measured.

## Deploying

Vercel, with **`web` as the root directory** — the repository holds the Python
service at `api/` and this at `web/`.

1. Import the repository, set Root Directory to `web`.
2. Set `NEXT_PUBLIC_API_URL` to the deployed service's origin, no trailing slash.
3. Deploy, then take the resulting origin and put it in the service's
   `ALLOWED_ORIGINS` — the two point at each other, and the service refuses
   cross-origin requests from anywhere it has not been told about.

Order matters on the second half. `ALLOWED_ORIGINS` is empty by default, which
means no cross-origin access rather than any; until the service names this
origin the page loads and every call fails at the browser.

The build reads `../docs/evals/latest.json` from the repository, so the report
must be committed for the deployed page to show numbers. It is.

## Tests

```bash
npm test
```

The run's orchestration lives in `lib/briefing.ts`, not in the component, so the
part worth testing runs without a browser: that a refused topic does not take
the run down with it, that the briefing is built from whatever survived, that an
all-failed run never sends an empty list to the service, and that topics extract
one at a time rather than colliding over a shared per-minute budget.

Rendering is not tested. The page shows what the service returned and what the
harness wrote; the behaviour worth protecting is the orchestration and the
package underneath it, which has its own suite.

CI type checks, lints, tests, builds, and asserts the eval numbers actually
reached the rendered HTML.
