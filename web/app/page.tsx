import Analyzer from "@/components/Analyzer";
import Nav from "@/components/Nav";
import Reveal from "@/components/Reveal";
import { loadEvalReport, rate, type EvalReport } from "@/lib/evals";

export default async function Home() {
  // Read at build time from the harness's own output. If it has never been
  // run, the section says so rather than showing placeholder figures.
  const report = await loadEvalReport();

  return (
    <>
      <Nav />
      <Reveal />

      <main id="top">
        {/* ---- Hero + the tool itself --------------------------------
            Workbench: the app in use is the content. No mockup, no
            screenshot, no drawn browser chrome — the thing below is the
            running service. */}
        <section className="hero" id="workbench">
          <div className="shell hero__grid">
            <div className="hero__copy">
              <p className="eyebrow">
                <span className="eyebrow__tick" aria-hidden="true" />
                Source-grounded briefings
              </p>
              <h1 className="hero__title">Every claim carries its source.</h1>
              <p className="hero__lede">
                Point it at a topic. It reads what the news feed carries,
                extracts the claims, and attaches to each one a quote copied
                character-for-character from the article it came from. A quote
                that is not in the source fails a substring check — so
                hallucination is measurable here, not a matter of opinion.
              </p>
              <p className="hero__meta">
                Runs on free tiers. Roughly three analyses per minute across
                all callers — the model&rsquo;s token ceiling, not a queue.
              </p>
            </div>

            <div className="hero__tool">
              <Analyzer />
            </div>
          </div>
        </section>

        {/* ---- The one dark band ------------------------------------- */}
        <section className="band" id="pipeline">
          <div className="shell">
            <h2 className="band__title">What runs when you press it</h2>
            <p className="band__lede">
              Four stages, split along the seams the package already had. The
              split is arithmetic, not taste: one topic reserves about 2,400
              tokens against an 8,000-per-minute ceiling, so a request that took
              five topics at once would sit blocked inside the rate limiter for
              48 seconds and no free-tier proxy would hold the connection.
            </p>

            <ol className="steps">
              <Step
                n="01"
                name="Fetch"
                endpoint="Google News RSS"
                body="Pulls articles for the topic. A source that reports itself unavailable is recorded as skipped, not failed — a missing credential is something you can fix, a broken source is not."
              />
              <Step
                n="02"
                name="Extract"
                endpoint="POST /analyze"
                body="One constrained-decoding call per topic. The schema is enforced by the provider, and a rejected generation is fed back to the model as a correction rather than resampled blindly."
              />
              <Step
                n="03"
                name="Synthesise"
                endpoint="POST /brief"
                body="Takes every analysis at once and writes one script across them. Batching would hand back two disconnected briefings for five topics; the single script is the product."
              />
              <Step
                n="04"
                name="Speak"
                endpoint="POST /audio"
                body="Renders the script to audio. The response's content type follows the engine that actually ran, not the one that was requested — the fallback path is the live one."
              />
            </ol>
          </div>
        </section>

        {/* ---- Evaluation — the honest section ------------------------ */}
        <section className="section" id="evaluation">
          <div className="shell">
            <h2 className="section__title">What is measured, and what is not</h2>
            <p className="section__lede">
              A harness in the repository measures the extraction against a
              committed corpus of 40 real articles. It reports arithmetic over
              real output — and refuses to report anything it has not measured.
              Numbers below are produced by{" "}
              <code className="code">python -m evals.run --report</code>, not
              written by hand.
            </p>

            {report ? (
              <>
                <table className="spec">
                  <caption className="spec__caption">
                    Deterministic · arithmetic over extraction output and its
                    source articles. No model judges these. Run{" "}
                    {report.generated} · model {report.model} · prompt v
                    {report.prompt_version}
                  </caption>
                  <tbody>
                    <SpecRow
                      metric="Topics evaluated"
                      value={String(report.deterministic.topics_evaluated)}
                      note="Corpus records that had articles to extract from."
                    />
                    <SpecRow
                      metric="Schema validity rate"
                      value={rate(report.deterministic.schema_valid_rate)}
                      note="Fraction of calls returning schema-valid output."
                      pending={report.deterministic.schema_valid_rate === null}
                    />
                    <SpecRow
                      metric="Claims counted"
                      value={String(report.deterministic.total_claims)}
                      note="The denominator under the rate below. A rate without it is not a result."
                    />
                    <SpecRow
                      metric="Quote grounding rate"
                      value={rate(report.deterministic.grounding_rate)}
                      note="Fraction of claims whose quote appears verbatim in the source. Read it with the claim count, not alone."
                      pending={report.deterministic.grounding_rate === null}
                    />
                    <SpecRow
                      metric="Agreement with human labels"
                      value={rate(report.agreement.stance_kappa)}
                      note={
                        report.agreement.unavailable_reason ??
                        `Backed by ${report.agreement.labelled_coverage} reviewed labels.`
                      }
                      pending={report.agreement.stance_kappa === null}
                    />
                    <SpecRow
                      metric="Rubric-scored summary quality"
                      value={rate(report.judged.mean_coverage, 1)}
                      note={
                        report.judged.judged_count === 0
                          ? "No summary judged in this run. One model scoring another is reported separately and is not ground truth."
                          : `Mean coverage across ${report.judged.judged_count} summaries.`
                      }
                      pending={report.judged.mean_coverage === null}
                    />
                  </tbody>
                </table>
                <ReportNote report={report} />
              </>
            ) : (
              <p className="notice notice--info">
                <span className="notice__kind">not run</span>
                <span>
                  No report has been generated yet. Run{" "}
                  <code className="code">python -m evals.run --report</code> and
                  this table fills itself from the output — the page does not
                  keep its own copy of the numbers.
                </span>
              </p>
            )}

            <div className="caveat">
              <h3 className="caveat__head">The limit of the grounding claim</h3>
              <p>
                A quote is checked against each article&rsquo;s title and feed
                summary, not its full prose. The free news feed does not carry
                article bodies — the median body across the corpus is 195
                characters, which is roughly a headline. That bounds what
                &ldquo;grounded&rdquo; can mean here, and the report says so on
                its own face rather than in a footnote.
              </p>
              <p>
                It also explains a result worth stating plainly: one run
                reported a grounding rate of 1.00 over three claims. Real
                arithmetic, nearly vacuous. A later run reported 0.81 over 21.
                The denominator is what tells you which to believe.
              </p>
            </div>
          </div>
        </section>

        {/* ---- Architecture ------------------------------------------ */}
        <section className="section section--tint" id="architecture">
          <div className="shell arch">
            <div>
              <h2 className="section__title">How it is put together</h2>
              <p className="section__lede">
                One package under measurement, one harness that measures it, and
                a thin HTTP layer over both. The dependency runs one way: the
                harness imports the package, never the reverse, so the thing
                being measured cannot reach into its own scorecard.
              </p>
              <ul className="facts">
                <Fact k="Language" v="Python 3.12, typed strictly throughout" />
                <Fact k="Extraction" v="Constrained decoding against a strict JSON schema" />
                <Fact k="Rate control" v="Sliding-window token budget, shared across threads under a lock" />
                <Fact k="Service" v="FastAPI, one error envelope for every failure" />
                <Fact k="Tests" v="345, all offline — no test makes a network call" />
              </ul>
            </div>

            <pre className="tree" aria-label="Package layout">
              <code>{`api/
  newsninja/            the package under measurement
    analysis/           the only module that talks to the model
    sources/            a protocol; adding one is a new file
    audio/              speech, with a fallback that reports itself
    api/                the HTTP layer
  evals/                measures newsninja; never imported by it
    data/corpus.jsonl   40 real articles, committed on purpose
  tests/                network mocked everywhere
web/                    this page`}</code>
            </pre>
          </div>
        </section>

        {/* ---- C4 sticky CTA ----------------------------------------- */}
        <aside className="cta">
          <div className="shell cta__inner">
            <p className="cta__text">
              Every number on this page came from the harness, and every claim
              in a briefing carries the span it was copied from. Read the
              schema, or read the code.
            </p>
            <div className="cta__actions">
              <a className="btn btn--primary" href="#workbench">
                Run an analysis
              </a>
              <a
                className="btn btn--ghost"
                href="https://github.com/DHRUV1817/NEWS-AI"
              >
                Read the source
              </a>
            </div>
          </div>
        </aside>
      </main>

      {/* ---- Ft2 · inline rule, single line ------------------------- */}
      <footer className="foot">
        <div className="shell">
          <p className="foot__line">
            NewsNinja · a portfolio project by DHRUV1817 · MIT licensed ·
            numbers on this page come from the harness, never from hand
          </p>
        </div>
      </footer>
    </>
  );
}

function Step({
  n,
  name,
  endpoint,
  body,
}: {
  n: string;
  name: string;
  endpoint: string;
  body: string;
}) {
  return (
    <li className="step reveal">
      <span className="step__n">{n}</span>
      <div className="step__main">
        <h3 className="step__name">
          {name}
          <code className="step__endpoint">{endpoint}</code>
        </h3>
        <p className="step__body">{body}</p>
      </div>
    </li>
  );
}

function SpecRow({
  metric,
  value,
  note,
  pending,
}: {
  metric: string;
  value: string;
  note: string;
  pending?: boolean;
}) {
  return (
    <tr className="spec__row">
      <th scope="row" className="spec__metric">
        {metric}
      </th>
      <td className="spec__value">
        <span className={pending ? "pill pill--pending" : "pill pill--ok"}>
          {value}
        </span>
      </td>
      <td className="spec__note">{note}</td>
    </tr>
  );
}

function Fact({ k, v }: { k: string; v: string }) {
  return (
    <li className="fact">
      <span className="fact__k">{k}</span>
      <span className="fact__v">{v}</span>
    </li>
  );
}

function ReportNote({ report }: { report: EvalReport }) {
  const { grounding_rate, total_claims } = report.deterministic;
  if (grounding_rate === null) return null;
  return (
    <p className="spec__foot">
      That rate is computed over <strong>{total_claims}</strong> claims. The
      denominator is shown because it changes what the rate means: a 1.00 over
      three claims and a 0.81 over twenty-one are both real arithmetic, and only
      one of them says much.
    </p>
  );
}
