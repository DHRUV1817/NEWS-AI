"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  analyze,
  audio,
  brief,
  LANGUAGES,
  MAX_TOPICS,
  type AnalyzeResponse,
  type Briefing,
} from "@/lib/api";
import {
  addTopic as addTopicTo,
  runBriefing,
  type Failure,
  type Phase,
  type TopicRow,
} from "@/lib/briefing";

/* One topic per /analyze call, then every analysis handed to /brief at once.
 * That shape is the product: build_briefing writes across the analyses, so
 * splitting them into separate briefs would return disconnected scripts rather
 * than one briefing.
 *
 * Topics run one at a time rather than in parallel. The token budget is shared
 * and per-minute, so firing five at once would not finish sooner — it would
 * just collide, and the failures would be less legible. */

const STARTERS = [
  "artificial intelligence",
  "renewable energy",
  "space exploration",
  "cryptocurrency",
  "climate change",
];

export default function Analyzer() {
  const [topics, setTopics] = useState<string[]>(["artificial intelligence"]);
  const [draft, setDraft] = useState("");
  const [language, setLanguage] = useState("en");

  const [phase, setPhase] = useState<Phase>("idle");
  const [rows, setRows] = useState<TopicRow[]>([]);
  const [briefing, setBriefing] = useState<Briefing | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [fatal, setFatal] = useState<Failure | null>(null);

  const objectUrl = useRef<string | null>(null);

  useEffect(() => {
    return () => {
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    };
  }, []);

  const addTopic = useCallback(
    (value: string) => {
      const trimmed = value.trim();
      if (!trimmed) return;
      setTopics((current) => addTopicTo(current, trimmed).topics);
      setDraft("");
    },
    [],
  );

  const removeTopic = useCallback((value: string) => {
    setTopics((current) => current.filter((t) => t !== value));
  }, []);

  const run = useCallback(async () => {
    if (topics.length === 0) return;

    setFatal(null);
    setBriefing(null);
    if (objectUrl.current) {
      URL.revokeObjectURL(objectUrl.current);
      objectUrl.current = null;
    }
    setAudioUrl(null);

    const outcome = await runBriefing(
      topics,
      language,
      { analyze, brief, audio },
      (progress, current) => {
        setRows(progress);
        setPhase(current);
      },
    );

    setRows(outcome.rows);
    setBriefing(outcome.briefing);
    setFatal(outcome.failure);

    if (outcome.audio) {
      const url = URL.createObjectURL(outcome.audio.blob);
      objectUrl.current = url;
      setAudioUrl(url);
      setPhase("done");
    } else {
      setPhase("idle");
    }
  }, [topics, language]);

  const busy = phase !== "idle" && phase !== "done";
  const succeeded = rows.filter((r) => r.status === "done").length;
  const failed = rows.filter((r) => r.status === "failed").length;

  return (
    <div className="analyzer">
      <form
        className="analyzer__form"
        onSubmit={(event) => {
          event.preventDefault();
          void run();
        }}
      >
        <div className="analyzer__topbar">
          <label className="analyzer__label" htmlFor="topic">
            Topics
            <span className="analyzer__count">
              {topics.length}/{MAX_TOPICS}
            </span>
          </label>
          <label className="analyzer__label" htmlFor="language">
            Language
          </label>
        </div>

        <div className="analyzer__row">
          <input
            id="topic"
            className="analyzer__input"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                addTopic(draft);
              }
            }}
            placeholder={
              topics.length >= MAX_TOPICS
                ? `${MAX_TOPICS} is the maximum`
                : "Add a topic, then press Enter"
            }
            maxLength={200}
            disabled={busy || topics.length >= MAX_TOPICS}
            autoComplete="off"
            spellCheck={false}
          />
          <select
            id="language"
            className="analyzer__select"
            value={language}
            onChange={(event) => setLanguage(event.target.value)}
            disabled={busy}
          >
            {LANGUAGES.map((l) => (
              <option key={l.code} value={l.code}>
                {l.label}
              </option>
            ))}
          </select>
        </div>

        {topics.length > 0 && (
          <ul className="picked">
            {topics.map((topic) => (
              <li key={topic} className="picked__item">
                {topic}
                <button
                  type="button"
                  className="picked__remove"
                  onClick={() => removeTopic(topic)}
                  disabled={busy}
                  aria-label={`Remove ${topic}`}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        )}

        <div className="analyzer__suggest">
          {STARTERS.filter((s) => !topics.includes(s)).map((s) => (
            <button
              key={s}
              type="button"
              className="chip"
              disabled={busy || topics.length >= MAX_TOPICS}
              onClick={() => addTopic(s)}
            >
              + {s}
            </button>
          ))}
        </div>

        <button
          className="btn btn--primary analyzer__go"
          type="submit"
          disabled={busy || topics.length === 0}
          data-state={busy ? "loading" : undefined}
        >
          {busy
            ? "Working…"
            : `Build briefing from ${topics.length} topic${topics.length === 1 ? "" : "s"}`}
        </button>

        {topics.length > 3 && !busy && (
          <p className="analyzer__warn">
            Four or more topics reserve more than the per-minute token budget
            allows. Expect a later one to be refused — the briefing is built
            from whatever succeeds.
          </p>
        )}
      </form>

      <ol className="stages" aria-live="polite">
        <Stagelet
          label="Extract"
          active={phase === "analyzing"}
          done={succeeded > 0 && phase !== "analyzing"}
          note={rows.length ? `${succeeded}/${rows.length}` : `${topics.length} calls`}
        />
        <Stagelet
          label="Synthesise"
          active={phase === "briefing"}
          done={!!briefing}
          note="one script, all topics"
        />
        <Stagelet
          label="Speak"
          active={phase === "speaking"}
          done={!!audioUrl}
          note="no model call"
        />
      </ol>

      {rows.length > 0 && (
        <ul className="runlist">
          {rows.map((row) => (
            <li key={row.topic} className="runrow" data-status={row.status}>
              <span className="runrow__dot" aria-hidden="true" />
              <span className="runrow__topic">{row.topic}</span>
              <span className="runrow__note">
                {row.status === "queued" && "queued"}
                {row.status === "running" && "running…"}
                {row.status === "done" &&
                  `${row.result!.analysis.key_claims.length} claim${
                    row.result!.analysis.key_claims.length === 1 ? "" : "s"
                  }`}
                {row.status === "failed" && (
                  <>
                    {row.error!.kind}
                    {row.error!.retryAfter
                      ? ` · retry in ~${Math.ceil(row.error!.retryAfter)}s`
                      : ""}
                  </>
                )}
              </span>
            </li>
          ))}
        </ul>
      )}

      {failed > 0 && succeeded > 0 && phase === "done" && (
        <p className="notice notice--info">
          <span className="notice__kind">partial</span>
          <span>
            {succeeded} of {rows.length} topics made it into the briefing.
            A refused topic is a budget limit, not a broken service.
          </span>
        </p>
      )}

      {fatal && (
        <div className="notice notice--error" role="alert">
          <span className="notice__kind">{fatal.kind}</span>
          <p>{fatal.message}</p>
          {fatal.retryAfter !== undefined && (
            <p className="notice__retry">
              Budget frees in about <strong>{Math.ceil(fatal.retryAfter)}s</strong>.
              The whole service supports roughly three analyses per minute — the
              free-tier token ceiling, not a queue.
            </p>
          )}
        </div>
      )}

      {rows.some((r) => r.result) && (
        <div className="result">
          {rows
            .filter((row) => row.result)
            .map((row) => (
              <TopicResult key={row.topic} data={row.result!} />
            ))}

          {briefing && (
            <div className="script">
              <h4 className="script__head">
                Briefing script
                <span className="script__topics">
                  {briefing.topics.join(" · ")} · {briefing.language}
                </span>
              </h4>
              <p className="script__body">{briefing.script}</p>
            </div>
          )}

          {audioUrl && (
            <div className="player">
              <audio controls src={audioUrl} className="player__el" />
              <a className="link" href={audioUrl} download="briefing.mp3">
                Download
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function TopicResult({ data }: { data: AnalyzeResponse }) {
  const { analysis, skipped_sources, source_errors, article_count } = data;
  // No articles means no model call: the stance and confidence below are the
  // placeholder's defaults, not findings, so the page must not present them as
  // a judgement about the topic.
  const measured = article_count > 0;
  return (
    <div className="topic">
      <div className="result__head">
        <h3 className="result__topic">{analysis.topic}</h3>
        <span className="tag">
          {measured
            ? `${analysis.stance} · confidence ${analysis.confidence.toFixed(2)} · ${article_count} articles`
            : "no articles found · nothing was analysed"}
        </span>
      </div>

      <p className="result__summary">{analysis.summary}</p>

      {analysis.entities.length > 0 && (
        <ul className="entities">
          {analysis.entities.map((entity) => (
            <li key={`${entity.kind}-${entity.name}`} className="entity">
              {entity.name}
              <span className="entity__kind">{entity.kind}</span>
            </li>
          ))}
        </ul>
      )}

      <div className="claims">
        <h4 className="claims__head">
          Claims<span className="claims__count">{analysis.key_claims.length}</span>
        </h4>
        {analysis.key_claims.length === 0 ? (
          <p className="claims__empty">
            No claim carried a span the model could quote verbatim, so it
            returned none. The feed supplies headlines, not article prose.
          </p>
        ) : (
          <ul className="claims__list">
            {analysis.key_claims.map((claim) => (
              <li key={claim.quote} className="claim">
                <p className="claim__text">{claim.text}</p>
                <blockquote className="claim__quote">{claim.quote}</blockquote>
              </li>
            ))}
          </ul>
        )}
      </div>

      {skipped_sources.map((name) => (
        <p key={name} className="notice notice--info">
          <span className="notice__kind">skipped</span>
          <span>
            <strong>{name}</strong> reported itself unavailable and was never
            tried — a missing credential, not a failure.
          </span>
        </p>
      ))}
      {Object.entries(source_errors).map(([name, messages]) => (
        <p key={name} className="notice notice--warn">
          <span className="notice__kind">failed</span>
          <span>
            <strong>{name}</strong> was tried and broke: {messages.join("; ")}
          </span>
        </p>
      ))}
    </div>
  );
}

function Stagelet({
  label,
  active,
  done,
  note,
}: {
  label: string;
  active: boolean;
  done: boolean;
  note: string;
}) {
  return (
    <li className="stage" data-active={active || undefined} data-done={done || undefined}>
      <span className="stage__dot" aria-hidden="true" />
      <span className="stage__label">{label}</span>
      <span className="stage__note">{note}</span>
    </li>
  );
}
