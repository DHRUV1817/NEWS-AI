"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  analyze,
  audio,
  brief,
  ApiError,
  NetworkError,
  type AnalyzeResponse,
  type Briefing,
} from "@/lib/api";

/* The three calls are separate because the token budget makes them separate.
 * One topic reserves ~2,400 tokens against an 8,000-per-minute ceiling, so a
 * request that did all five topics at once would sit inside the limiter long
 * enough for a platform proxy to sever it. Showing the stages is honest about
 * that shape rather than hiding it behind one spinner. */
type Stage = "idle" | "analyzing" | "briefing" | "speaking" | "done";

const SUGGESTIONS = [
  "artificial intelligence",
  "renewable energy",
  "space exploration",
  "cryptocurrency",
];

interface Failure {
  kind: string;
  message: string;
  retryAfter?: number;
}

export default function Analyzer() {
  const [topic, setTopic] = useState("artificial intelligence");
  const [stage, setStage] = useState<Stage>("idle");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [briefing, setBriefing] = useState<Briefing | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [failure, setFailure] = useState<Failure | null>(null);
  const [retryIn, setRetryIn] = useState<number | null>(null);
  const objectUrl = useRef<string | null>(null);

  // Revoke the previous blob before replacing it; a run per topic would
  // otherwise leak one object URL per run for the life of the tab.
  useEffect(() => {
    return () => {
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    };
  }, []);

  const describe = useCallback((error: unknown): Failure => {
    if (error instanceof ApiError) {
      return {
        kind: error.kind,
        message: error.message,
        retryAfter: error.retryAfter,
      };
    }
    if (error instanceof NetworkError) {
      return { kind: "unreachable", message: error.message };
    }
    return {
      kind: "unknown",
      message: error instanceof Error ? error.message : String(error),
    };
  }, []);

  const run = useCallback(async () => {
    const trimmed = topic.trim();
    if (!trimmed) return;

    setFailure(null);
    setRetryIn(null);
    setResult(null);
    setBriefing(null);
    if (objectUrl.current) {
      URL.revokeObjectURL(objectUrl.current);
      objectUrl.current = null;
    }
    setAudioUrl(null);

    try {
      setStage("analyzing");
      const analyzed = await analyze(trimmed);
      setResult(analyzed);

      setStage("briefing");
      const briefed = await brief([analyzed.analysis]);
      setBriefing(briefed.briefing);

      setStage("speaking");
      const spoken = await audio(briefed.briefing.script, briefed.briefing.language);
      const url = URL.createObjectURL(spoken.blob);
      objectUrl.current = url;
      setAudioUrl(url);

      setStage("done");
    } catch (error) {
      const described = describe(error);
      setFailure(described);
      setRetryIn(described.retryAfter ? Math.ceil(described.retryAfter) : null);
      setStage("idle");
    }
  }, [topic, describe]);

  const busy = stage !== "idle" && stage !== "done";

  return (
    <div className="analyzer">
      <form
        className="analyzer__form"
        onSubmit={(event) => {
          event.preventDefault();
          void run();
        }}
      >
        <label className="analyzer__label" htmlFor="topic">
          Topic
        </label>
        <div className="analyzer__row">
          <input
            id="topic"
            className="analyzer__input"
            value={topic}
            onChange={(event) => setTopic(event.target.value)}
            placeholder="artificial intelligence"
            maxLength={200}
            disabled={busy}
            autoComplete="off"
            spellCheck={false}
          />
          <button
            className="btn btn--primary"
            type="submit"
            disabled={busy || !topic.trim()}
            data-state={busy ? "loading" : undefined}
          >
            {busy ? "Working…" : "Run analysis"}
          </button>
        </div>
        <div className="analyzer__suggest">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              type="button"
              className="chip"
              disabled={busy}
              onClick={() => setTopic(s)}
            >
              {s}
            </button>
          ))}
        </div>
      </form>

      <ol className="stages" aria-live="polite">
        <Stagelet label="Fetch + extract" active={stage === "analyzing"} done={!!result} note="1 model call" />
        <Stagelet label="Synthesise script" active={stage === "briefing"} done={!!briefing} note="1–2 model calls" />
        <Stagelet label="Render speech" active={stage === "speaking"} done={!!audioUrl} note="no model call" />
      </ol>

      {failure && (
        <div className="notice notice--error" role="alert">
          <span className="notice__kind">{failure.kind}</span>
          <p>{failure.message}</p>
          {retryIn !== null && (
            <p className="notice__retry">
              Budget frees in about <strong>{retryIn}s</strong>. The whole
              service supports roughly three analyses per minute — that is the
              free-tier token ceiling, not a queue.
            </p>
          )}
        </div>
      )}

      {result && (
        <div className="result">
          <div className="result__head">
            <h3 className="result__topic">{result.analysis.topic}</h3>
            <span className="tag">
              {result.analysis.stance} · confidence{" "}
              {result.analysis.confidence.toFixed(2)}
            </span>
          </div>

          <p className="result__summary">{result.analysis.summary}</p>

          {result.analysis.entities.length > 0 && (
            <ul className="entities">
              {result.analysis.entities.map((entity) => (
                <li key={`${entity.kind}-${entity.name}`} className="entity">
                  {entity.name}
                  <span className="entity__kind">{entity.kind}</span>
                </li>
              ))}
            </ul>
          )}

          <div className="claims">
            <h4 className="claims__head">
              Claims
              <span className="claims__count">
                {result.analysis.key_claims.length}
              </span>
            </h4>
            {result.analysis.key_claims.length === 0 ? (
              /* Zero claims is a result, not an empty state to apologise for.
                 The prompt tells the model to omit any claim it cannot quote
                 verbatim, and the feed carries headlines rather than prose. */
              <p className="claims__empty">
                No claim carried a span the model could quote verbatim, so it
                returned none. The source feed supplies headlines, not article
                text — see the measurement note below.
              </p>
            ) : (
              <ul className="claims__list">
                {result.analysis.key_claims.map((claim) => (
                  <li key={claim.quote} className="claim">
                    <p className="claim__text">{claim.text}</p>
                    <blockquote className="claim__quote">
                      {claim.quote}
                    </blockquote>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {(result.skipped_sources.length > 0 ||
            Object.keys(result.source_errors).length > 0) && (
            <div className="sources">
              {result.skipped_sources.map((name) => (
                <p key={name} className="notice notice--info">
                  <span className="notice__kind">skipped</span>
                  <span>
                    <strong>{name}</strong> reported itself unavailable and was
                    never tried — a missing credential, not a failure.
                  </span>
                </p>
              ))}
              {Object.entries(result.source_errors).map(([name, messages]) => (
                <p key={name} className="notice notice--warn">
                  <span className="notice__kind">failed</span>
                  <span>
                    <strong>{name}</strong> was tried and broke:{" "}
                    {messages.join("; ")}
                  </span>
                </p>
              ))}
            </div>
          )}

          {briefing && (
            <div className="script">
              <h4 className="script__head">Briefing script</h4>
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
    <li
      className="stage"
      data-active={active || undefined}
      data-done={done || undefined}
    >
      <span className="stage__dot" aria-hidden="true" />
      <span className="stage__label">{label}</span>
      <span className="stage__note">{note}</span>
    </li>
  );
}
