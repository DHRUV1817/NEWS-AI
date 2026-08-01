import {
  ApiError,
  NetworkError,
  MAX_TOPICS,
  type AnalyzeResponse,
  type ArticleAnalysis,
  type BriefResponse,
  type Briefing,
} from "./api";

/* The run's orchestration, kept out of the component so it can be exercised
 * without a browser. What is worth testing here is not what the page looks
 * like: it is that a refused topic does not take the run down with it, that the
 * briefing is built from whatever survived, and that a run where everything
 * failed is reported as such rather than sending an empty list to /brief. */

export type TopicStatus = "queued" | "running" | "done" | "failed";
export type Phase = "idle" | "analyzing" | "briefing" | "speaking" | "done";

export interface Failure {
  kind: string;
  message: string;
  retryAfter?: number;
}

export interface TopicRow {
  topic: string;
  status: TopicStatus;
  result?: AnalyzeResponse;
  error?: Failure;
}

export interface RunDeps {
  analyze: (topic: string) => Promise<AnalyzeResponse>;
  brief: (analyses: ArticleAnalysis[], language: string) => Promise<BriefResponse>;
  audio: (
    script: string,
    language: string,
  ) => Promise<{ blob: Blob; mediaType: string }>;
}

export interface RunOutcome {
  rows: TopicRow[];
  briefing: Briefing | null;
  audio: { blob: Blob; mediaType: string } | null;
  failure: Failure | null;
}

/** Classify anything thrown into something the page can render.
 *
 *  A service that reported a rate limit and a service that could not be
 *  reached are different facts with different remedies, and flattening both
 *  into "something went wrong" throws away the one the API took trouble to
 *  say. */
export function describeFailure(error: unknown): Failure {
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
}

/** Add a topic, or return the list unchanged and say why not.
 *
 *  Returning a reason rather than silently doing nothing is what lets the page
 *  explain itself; a control that ignores a click looks broken. */
export function addTopic(
  current: string[],
  value: string,
): { topics: string[]; rejected: "blank" | "duplicate" | "full" | null } {
  const trimmed = value.trim();
  if (!trimmed) return { topics: current, rejected: "blank" };
  if (current.length >= MAX_TOPICS) return { topics: current, rejected: "full" };
  // A duplicate would spend a model call to say the same thing twice.
  if (current.some((t) => t.toLowerCase() === trimmed.toLowerCase())) {
    return { topics: current, rejected: "duplicate" };
  }
  return { topics: [...current, trimmed], rejected: null };
}

/** Extract every topic, then build one briefing across whatever survived.
 *
 *  Topics run one at a time on purpose. The token budget is shared and
 *  per-minute, so firing them together would not finish sooner — it would
 *  collide, and the failures would be harder to read. */
export async function runBriefing(
  topics: string[],
  language: string,
  deps: RunDeps,
  onProgress: (rows: TopicRow[], phase: Phase) => void = () => {},
): Promise<RunOutcome> {
  const rows: TopicRow[] = topics.map((topic) => ({ topic, status: "queued" }));
  onProgress([...rows], "analyzing");

  for (let i = 0; i < rows.length; i += 1) {
    rows[i] = { ...rows[i], status: "running" };
    onProgress([...rows], "analyzing");
    try {
      rows[i] = { ...rows[i], status: "done", result: await deps.analyze(rows[i].topic) };
    } catch (error) {
      rows[i] = { ...rows[i], status: "failed", error: describeFailure(error) };
    }
    onProgress([...rows], "analyzing");
  }

  const analyses = rows
    .map((row) => row.result?.analysis)
    .filter((a): a is ArticleAnalysis => a !== undefined);

  if (analyses.length === 0) {
    // Sending an empty list would earn a 422 from the schema bound and report
    // it as a validation error, which describes the request rather than what
    // actually happened.
    onProgress([...rows], "idle");
    return {
      rows,
      briefing: null,
      audio: null,
      failure: {
        kind: "nothing_to_brief",
        message:
          "Every topic failed, so there is nothing to synthesise. The per-topic errors are listed above.",
      },
    };
  }

  try {
    onProgress([...rows], "briefing");
    const briefed = await deps.brief(analyses, language);

    onProgress([...rows], "speaking");
    const spoken = await deps.audio(
      briefed.briefing.script,
      briefed.briefing.language,
    );

    onProgress([...rows], "done");
    return { rows, briefing: briefed.briefing, audio: spoken, failure: null };
  } catch (error) {
    onProgress([...rows], "idle");
    return {
      rows,
      briefing: null,
      audio: null,
      failure: describeFailure(error),
    };
  }
}
