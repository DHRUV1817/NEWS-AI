/* The service answers with one error envelope for every failure, so the client
 * parses one shape. Anything it cannot parse is reported as-is rather than
 * flattened into "something went wrong" — a UI that hides which source broke,
 * or how long to wait, throws away the thing the API went to trouble to say. */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

/** The server bounds /brief to this many analyses and rejects more with a 422.
 *  Mirrored here so the UI can stop you before the round trip — the server is
 *  still the authority, this is only a courtesy. */
export const MAX_TOPICS = 5;

/** The speech engine rewrites any code outside this set to English, so the
 *  service refuses them at the edge rather than returning a Swedish script
 *  spoken in English with nothing saying so. */
export const LANGUAGES: { code: string; label: string }[] = [
  { code: "en", label: "English" },
  { code: "es", label: "Spanish" },
  { code: "fr", label: "French" },
  { code: "de", label: "German" },
  { code: "it", label: "Italian" },
  { code: "pt", label: "Portuguese" },
  { code: "ru", label: "Russian" },
  { code: "ar", label: "Arabic" },
  { code: "hi", label: "Hindi" },
  { code: "ja", label: "Japanese" },
  { code: "ko", label: "Korean" },
  { code: "zh", label: "Chinese" },
];

export type EntityKind = "person" | "org" | "place" | "product" | "other";
export type Stance = "positive" | "negative" | "neutral";

export interface Entity {
  name: string;
  kind: EntityKind;
}

export interface Claim {
  /** The assertion. */
  text: string;
  /** Copied character-for-character from the source. This is what makes
   *  hallucination a substring check rather than an opinion. */
  quote: string;
}

export interface ArticleAnalysis {
  topic: string;
  summary: string;
  entities: Entity[];
  stance: Stance;
  confidence: number;
  key_claims: Claim[];
}

export interface AnalyzeResponse {
  analysis: ArticleAnalysis;
  /** Sources that were tried and broke, by name. */
  source_errors: Record<string, string[]>;
  /** Sources that reported themselves unavailable and were never tried.
   *  Deliberately separate from source_errors: a missing credential is
   *  something the caller can fix, a broken source is not. */
  skipped_sources: string[];
}

export interface Briefing {
  topics: string[];
  script: string;
  analyses: ArticleAnalysis[];
  language: string;
}

export interface BriefResponse {
  briefing: Briefing;
}

export interface ApiErrorBody {
  type: string;
  message: string;
  retry_after?: number;
  detail?: unknown;
}

export class ApiError extends Error {
  readonly status: number;
  readonly kind: string;
  /** Seconds until the budget frees. Present only on a rate limit. */
  readonly retryAfter?: number;

  constructor(status: number, body: ApiErrorBody) {
    super(body.message);
    this.name = "ApiError";
    this.status = status;
    this.kind = body.type;
    this.retryAfter = body.retry_after;
  }
}

/** The service is unreachable — a different failure from one it reported. */
export class NetworkError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "NetworkError";
  }
}

async function post<T>(path: string, payload: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new NetworkError(
      `Could not reach the service at ${API_BASE}. Start it with: uv run --python 3.12 uvicorn newsninja.api:create_app --factory`,
    );
  }

  if (!response.ok) {
    let body: ApiErrorBody = {
      type: "unknown",
      message: `${response.status} ${response.statusText}`,
    };
    try {
      const parsed = (await response.json()) as { error?: ApiErrorBody };
      if (parsed?.error) body = parsed.error;
    } catch {
      /* Body was not the envelope. The status line above stands. */
    }
    throw new ApiError(response.status, body);
  }

  return (await response.json()) as T;
}

export function analyze(topic: string, limit = 6): Promise<AnalyzeResponse> {
  return post<AnalyzeResponse>("/analyze", { topic, limit });
}

export function brief(
  analyses: ArticleAnalysis[],
  language = "en",
): Promise<BriefResponse> {
  return post<BriefResponse>("/brief", { analyses, language });
}

/** Returns the audio bytes and the media type the service actually produced —
 *  the header follows the engine that ran, not the one that was requested. */
export async function audio(
  script: string,
  language = "en",
): Promise<{ blob: Blob; mediaType: string }> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/audio`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ script, language }),
    });
  } catch {
    throw new NetworkError(`Could not reach the service at ${API_BASE}.`);
  }

  if (!response.ok) {
    let body: ApiErrorBody = {
      type: "unknown",
      message: `${response.status} ${response.statusText}`,
    };
    try {
      const parsed = (await response.json()) as { error?: ApiErrorBody };
      if (parsed?.error) body = parsed.error;
    } catch {
      /* Not the envelope. */
    }
    throw new ApiError(response.status, body);
  }

  return {
    blob: await response.blob(),
    mediaType: response.headers.get("content-type") ?? "audio/mpeg",
  };
}
