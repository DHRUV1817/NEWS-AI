import { describe, expect, it, vi } from "vitest";

import { ApiError, NetworkError, type ArticleAnalysis } from "./api";
import { addTopic, describeFailure, runBriefing, type RunDeps } from "./briefing";

function analysis(topic: string): ArticleAnalysis {
  return {
    topic,
    summary: `about ${topic}`,
    entities: [],
    stance: "neutral",
    confidence: 0.5,
    key_claims: [],
  };
}

function analyzed(topic: string, articleCount = 6) {
  return {
    analysis: analysis(topic),
    source_errors: {},
    skipped_sources: [],
    article_count: articleCount,
  };
}

function deps(over: Partial<RunDeps> = {}): RunDeps {
  return {
    analyze: async (topic) => analyzed(topic),
    brief: async (analyses, language) => ({
      briefing: {
        topics: analyses.map((a) => a.topic),
        script: "one script",
        analyses,
        language,
      },
    }),
    audio: async () => ({ blob: new Blob(["x"]), mediaType: "audio/mpeg" }),
    ...over,
  };
}

describe("addTopic", () => {
  it("adds a trimmed topic", () => {
    expect(addTopic([], "  climate change ")).toEqual({
      topics: ["climate change"],
      rejected: null,
    });
  });

  it("refuses a duplicate regardless of case, so no call is spent twice", () => {
    const { topics, rejected } = addTopic(["Climate Change"], "climate change");
    expect(rejected).toBe("duplicate");
    expect(topics).toEqual(["Climate Change"]);
  });

  it("refuses past the bound the service enforces", () => {
    const full = ["a", "b", "c", "d", "e"];
    expect(addTopic(full, "f")).toEqual({ topics: full, rejected: "full" });
  });

  it("refuses blank input rather than adding an empty topic", () => {
    expect(addTopic([], "   ").rejected).toBe("blank");
  });
});

describe("describeFailure", () => {
  it("keeps the kind and the retry the service reported", () => {
    const failure = describeFailure(
      new ApiError(429, {
        type: "rate_limit",
        message: "too many requests",
        retry_after: 28.9,
      }),
    );
    expect(failure).toEqual({
      kind: "rate_limit",
      message: "too many requests",
      retryAfter: 28.9,
    });
  });

  it("separates unreachable from anything the service said", () => {
    expect(describeFailure(new NetworkError("down")).kind).toBe("unreachable");
  });

  it("does not pretend to know what an unknown throw was", () => {
    expect(describeFailure("weird").kind).toBe("unknown");
  });
});

describe("runBriefing", () => {
  it("briefs across every topic that succeeded", async () => {
    const outcome = await runBriefing(["ai", "energy"], "en", deps());
    expect(outcome.failure).toBeNull();
    expect(outcome.briefing?.topics).toEqual(["ai", "energy"]);
    expect(outcome.rows.every((r) => r.status === "done")).toBe(true);
  });

  it("keeps going when a topic is refused, and briefs on the rest", async () => {
    // The reason this matters: five topics reserve more than a minute of token
    // budget allows, so a refusal partway through is the expected path.
    const outcome = await runBriefing(
      ["ai", "energy", "space"],
      "en",
      deps({
        analyze: async (topic) => {
          if (topic === "energy") {
            throw new ApiError(429, {
              type: "rate_limit",
              message: "budget full",
              retry_after: 30,
            });
          }
          return analyzed(topic);
        },
      }),
    );

    expect(outcome.failure).toBeNull();
    expect(outcome.briefing?.topics).toEqual(["ai", "space"]);
    expect(outcome.rows.map((r) => r.status)).toEqual(["done", "failed", "done"]);
    expect(outcome.rows[1].error?.retryAfter).toBe(30);
  });

  it("never sends an empty list to the service when everything failed", async () => {
    const brief = vi.fn();
    const outcome = await runBriefing(
      ["ai", "energy"],
      "en",
      deps({
        analyze: async () => {
          throw new NetworkError("down");
        },
        brief,
      }),
    );

    expect(brief).not.toHaveBeenCalled();
    expect(outcome.failure?.kind).toBe("nothing_to_brief");
    expect(outcome.briefing).toBeNull();
  });

  it("extracts one topic at a time rather than all at once", async () => {
    // Concurrency here would not finish sooner — the budget is shared and
    // per-minute — it would only collide and make the failures less legible.
    let inFlight = 0;
    let peak = 0;
    await runBriefing(
      ["a", "b", "c"],
      "en",
      deps({
        analyze: async (topic) => {
          inFlight += 1;
          peak = Math.max(peak, inFlight);
          await new Promise((r) => setTimeout(r, 1));
          inFlight -= 1;
          return analyzed(topic);
        },
      }),
    );
    expect(peak).toBe(1);
  });

  it("reports a failure at the briefing step without losing the analyses", async () => {
    const outcome = await runBriefing(
      ["ai"],
      "en",
      deps({
        brief: async () => {
          throw new ApiError(502, {
            type: "extraction_failure",
            message: "model would not comply",
          });
        },
      }),
    );
    expect(outcome.failure?.kind).toBe("extraction_failure");
    expect(outcome.rows[0].status).toBe("done");
  });

  it("passes the chosen language through to both later calls", async () => {
    const brief = vi.fn(async (analyses: ArticleAnalysis[], language: string) => ({
      briefing: { topics: ["ai"], script: "s", analyses, language },
    }));
    const audio = vi.fn(async () => ({
      blob: new Blob(["x"]),
      mediaType: "audio/mpeg",
    }));

    await runBriefing(["ai"], "fr", deps({ brief, audio }));

    expect(brief).toHaveBeenCalledWith(expect.anything(), "fr");
    expect(audio).toHaveBeenCalledWith("s", "fr");
  });

  it("walks the phases in order so the page can show where it is", async () => {
    const phases: string[] = [];
    await runBriefing(["ai"], "en", deps(), (_rows, phase) => {
      if (phases[phases.length - 1] !== phase) phases.push(phase);
    });
    expect(phases).toEqual(["analyzing", "briefing", "speaking", "done"]);
  });
});
