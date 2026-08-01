import { readFile } from "node:fs/promises";
import path from "node:path";

/* Read at build time from the file the harness writes. The page shows numbers
 * it did not compute, so it reads the harness's own output rather than keeping
 * a copy — a hand-maintained duplicate is how a page ends up quoting a figure
 * that no run ever produced.
 *
 * Anything the harness could not compute arrives as null and renders as
 * unavailable. It must never be coerced to zero on the way through: a zero
 * reads as a measured result. */

export interface Deterministic {
  topics_evaluated: number;
  schema_valid_rate: number | null;
  grounding_rate: number | null;
  ungrounded_claim_rate: number | null;
  total_claims: number;
  total_ungrounded_claims: number;
  mean_claims_per_topic: number | null;
  mean_entities_per_topic: number | null;
}

export interface Agreement {
  labelled_coverage: number;
  total_labels: number;
  entity_precision: number | null;
  entity_recall: number | null;
  entity_f1: number | null;
  stance_accuracy: number | null;
  stance_kappa: number | null;
  unavailable_reason: string | null;
}

export interface Judged {
  judged_count: number;
  mean_coverage: number | null;
  mean_neutrality: number | null;
  mean_coherence: number | null;
}

export interface EvalReport {
  generated: string | null;
  model: string | null;
  prompt_version: string | null;
  skipped_records: number;
  deterministic: Deterministic;
  agreement: Agreement;
  judged: Judged;
}

const REPORT_PATH = path.join(process.cwd(), "..", "docs", "evals", "latest.json");

/** Null when the harness has not been run. The page says so plainly rather
 *  than filling the table with placeholder figures. */
export async function loadEvalReport(): Promise<EvalReport | null> {
  try {
    const raw = await readFile(REPORT_PATH, "utf8");
    return JSON.parse(raw) as EvalReport;
  } catch {
    return null;
  }
}

/** Formats a rate, or the word the project uses for a number it does not have. */
export function rate(value: number | null, digits = 2): string {
  return value === null ? "unavailable" : value.toFixed(digits);
}
