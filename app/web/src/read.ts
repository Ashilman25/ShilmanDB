// Turns the server's JSON into the page's types, checking every field on the way in,
// so a reply that drifted from the server's shape fails loudly instead of drawing wrong.
import type {
  Alternative, AnswerView, Asked, BarPoint, CacheStatus, Chart, CheckRow, Column, DatasetSchema, DatasetStatus,
  HistoryEntry, JobError, JobKind, LoadState, LockStatus, ModelState, ModelStatus, PlanRow, Preview, RowsPage, Schema,
  SchemaColumn, SchemaTable, Snapshot, Stage, Status, StopReply, TermRow, Tested, Timing, Verdict, VerdictKind,
} from "./types.js";

// A reply that doesn't have the shape the page expects.
export class ShapeError extends Error {}

export type Read<T> = (value: unknown, at: string) => T;

const STAGES: readonly Stage[] = ["queued", "writing", "first", "done", "stopped", "error"];
const KINDS: readonly JobKind[] = ["ask", "reask", "sql", "edit"];
const VERDICTS: readonly VerdictKind[] = ["pending", "clean", "warned", "refused", "cannot", "failed", "not_run", "ran", "stopped"];
const LOAD_STATES: readonly LoadState[] = ["loading", "ready", "missing", "failed"];
const MODEL_STATES: readonly ModelState[] = ["loading", "ready", "off", "failed"];
const CHART_KINDS: readonly Chart["kind"][] = ["bars", "number", "none"];

function fail(at: string, wanted: string): never {
  throw new ShapeError(`${at} should be ${wanted}`);
}

function isFields(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// The fields of an object, read one at a time with a reader for each.
function fields(value: unknown, at: string): <T>(key: string, read: Read<T>) => T {
  if (!isFields(value)) fail(at, "an object");
  return (key, read) => read(value[key], `${at}.${key}`);
}

const text: Read<string> = (v, at) => (typeof v === "string" ? v : fail(at, "text"));
const flag: Read<boolean> = (v, at) => (typeof v === "boolean" ? v : fail(at, "true or false"));
const number: Read<number> = (v, at) => (typeof v === "number" && Number.isFinite(v) ? v : fail(at, "a number"));

function maybe<T>(read: Read<T>): Read<T | null> {
  return (v, at) => (v === null ? null : read(v, at));
}

// A field the server leaves out when it has nothing to say; absent reads as the fallback.
function absent<T>(read: Read<T>, fallback: T): Read<T> {
  return (v, at) => (v === undefined ? fallback : read(v, at));
}

function list<T>(read: Read<T>): Read<T[]> {
  return (v, at) => {
    if (!Array.isArray(v)) fail(at, "a list");
    const items: unknown[] = v;
    return items.map((item, i) => read(item, `${at}[${i}]`));
  };
}

function oneOf<T extends string>(allowed: readonly T[]): Read<T> {
  return (v, at) => allowed.find((a) => a === v) ?? fail(at, `one of ${allowed.join(", ")}`);
}

// An object whose values all have one shape, keeping the server's key order.
function table<T>(read: Read<T>): Read<Record<string, T>> {
  return (v, at) => {
    if (!isFields(v)) fail(at, "an object");
    const out: Record<string, T> = {};
    for (const [key, item] of Object.entries(v)) out[key] = read(item, `${at}.${key}`);
    return out;
  };
}

const cells: Read<string[][]> = list(list(text));

const column: Read<Column> = (v, at) => {
  const f = fields(v, at);
  return { name: f("name", text), type: f("type", text) };
};

const barPoint: Read<BarPoint> = (v, at) => {
  if (!Array.isArray(v) || v.length !== 2) fail(at, "a label and a number");
  const pair: unknown[] = v;
  return [text(pair[0], `${at}[0]`), number(pair[1], `${at}[1]`)];
};

const chart: Read<Chart> = (v, at): Chart => {
  const f = fields(v, at);
  const kind = f("kind", oneOf(CHART_KINDS));
  if (kind === "bars") return { kind, label: f("label", number), value: f("value", number), points: f("points", list(barPoint)) };
  if (kind === "number") return { kind, value: f("value", number), number: f("number", number), text: f("text", text) };
  return { kind };
};

const planRow: Read<PlanRow> = (v, at) => {
  const f = fields(v, at);
  return { name: f("name", text), depth: f("depth", number), rows: f("rows", number), ms: f("ms", number), ms_text: f("ms_text", text) };
};

const answerView: Read<AnswerView> = (v, at) => {
  const f = fields(v, at);
  return {
    columns: f("columns", list(column)),
    rows: f("rows", cells),
    row_count: f("row_count", number),
    shown: f("shown", number),
    truncated: f("truncated", flag),
    chart: f("chart", chart),
    csv_label: f("csv_label", text),
    sql: f("sql", text),
    grain: f("grain", maybe(text)),
    plan: f("plan", list(planRow)),
    ms: f("ms", number),
    meta: f("meta", text),
    question: f("question", maybe(text)),
  };
};

const verdict: Read<Verdict> = (v, at) => {
  const f = fields(v, at);
  return { kind: f("kind", oneOf(VERDICTS)), text: f("text", text), sql: f("sql", maybe(text)) };
};

const checkRow: Read<CheckRow> = (v, at) => {
  const f = fields(v, at);
  return { name: f("name", text), word: f("word", text), text: f("text", text), detail: f("detail", maybe(text)) };
};

const alternative: Read<Alternative> = (v, at) => {
  const f = fields(v, at);
  return { reading: f("reading", text), label: f("label", text) };
};

const termRow: Read<TermRow> = (v, at) => {
  const f = fields(v, at);
  return { term: f("term", text), word: f("word", text), text: f("text", text), alternatives: f("alternatives", list(alternative)) };
};

const tested: Read<Tested> = (v, at) => {
  const f = fields(v, at);
  return { text: f("text", text), applies: f("applies", flag) };
};

const timing: Read<Timing> = (v, at) => {
  const f = fields(v, at);
  return {
    first_s: f("first_s", maybe(number)),
    verdict_s: f("verdict_s", maybe(number)),
    engine_ms: f("engine_ms", maybe(number)),
    first_text: f("first_text", maybe(text)),
    verdict_text: f("verdict_text", maybe(text)),
  };
};

const jobError: Read<JobError> = (v, at) => ({ message: fields(v, at)("message", text) });

const asked: Read<Asked> = (v, at) => {
  const f = fields(v, at);
  return { term: f("term", text), reading: f("reading", text), question: f("question", text) };
};

export const readSnapshot: Read<Snapshot> = (v, at) => {
  const f = fields(v, at);
  return {
    id: f("id", text),
    version: f("version", number),
    kind: f("kind", oneOf(KINDS)),
    dataset: f("dataset", text),
    text: f("text", text),
    asked: f("asked", maybe(asked)),
    stage: f("stage", oneOf(STAGES)),
    ahead: f("ahead", number),
    verdict: f("verdict", maybe(verdict)),
    first: f("first", maybe(answerView)),
    second: f("second", maybe(answerView)),
    checks: f("checks", list(checkRow)),
    terms: f("terms", list(termRow)),
    tested: f("tested", maybe(tested)),
    timing: f("timing", timing),
    error: f("error", maybe(jobError)),
  };
};

const datasetStatus: Read<DatasetStatus> = (v, at) => {
  const f = fields(v, at);
  return { label: f("label", text), state: f("state", oneOf(LOAD_STATES)), detail: f("detail", text) };
};

const modelStatus: Read<ModelStatus> = (v, at) => {
  const f = fields(v, at);
  return { state: f("state", oneOf(MODEL_STATES)), mode: f("mode", text), detail: f("detail", text) };
};

const lockStatus: Read<LockStatus> = (v, at) => {
  const f = fields(v, at);
  return { holds: f("holds", flag), reason: f("reason", maybe(text)) };
};

const cacheStatus: Read<CacheStatus> = (v, at) => {
  const f = fields(v, at);
  return { on: f("on", flag), reason: f("reason", maybe(text)) };
};

export const readStatus: Read<Status> = (v, at) => {
  const f = fields(v, at);
  return {
    datasets: f("datasets", table(datasetStatus)),
    model: f("model", modelStatus),
    lock: f("lock", lockStatus),
    cache: f("cache", cacheStatus),
    examples: f("examples", table(list(text))),
    tested_panel: f("tested_panel", list(text)),
  };
};

const historyEntry: Read<HistoryEntry> = (v, at) => {
  const f = fields(v, at);
  return {
    id: f("id", text),
    kind: f("kind", oneOf(KINDS)),
    dataset: f("dataset", text),
    text: f("text", text),
    verdict: f("verdict", maybe(verdict)),
    from_job: f("from_job", maybe(text)),
  };
};

export const readHistory: Read<HistoryEntry[]> = (v, at) => fields(v, at)("history", list(historyEntry));

export const readRowsPage: Read<RowsPage> = (v, at) => {
  const f = fields(v, at);
  return { offset: f("offset", number), rows: f("rows", cells) };
};

export const readStopReply: Read<StopReply> = (v, at) => ({ stopped: fields(v, at)("stopped", flag) });

const schemaColumn: Read<SchemaColumn> = (v, at) => {
  const f = fields(v, at);
  return {
    name: f("name", text),
    type: f("type", text),
    description: f("description", text),
    key: f("key", absent(flag, false)),
    references: f("references", absent<string | null>(text, null)),
    values: f("values", absent<string[] | null>(list(text), null)),
  };
};

const schemaTable: Read<SchemaTable> = (v, at) => {
  const f = fields(v, at);
  return { name: f("name", text), rows: f("rows", number), primary_key: f("primary_key", list(text)), columns: f("columns", list(schemaColumn)) };
};

const datasetSchema: Read<DatasetSchema> = (v, at) => {
  const f = fields(v, at);
  return {
    label: f("label", text), state: f("state", oneOf(LOAD_STATES)), detail: f("detail", text),
    about: f("about", text), tables: f("tables", list(schemaTable)),
  };
};

export const readSchema: Read<Schema> = (v, at) => ({ datasets: fields(v, at)("datasets", table(datasetSchema)) });

export const readPreview: Read<Preview> = (v, at) => {
  const f = fields(v, at);
  return {
    dataset: f("dataset", text), table: f("table", text), columns: f("columns", list(column)),
    rows: f("rows", cells), total: f("total", number), ms: f("ms", number),
  };
};

// The message of an error reply ({"error": "..."}), or null when the body isn't one.
export function errorMessage(body: unknown): string | null {
  return isFields(body) && typeof body["error"] === "string" ? body["error"] : null;
}
