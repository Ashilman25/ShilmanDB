// The server's replies as the page reads them. Field names match app/jobs.py, app/views.py
// and app/server.py, and a Python test holds Snapshot and AnswerView to the server's keys.

export type Stage = "queued" | "writing" | "first" | "done" | "stopped" | "error";
export type JobKind = "ask" | "reask" | "sql" | "edit";
export type VerdictKind = "pending" | "clean" | "warned" | "refused" | "cannot" | "failed" | "not_run" | "ran" | "stopped";
export type LoadState = "loading" | "ready" | "missing" | "failed";
export type ModelState = "loading" | "ready" | "off" | "failed";
export type AnswerName = "first" | "second";

export interface Column {
  name: string;
  type: string;
}

// One bar: the label as the table shows it, and the number it's drawn from.
export type BarPoint = [string, number];

export interface ChartBars {
  kind: "bars";
  label: number;
  value: number;
  points: BarPoint[];
}

export interface ChartNumber {
  kind: "number";
  value: number;
  number: number;
  text: string;
}

export interface ChartNone {
  kind: "none";
}

export type Chart = ChartBars | ChartNumber | ChartNone;

export interface PlanRow {
  name: string;
  depth: number;
  rows: number;
  ms: number;
  ms_text: string;
}

export interface AnswerView {
  columns: Column[];
  rows: string[][];
  row_count: number;
  shown: number;
  truncated: boolean;
  chart: Chart;
  csv_label: string;
  sql: string;
  grain: string | null;
  plan: PlanRow[];
  ms: number;
  meta: string;
  question: string | null;
}

export interface Verdict {
  kind: VerdictKind;
  text: string;
  sql: string | null;
}

export interface CheckRow {
  name: string;
  word: string;
  text: string;
  detail: string | null;
}

export interface Alternative {
  reading: string;
  label: string;
}

export interface TermRow {
  term: string;
  word: string;
  text: string;
  alternatives: Alternative[];
}

export interface Tested {
  text: string;
  applies: boolean;
}

export interface Timing {
  first_s: number | null;
  verdict_s: number | null;
  engine_ms: number | null;
  first_text: string | null;
  verdict_text: string | null;
}

export interface JobError {
  message: string;
}

export interface Asked {
  term: string;
  reading: string;
  question: string;
}

export interface Snapshot {
  id: string;
  version: number;
  kind: JobKind;
  dataset: string;
  text: string;
  asked: Asked | null;
  stage: Stage;
  ahead: number;
  verdict: Verdict | null;
  first: AnswerView | null;
  second: AnswerView | null;
  checks: CheckRow[];
  terms: TermRow[];
  tested: Tested | null;
  timing: Timing;
  error: JobError | null;
}

export interface DatasetStatus {
  label: string;
  state: LoadState;
  detail: string;
}

export interface ModelStatus {
  state: ModelState;
  mode: string;
  detail: string;
}

export interface LockStatus {
  holds: boolean;
  reason: string | null;
}

export interface CacheStatus {
  on: boolean;
  reason: string | null;
}

export interface Status {
  datasets: Record<string, DatasetStatus>;
  model: ModelStatus;
  lock: LockStatus;
  cache: CacheStatus;
  examples: Record<string, string[]>;
  tested_panel: string[];
}

export interface HistoryEntry {
  id: string;
  kind: JobKind;
  dataset: string;
  text: string;
  verdict: Verdict | null;
  from_job: string | null;
}

export interface RowsPage {
  offset: number;
  rows: string[][];
}

export interface StopReply {
  stopped: boolean;
}

// One column of a table as /api/schema gives it: key, references and values are filled only when it has them.
export interface SchemaColumn {
  name: string;
  type: string;
  description: string;
  key: boolean;
  references: string | null;
  values: string[] | null;
}

export interface SchemaTable {
  name: string;
  rows: number;
  primary_key: string[];
  columns: SchemaColumn[];
}

export interface DatasetSchema {
  label: string;
  state: LoadState;
  detail: string;
  about: string;
  tables: SchemaTable[];
}

export interface Schema {
  datasets: Record<string, DatasetSchema>;
}

// The first rows of one table, formatted for the grid, with the table's full row count.
export interface Preview {
  dataset: string;
  table: string;
  columns: Column[];
  rows: string[][];
  total: number;
  ms: number;
}
