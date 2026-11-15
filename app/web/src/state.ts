// What the page holds between redraws, the actions the panels can call, and the rules
// derived from them (is the shown job running, why Ask or Run is off).
import { CONNECTING } from "./copy.js";
import type { Theme } from "./theme.js";
import type { AnswerName, CheckRow, HistoryEntry, Snapshot, Status } from "./types.js";

export type Mode = "ask" | "sql";

export interface State {
  status: Status | null;
  statusError: string | null;
  dataset: string;
  mode: Mode;
  theme: Theme;
  testedOpen: boolean;
  history: HistoryEntry[];
  job: Snapshot | null;
  // The snapshot drawn just before this one, set only for the redraw that shows a change.
  changedFrom: Snapshot | null;
  jobError: string | null;
  submitError: string | null;
  submitting: boolean;
  stopping: boolean;
  extra: Record<AnswerName, string[][]>;
  loadingMore: AnswerName | null;
  moreError: { answer: AnswerName; message: string } | null;
  planOpen: boolean;
  sqlOpen: boolean;
  editFrom: Snapshot | null;
}

export interface Actions {
  setDataset(name: string): void;
  setMode(mode: Mode): void;
  setTheme(theme: Theme): void;
  toggleTested(): void;
  openJob(id: string): void;
  askExample(question: string): void;
  submit(): void;
  stop(): void;
  reask(term: string, reading: string): void;
  editSql(sql: string): void;
  togglePlan(): void;
  toggleSql(): void;
  showMore(answer: AnswerName): void;
}

export const SECOND_ROW = "second_phrasing";

export function isRunning(job: Snapshot | null): boolean {
  return job !== null && (job.stage === "queued" || job.stage === "writing" || job.stage === "first");
}

export function isSql(job: Snapshot): boolean {
  return job.kind === "sql" || job.kind === "edit";
}

export function secondRow(job: Snapshot | null): CheckRow | undefined {
  return job?.checks.find((row) => row.name === SECOND_ROW);
}

// The second answer sits beside the first only when it was shown and gave a different result.
export function showsSecond(job: Snapshot): boolean {
  return job.second !== null && secondRow(job)?.word === "Differs";
}

export function datasetLabel(state: State, name: string = state.dataset): string {
  return state.status?.datasets[name]?.label ?? name;
}

// Why typed SQL can't run on the current dataset, or null when it can.
export function runBlock(state: State): string | null {
  if (state.statusError) return state.statusError;
  const info = state.status?.datasets[state.dataset];
  if (!info) return CONNECTING;
  if (info.state === "loading") return `${info.label} isn't loaded yet.`;
  if (info.state !== "ready") return `${info.label} isn't available.`;
  return null;
}

// Why a question can't be asked now, or null when it can.
export function askBlock(state: State): string | null {
  const model = state.status?.model;
  if (state.statusError || !model) return runBlock(state);
  if (model.state === "loading") return "The model is still loading.";
  if (model.state === "off") return "The analyst is off (started with --no-model).";
  if (model.state === "failed") return `The model didn't load: ${model.detail}`;
  return runBlock(state);
}

// Whether a new question or SQL run can start from the page right now.
export function canStart(state: State): boolean {
  return !isRunning(state.job) && !state.submitting;
}
