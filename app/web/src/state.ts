// What the page holds between redraws, the actions the panels can call, and the rules
// derived from them (is the shown job running, why Ask or Run is off).
import { BUSY, CONNECTING } from "./copy.js";
import type { Theme } from "./theme.js";
import type { AnswerName, CheckRow, HistoryEntry, Preview, Schema, SchemaTable, Snapshot, Status } from "./types.js";

export type Mode = "ask" | "sql";
export type SideTab = "tables" | "history";
export type ResultView = "table" | "chart";

export const QUERY_TAB = "query";
export const MAX_TABLE_TABS = 8;
// The table each database opens with.
export const START_TABLES: Record<string, string> = { tpch: "orders", reviews: "reviews" };

// One open table: its tab key is "dataset.table".
export interface TableTab {
  key: string;
  dataset: string;
  table: string;
}

// A table tab's rows: the preview once it's in, how many are on screen, and the picked column.
export interface PreviewState {
  data: Preview | null;
  error: string | null;
  lost: boolean;  // the error came from a call that never reached the server
  loading: boolean;
  shown: number;
  column: string | null;
}

// The picked cell of one grid, as the status bar shows it.
export interface Pick {
  grid: string;
  column: string;
  value: string;
}

export interface State {
  status: Status | null;
  statusError: string | null;
  schema: Schema | null;
  schemaError: string | null;
  dataset: string;
  mode: Mode;
  theme: Theme;
  testedOpen: boolean;
  sideTab: SideTab;
  filter: string;
  expanded: Set<string>;
  tabs: TableTab[];
  active: string;
  previews: Map<string, PreviewState>;
  pick: Pick | null;
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
  resultView: ResultView;
  editFrom: Snapshot | null;
}

export interface Actions {
  setDataset(name: string): void;
  setMode(mode: Mode): void;
  setTheme(theme: Theme): void;
  toggleTested(): void;
  setSideTab(tab: SideTab): void;
  setFilter(text: string): void;
  toggleNode(key: string): void;
  openTable(dataset: string, table: string, column?: string): void;
  closeTab(key: string): void;
  activate(key: string): void;
  queryTable(dataset: string, table: string): void;
  showMorePreview(key: string): void;
  pickColumn(key: string, column: string): void;
  pickCell(pick: Pick): void;
  openJob(id: string): void;
  askExample(question: string): void;
  loadSql(sql: string): void;
  submit(): void;
  stop(): void;
  reask(term: string, reading: string): void;
  editSql(sql: string): void;
  togglePlan(): void;
  toggleSql(): void;
  setResultView(view: ResultView): void;
  showMore(answer: AnswerName): void;
}

export const SECOND_ROW = "second_phrasing";

export function tableKey(dataset: string, table: string): string {
  return `${dataset}.${table}`;
}

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
  return state.status?.datasets[name]?.label ?? state.schema?.datasets[name]?.label ?? name;
}

// A table's schema entry, once /api/schema is in.
export function schemaTable(state: State, dataset: string, table: string): SchemaTable | undefined {
  return state.schema?.datasets[dataset]?.tables.find((entry) => entry.name === table);
}

export function activeTable(state: State): TableTab | undefined {
  return state.tabs.find((tab) => tab.key === state.active);
}

// Why typed SQL can't run on a dataset (the Query tab's, unless named), or null when it can.
export function runBlock(state: State, dataset: string = state.dataset): string | null {
  if (state.statusError) return state.statusError;
  const info = state.status?.datasets[dataset];
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

// Why Query this table can't run now, or null when it can: a run already under way, or the Run button's reason.
export function queryBlock(state: State, dataset: string): string | null {
  if (!canStart(state)) return BUSY;
  return runBlock(state, dataset);
}
