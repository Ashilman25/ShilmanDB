// The page: it holds the state, follows the shown job with the long poll, and redraws a panel
// in full from the state whenever something it shows changes.
import * as api from "./api.js";
import { ApiError, isAbort } from "./api.js";
import { NO_SUCH_JOB, UNREACHABLE } from "./copy.js";
import { byId } from "./dom.js";
import { drawEvidence } from "./evidence.js";
import { makeInput } from "./input.js";
import { drawRail } from "./rail.js";
import { drawResult } from "./result.js";
import { askBlock, canStart, isRunning, isSql, runBlock, type Actions, type State } from "./state.js";
import { applyTheme, savedTheme, saveTheme } from "./theme.js";
import { drawTested, drawTop } from "./topbar.js";
import type { AnswerName, Snapshot } from "./types.js";

const STATUS_POLL_MS = 1000;
const RETRY_MS = 2000;

const state: State = {
  status: null,
  statusError: null,
  dataset: "tpch",
  mode: "ask",
  theme: savedTheme(),
  testedOpen: false,
  history: [],
  job: null,
  changedFrom: null,
  jobError: null,
  submitError: null,
  submitting: false,
  stopping: false,
  extra: { first: [], second: [] },
  loadingMore: null,
  moreError: null,
  planOpen: false,
  sqlOpen: false,
  editFrom: null,
};

const actions: Actions = {
  setDataset(name) {
    state.dataset = name;
    state.submitError = null;
    paint("top", "rail", "ask");
  },
  setMode(mode) {
    state.mode = mode;
    state.submitError = null;
    paint("ask");
  },
  setTheme(theme) {
    state.theme = theme;
    saveTheme(theme);
    applyTheme(theme);
    paint("top");
  },
  toggleTested() {
    state.testedOpen = !state.testedOpen;
    paint("top", "tested");
  },
  openJob(id) {
    void follow(id, null);
  },
  askExample(question) {
    state.mode = "ask";
    input.setQuestion(question);
    void submit();
  },
  submit() {
    void submit();
  },
  stop() {
    void stop();
  },
  reask(term, reading) {
    const job = state.job;
    if (job) void start(() => api.reask(job.id, term, reading));
  },
  editSql(sql) {
    const job = state.job;
    if (!job) return;
    state.mode = "sql";
    state.dataset = job.dataset;
    state.editFrom = job.kind === "sql" || job.kind === "edit" ? null : job;
    state.submitError = null;
    input.setSql(sql);
    paint("top", "rail", "ask");
    input.focusSql();
  },
  togglePlan() {
    state.planOpen = !state.planOpen;
    paint("evidence");
  },
  toggleSql() {
    state.sqlOpen = !state.sqlOpen;
    paint("result");
  },
  showMore(answer) {
    void showMore(answer);
  },
};

const input = makeInput(actions);

const panels = {
  top: () => drawTop(byId("top"), state, actions),
  tested: () => drawTested(byId("tested"), state),
  rail: () => drawRail(byId("rail"), state, actions),
  ask: () => input.draw(state),
  result: () => drawResult(byId("result"), state, actions),
  evidence: () => drawEvidence(byId("evidence"), state, actions),
};

type Panel = keyof typeof panels;

function paint(...names: Panel[]): void {
  for (const name of names) panels[name]();
}

function message(error: unknown): string {
  return error instanceof ApiError ? error.message : UNREACHABLE;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// The status line for the live region, said once per stage.
function announcement(job: Snapshot): string {
  switch (job.stage) {
    case "queued": return "Queued.";
    case "writing": return job.kind === "sql" || job.kind === "edit" ? "Running." : "Writing SQL.";
    case "first": return "First answer back. The second is running.";
    case "done": return job.verdict?.text ?? "Done.";
    case "stopped": return "Stopped.";
    case "error": return job.error?.message ?? "The job hit an error.";
  }
}

// Puts the input back to what a job asked: questions in Ask mode with the question as typed
// (a re-ask's original), SQL in SQL mode, on the job's dataset.
function restoreInput(job: Snapshot): void {
  state.dataset = job.dataset;
  state.editFrom = null;
  if (isSql(job)) {
    state.mode = "sql";
    input.setSql(job.text);
  } else {
    state.mode = "ask";
    input.setQuestion(job.asked?.question ?? job.text);
  }
}

// Draws a snapshot. A newer version of the shown job keeps what's on screen around it, the input
// included, so typing while your own job runs is never overwritten; another job starts from a clean slate.
function show(job: Snapshot): void {
  const before = state.job?.id === job.id ? state.job : null;
  if (!before) {
    state.extra = { first: [], second: [] };
    state.loadingMore = null;
    state.moreError = null;
    state.planOpen = false;
    state.sqlOpen = false;
    state.stopping = false;
    restoreInput(job);
    byId("main").scrollTop = 0;
  }
  if (!isRunning(job)) state.stopping = false;
  if (before?.stage !== job.stage) byId("announce").textContent = announcement(job);

  state.job = job;
  state.jobError = null;
  state.changedFrom = before;
  if (!before) paint("top");
  paint("ask", "rail", "result", "evidence");
  state.changedFrom = null;
}

let following = 0;
let controller: AbortController | null = null;

// Shows a job and keeps it current: each long poll sends the last version drawn and waits for a newer one.
async function follow(id: string, first: Snapshot | null): Promise<void> {
  const token = ++following;
  controller?.abort();
  const own = new AbortController();
  controller = own;
  window.history.replaceState(null, "", `#job=${encodeURIComponent(id)}`);

  let after = -1;
  if (first) {
    show(first);
    after = first.version;
  }
  while (token === following) {
    let job: Snapshot;
    try {
      job = await api.getJob(id, after, own.signal);
    } catch (error) {
      if (token !== following || isAbort(error)) return;
      if (error instanceof ApiError && error.status === 404) {
        state.job = null;
        state.jobError = NO_SUCH_JOB;
        window.history.replaceState(null, "", window.location.pathname);
        paint("ask", "rail", "result", "evidence");
        return;
      }
      const problem = message(error);
      if (problem !== state.jobError) {
        state.jobError = problem;
        paint("result");
      }
      await sleep(RETRY_MS);
      continue;
    }
    if (token !== following) return;
    if (job.version !== after || state.jobError) show(job);
    after = job.version;
    if (!isRunning(job)) {
      void loadHistory();
      return;
    }
  }
}

// Sends a new job (a question, SQL or a re-ask) and follows it; a refusal shows under the input.
async function start(send: () => Promise<Snapshot>): Promise<void> {
  if (!canStart(state)) return;
  state.submitting = true;
  state.submitError = null;
  paint("ask", "rail", "evidence");
  try {
    const job = await send();
    state.submitting = false;
    if (job.kind === "sql" || job.kind === "edit") state.editFrom = null;
    void loadHistory();
    void follow(job.id, job);
  } catch (error) {
    state.submitting = false;
    state.submitError = message(error);
    paint("ask", "rail", "evidence");
  }
}

async function submit(): Promise<void> {
  if (state.mode === "ask") {
    if (askBlock(state) === null) await start(() => api.ask(state.dataset, input.question()));
  } else if (runBlock(state) === null) {
    const from = state.editFrom;
    await start(() => api.runSql(state.dataset, input.sql(), from ? from.id : null));
  }
}

async function stop(): Promise<void> {
  const job = state.job;
  if (!job || !isRunning(job)) return;
  state.stopping = true;
  paint("ask");
  try {
    await api.stopJob(job.id);
  } catch (error) {
    state.stopping = false;
    state.submitError = message(error);
    paint("ask");
  }
}

// Fetches the next page of an answer's rows, starting after the rows on screen.
async function showMore(answer: AnswerName): Promise<void> {
  const job = state.job;
  const view = job?.[answer];
  if (!job || !view || state.loadingMore) return;
  const onScreen = view.rows.length + state.extra[answer].length;
  if (onScreen >= view.shown) return;

  state.loadingMore = answer;
  state.moreError = null;
  paint("result");
  try {
    const page = await api.getRows(job.id, answer, onScreen);
    if (state.job?.id === job.id) state.extra[answer] = [...state.extra[answer], ...page.rows];
  } catch (error) {
    if (state.job?.id === job.id) state.moreError = { answer, message: message(error) };
  }
  state.loadingMore = null;
  paint("result");
}

async function loadHistory(): Promise<void> {
  try {
    state.history = await api.getHistory();
    paint("rail");
  } catch {
    // the status line already says when the app can't be reached
  }
}

function stillLoading(): boolean {
  const status = state.status;
  if (!status) return true;
  return status.model.state === "loading" || Object.values(status.datasets).some((info) => info.state === "loading");
}

// Reads what's loaded; polls about once a second only while something is still loading.
async function loadStatus(): Promise<void> {
  try {
    state.status = await api.getStatus();
    state.statusError = null;
    if (!(state.dataset in state.status.datasets)) state.dataset = Object.keys(state.status.datasets)[0] ?? state.dataset;
  } catch (error) {
    state.statusError = message(error);
  }
  paint("top", "tested", "rail", "ask", "evidence");
  if (state.statusError || stillLoading()) setTimeout(() => void loadStatus(), state.statusError ? RETRY_MS : STATUS_POLL_MS);
}

function jobFromHash(): string | null {
  const found = /^#job=([^&]+)$/.exec(window.location.hash);
  if (!found?.[1]) return null;
  try {
    return decodeURIComponent(found[1]);
  } catch {
    return null;
  }
}

applyTheme(state.theme);
paint("top", "tested", "rail", "ask", "result", "evidence");
void loadStatus();
void loadHistory();
const shown = jobFromHash();
if (shown) void follow(shown, null);

window.addEventListener("hashchange", () => {
  const id = jobFromHash();
  if (id && id !== state.job?.id) void follow(id, null);
});
