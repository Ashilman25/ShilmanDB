// The page: it holds the state, follows the shown job with the long poll, and redraws a panel
// in full from the state whenever something it shows changes.
import * as api from "./api.js";
import { ApiError, isAbort } from "./api.js";
import { NO_SUCH_JOB, UNREACHABLE } from "./copy.js";
import { byId, resetScrolls } from "./dom.js";
import { drawEvidence } from "./evidence.js";
import { makeInput } from "./input.js";
import { drawRail } from "./rail.js";
import { drawResult, failedSentence, forgetAnswers } from "./result.js";
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
    modeChosen = true;
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

// Whether you've switched the mode yourself, so the page never switches it on its own after that.
let modeChosen = false;
// The job this page just started from the input; the input keeps your text, not the server's copy.
let started: string | null = null;
// Bumped whenever another job is shown, so a page of rows asked for before that is dropped.
let switches = 0;

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

// A call that never reached the server: recheck the status, whose error path says Not connected
// and retries. Nothing starts when a status call is already in flight or waiting.
function noticeGone(error: unknown): void {
  if (error instanceof ApiError && error.status === 0 && !statusBusy) void loadStatus();
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
    case "error": return failedSentence(job);
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

// Draws a snapshot. A newer version of the shown job keeps what's on screen, the input included;
// another job starts clean, but one you just started from the input keeps your text there.
function show(job: Snapshot): void {
  const before = state.job?.id === job.id ? state.job : null;
  if (!before) {
    switches++;
    forgetAnswers();
    resetScrolls(byId("result"));
    resetScrolls(byId("evidence"));
    state.submitError = null;
    state.extra = { first: [], second: [] };
    state.loadingMore = null;
    state.moreError = null;
    state.planOpen = false;
    state.sqlOpen = false;
    state.stopping = false;
    if (job.id === started) state.editFrom = null;
    else restoreInput(job);
    started = null;
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
  if (job.id === focusJob) focusShownSql(job);
}

// The job named in the URL at load. When it turns out to be SQL, the SQL editor takes the focus
// the question line had, unless you've already moved it.
let focusJob: string | null = null;

function focusShownSql(job: Snapshot): void {
  focusJob = null;
  const active = document.activeElement;
  if (isSql(job) && (active === null || active === document.body || active === byId("question"))) input.focusSql();
}

let following = 0;
let controller: AbortController | null = null;

// Shows a job and keeps it current: each long poll sends the last version drawn and waits for a newer one.
async function follow(id: string, first: Snapshot | null): Promise<void> {
  const token = ++following;
  if (id !== focusJob) focusJob = null;
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
        forgetAnswers();
        focusJob = null;
        state.job = null;
        state.jobError = NO_SUCH_JOB;
        window.history.replaceState(null, "", window.location.pathname);
        paint("ask", "rail", "result", "evidence");
        return;
      }
      noticeGone(error);
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
  const ticket = switches;
  try {
    const job = await send();
    state.submitting = false;
    if (job.kind === "sql" || job.kind === "edit") state.editFrom = null;
    // a re-ask isn't typed text, and a job opened meanwhile has taken the input, so both restore it
    started = job.kind !== "reask" && ticket === switches ? job.id : null;
    void loadHistory();
    void follow(job.id, job);
  } catch (error) {
    noticeGone(error);
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
    noticeGone(error);
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

  const ticket = switches;
  state.loadingMore = answer;
  state.moreError = null;
  paint("result");
  try {
    const page = await api.getRows(job.id, answer, onScreen);
    if (ticket !== switches || state.job?.id !== job.id) return;
    state.extra[answer] = [...state.extra[answer], ...page.rows];
  } catch (error) {
    noticeGone(error);
    if (ticket !== switches || state.job?.id !== job.id) return;
    state.moreError = { answer, message: message(error) };
  }
  const focused = document.activeElement instanceof HTMLElement && document.activeElement.dataset["focus"] === `more-${answer}`;
  state.loadingMore = null;
  paint("result");
  if (focused) keepFocusAfterMore(answer);
}

// When the last page is in, Show more is gone; focus moves to the same table's CSV link.
function keepFocusAfterMore(answer: AnswerName): void {
  const result = byId("result");
  if (result.querySelector(`[data-focus="more-${answer}"]`)) return;
  const csv = result.querySelector(`[data-focus="csv-${answer}"]`);
  if (csv instanceof HTMLElement) csv.focus({ preventScroll: true });
}

async function loadHistory(): Promise<void> {
  try {
    state.history = await api.getHistory();
    paint("rail");
  } catch (error) {
    noticeGone(error);  // the status line then says the app can't be reached
  }
}

function stillLoading(): boolean {
  const status = state.status;
  if (!status) return true;
  return status.model.state === "loading" || Object.values(status.datasets).some((info) => info.state === "loading");
}

// The last status reply, error and dataset drawn, so a poll that brings nothing new paints nothing.
let statusDrawn: string | null = null;
let statusSeen = false;
let statusBusy = false;  // a status call is in flight or waiting to retry

// Started with --no-model, Ask can't work: on the first reply, an untouched page with no job opens in SQL.
function sqlWhenModelOff(): void {
  if (statusSeen || !state.status) return;
  statusSeen = true;
  if (state.status.model.state !== "off" || modeChosen || state.mode !== "ask" || state.job || jobFromHash()) return;
  const focused = document.activeElement === byId("question");
  state.mode = "sql";
  paint("ask");
  if (focused) input.focusSql();
}

// Reads what's loaded; polls about once a second only while something is still loading.
async function loadStatus(): Promise<void> {
  statusBusy = true;
  try {
    state.status = await api.getStatus();
    state.statusError = null;
    if (!(state.dataset in state.status.datasets)) state.dataset = Object.keys(state.status.datasets)[0] ?? state.dataset;
  } catch (error) {
    state.statusError = message(error);
  }
  const seen = JSON.stringify([state.status, state.statusError, state.dataset]);
  if (seen !== statusDrawn) {
    statusDrawn = seen;
    paint("top", "tested", "rail", "ask", "evidence");
    if (!state.job) paint("result");  // the empty hint follows the model's state
  }
  sqlWhenModelOff();
  if (state.statusError || stillLoading()) setTimeout(() => void loadStatus(), state.statusError ? RETRY_MS : STATUS_POLL_MS);
  else statusBusy = false;
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
input.focusQuestion();
void loadStatus();
void loadHistory();
const shown = jobFromHash();
focusJob = shown;
if (shown) void follow(shown, null);

window.addEventListener("hashchange", () => {
  const id = jobFromHash();
  if (id && id !== state.job?.id) void follow(id, null);
});
