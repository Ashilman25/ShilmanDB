// The Checks inspector: the verdict line, the checks, the terms, the SQL and its plan, then the tested line.
// From the first answer on its layout holds still: a question's verdict line keeps two lines, the Running
// row keeps the room its final text usually takes, turns into Same or Differs where it stands, and the
// verdict line settles last.
import { leadText, skeleton, word } from "./controls.js";
import { EMPTY_EVIDENCE, NOT_TESTED, rowsText, STOPPED_QUEUED } from "./copy.js";
import { h, redraw, type Child } from "./dom.js";
import { failedSentence } from "./result.js";
import { askBlock, isSql, secondRow, SECOND_ROW, showsSecond, type Actions, type State } from "./state.js";
import type { CheckRow, PlanRow, Snapshot, TermRow, VerdictKind } from "./types.js";

// Verdicts whose banner is the whole result: no numbers came back to judge.
const BANNER_IS_RESULT: readonly VerdictKind[] = ["refused", "cannot", "failed", "not_run"];

// The words a run that ended without a judged answer leads with, as its result banner says them:
// failed, not run, refused, can't answer, or stopped. Null for a run that has a verdict to wait for.
function endedWords(job: Snapshot): { kind: VerdictKind | "error"; text: string } | null {
  if (job.stage === "error") return { kind: "error", text: failedSentence(job) };
  const verdict = job.verdict;
  if (verdict && BANNER_IS_RESULT.includes(verdict.kind)) return verdict;
  if (job.stage === "stopped") return { kind: "error", text: verdict?.text ?? STOPPED_QUEUED };
  return null;
}

// The verdict line. A run that ended without numbers to judge leads with its banner's words, so a Pass
// under it never reads as approval. A question's line keeps two lines in every stage, so a two-line
// warning lands without moving the rows.
function verdictLine(job: Snapshot, from: Snapshot | null): HTMLElement | null {
  const ended = endedWords(job);
  if (ended) return h("p", { class: isSql(job) ? "verdict ended" : "verdict ended reserve" }, leadText(ended.kind, ended.text));
  const verdict = job.verdict;
  if (!verdict) return null;
  const pending = verdict.kind === "pending";
  const settled = !pending && from !== null && from.verdict?.kind === "pending";
  const classes = ["verdict", isSql(job) ? "" : "reserve", pending ? "pending" : "", settled ? "settle" : ""];
  return h("p", { class: classes.filter(Boolean).join(" ") }, verdict.text);
}

// Words a second-answer row has while it runs or once it compares; only those rows keep the reserved room.
const SECOND_WORDS = ["Running", "Same", "Differs"];

function checkRow(row: CheckRow, from: Snapshot | null): HTMLElement {
  const reserved = row.name === SECOND_ROW && SECOND_WORDS.includes(row.word);
  const running = row.word === "Running";
  const turned = reserved && !running && secondRow(from)?.word === "Running";
  const classes = ["entry", reserved ? "second" : "", turned ? "arrive" : ""].filter(Boolean).join(" ");
  return h("div", { class: classes },
    word(row.word),
    h("div", { class: "entry-body" },
      row.text,
      row.detail ? [" ", h("span", { class: "detail" }, row.detail)] : null,
      running ? skeleton([70, 45]) : null));
}

// A term's other readings: live (true), waiting while the question runs (false), or none on a
// stopped or failed one (null). The note shows from the first answer on, so nothing moves later.
function termRow(term: TermRow, live: boolean | null, actions: Actions): HTMLElement {
  const links = live !== null && term.alternatives.length ? h("div", { class: "alts" },
    term.alternatives.map((alt) => h("button", {
      class: "link",
      attrs: { type: "button", disabled: !live, "data-focus": `alt-${term.term}-${alt.reading}` },
      onClick: () => actions.reask(term.term, alt.reading),
    }, alt.label)),
    h("span", { class: "detail" }, NOT_TESTED)) : null;
  return h("div", { class: "entry" }, word(term.word), h("div", { class: "entry-body" }, term.text, links));
}

// The plan as an indented list, root first, with each operator's rows and time.
function planList(plan: PlanRow[]): HTMLElement {
  return h("ol", { class: "plan", attrs: { id: "plan" } }, plan.map((op) => {
    const item = h("li", {}, h("span", { class: "op" }, op.name), h("span", { class: "detail" }, `${rowsText(op.rows)}, ${op.ms_text}`));
    item.style.setProperty("--depth", String(op.depth));
    return item;
  }));
}

// The answer's SQL; typed SQL that never ran (not run, or an engine error) shows as typed, so it can be fixed.
function sqlSection(job: Snapshot, state: State, actions: Actions): Child[] {
  const sql = job.first?.sql ?? (isSql(job) ? job.text : null);
  if (sql === null) return [];
  const plan = job.first?.plan ?? [];
  const second = job.second && showsSecond(job) ? job.second : null;
  return [
    h("h2", {}, "SQL"),
    second ? h("p", { class: "sql-label" }, "First answer") : null,
    h("pre", { class: "sql", attrs: { "data-scroll": "sql-first" } }, sql),
    second ? [h("p", { class: "sql-label" }, "Second answer"), h("pre", { class: "sql", attrs: { "data-scroll": "sql-second" } }, second.sql)] : null,
    h("div", { class: "actions" },
      h("button", { class: "btn2", attrs: { type: "button", "data-focus": "edit-sql" }, onClick: () => actions.editSql(sql) }, "Edit and run"),
      plan.length ? h("button", {
        class: "btn2",
        attrs: { type: "button", "aria-expanded": String(state.planOpen), "aria-controls": "plan", "data-focus": "plan" },
        onClick: () => actions.togglePlan(),
      }, state.planOpen ? "Hide plan" : "Show plan") : null),
    state.planOpen && plan.length ? planList(plan) : null,
  ];
}

function placeholder(): Child[] {
  return [h("div", { class: "ledger" },
    [0, 1, 2].map(() => h("div", { class: "entry" }, skeleton([100]), h("div", { class: "entry-body" }, skeleton([85])))))];
}

// What the inspector holds for the shown job; the hint when no check, SQL or tested line applies.
function panel(state: State, actions: Actions): Child[] {
  const job = state.job;
  if (!job) return [h("p", { class: "hint" }, EMPTY_EVIDENCE)];
  if (job.stage === "queued" || job.stage === "writing") return placeholder();

  const from = state.changedFrom;
  const finished = job.stage === "done" && askBlock(state) === null;
  const ended = job.stage === "stopped" || job.stage === "error";
  const live = ended ? null : finished && !state.submitting;
  const line = verdictLine(job, from);
  const parts: Child[] = [
    line,
    job.checks.length ? h("div", { class: "ledger" }, job.checks.map((row) => checkRow(row, from))) : null,
    job.terms.length ? [h("h2", {}, "Terms"), h("div", { class: "ledger" }, job.terms.map((term) => termRow(term, live, actions)))] : null,
    sqlSection(job, state, actions),
    job.tested ? h("p", { class: "tested" }, job.tested.text) : null,
  ];
  return parts.flat().some(Boolean) ? parts : [h("p", { class: "hint" }, EMPTY_EVIDENCE)];
}

export function drawEvidence(root: HTMLElement, state: State, actions: Actions): void {
  redraw(root,
    h("div", { class: "ins-head" }, h("h2", {}, "Checks")),
    h("div", { class: "ins-body", attrs: { "data-scroll": "checks" } }, panel(state, actions)));
}
