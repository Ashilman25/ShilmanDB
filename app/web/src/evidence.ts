// The evidence panel: the verdict line, the checks, the terms, the SQL and its plan, then the tested line.
// From the first answer on its layout holds still: a question's verdict line keeps two lines, the Running
// row keeps the room its final text usually takes (four lines beside the result, three under it), turns
// into Same or Differs where it stands, and the verdict line settles last.
import { skeleton, word } from "./controls.js";
import { EMPTY_EVIDENCE, NOT_TESTED, rowsText } from "./copy.js";
import { byId, h, redraw, type Child } from "./dom.js";
import { askBlock, isSql, secondRow, SECOND_ROW, showsSecond, type Actions, type State } from "./state.js";
import type { CheckRow, PlanRow, Snapshot, TermRow, VerdictKind } from "./types.js";

// Verdicts whose banner is the whole result; the banner carries them, so the panel doesn't repeat them.
const BANNER_IS_RESULT: readonly VerdictKind[] = ["refused", "cannot", "failed", "not_run"];

// The verdict line at the top of the panel, unless the result column's banner already says it.
// A typed SQL that failed in the engine has no verdict, so its banner stands alone too.
// A question's line keeps two lines in every stage, so a two-line warning lands without moving the rows.
function verdictLine(job: Snapshot, from: Snapshot | null): HTMLElement | null {
  const verdict = job.verdict;
  if (!verdict || BANNER_IS_RESULT.includes(verdict.kind)) return null;
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

// How a term's other readings show: live links on a finished question, links that wait while it runs,
// and none on a stopped one, which can't be asked again. The note sits beside them from the first answer
// on, so it never arrives later and moves the rows below.
interface Readings {
  live: boolean;
}

function termRow(term: TermRow, readings: Readings | null, actions: Actions): HTMLElement {
  const links = readings && term.alternatives.length ? h("div", { class: "alts" },
    term.alternatives.map((alt) => h("button", {
      class: "link",
      attrs: { type: "button", disabled: !readings.live, "data-focus": `alt-${term.term}-${alt.reading}` },
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
    h("pre", { class: "sql" }, sql),
    second ? [h("p", { class: "sql-label" }, "Second answer"), h("pre", { class: "sql" }, second.sql)] : null,
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
  return [h("h2", {}, "Checks"), h("div", { class: "ledger" },
    [0, 1, 2].map(() => h("div", { class: "entry" }, skeleton([100]), h("div", { class: "entry-body" }, skeleton([85])))))];
}

// What the panel holds for the shown job; nothing at all when no check, SQL or tested line applies.
function panel(state: State, actions: Actions): Child[] {
  const job = state.job;
  if (!job) return [h("p", { class: "hint" }, EMPTY_EVIDENCE)];
  if (job.stage === "queued" || job.stage === "writing") return placeholder();

  const from = state.changedFrom;
  const finished = job.stage === "done" && askBlock(state) === null;
  const readings: Readings | null = job.stage === "stopped" ? null : { live: finished && !state.submitting };
  const line = verdictLine(job, from);
  return [
    line || job.checks.length ? h("h2", {}, "Checks") : null,
    line,
    job.checks.length ? h("div", { class: "ledger" }, job.checks.map((row) => checkRow(row, from))) : null,
    job.terms.length ? [h("h2", {}, "Terms"), h("div", { class: "ledger" }, job.terms.map((term) => termRow(term, readings, actions)))] : null,
    sqlSection(job, state, actions),
    job.tested ? h("p", { class: "tested" }, job.tested.text) : null,
  ];
}

// Draws the panel. An empty one steps aside: the columns get the "alone" class, the result takes
// the room and keeps a readable measure (set here rather than by a CSS :has() rule, so it never waits on
// the browser re-checking the panel).
export function drawEvidence(root: HTMLElement, state: State, actions: Actions): void {
  redraw(root, panel(state, actions));
  byId("cols").classList.toggle("alone", root.children.length === 0);
}
