// The Query tab's editor: the bar (the Ask / SQL switch, the database it uses, the action button and its key),
// the question line or the SQL editor, and the messages under them. The two text fields are in index.html, so
// typing is never lost to a redraw and the editor's attributes (no spellcheck, no autocorrect) are there before
// any script runs. The database select is made once for the same reason.
import { choice, chevron } from "./controls.js";
import { byId, fill, h, redraw } from "./dom.js";
import { makeSqlEditor } from "./sqledit.js";
import {
  askBlock, datasetLabel, isRunning, runBlock, type Actions, type Mode, type State,
} from "./state.js";

export interface InputPanel {
  draw(state: State): void;
  question(): string;
  sql(): string;
  setQuestion(text: string): void;
  setSql(text: string): void;
  focusQuestion(): void;
  focusSql(): void;
}

const MODES: { value: Mode; text: string }[] = [{ value: "ask", text: "Ask" }, { value: "sql", text: "SQL" }];

function inputById(id: string): HTMLInputElement {
  const node = byId(id);
  if (!(node instanceof HTMLInputElement)) throw new Error(`#${id} isn't an input`);
  return node;
}

function textareaById(id: string): HTMLTextAreaElement {
  const node = byId(id);
  if (!(node instanceof HTMLTextAreaElement)) throw new Error(`#${id} isn't a textarea`);
  return node;
}

function actionButton(state: State, actions: Actions): HTMLElement {
  if (isRunning(state.job)) {
    return h("button", {
      class: "btn stop",
      attrs: { type: "button", disabled: state.stopping, "data-focus": "action" },
      onClick: () => actions.stop(),
    }, state.stopping ? "Stopping" : "Stop");
  }
  const blocked = state.mode === "ask" ? askBlock(state) : runBlock(state);
  return h("button", {
    class: "btn",
    attrs: {
      type: "button", disabled: blocked !== null || state.submitting, "aria-describedby": blocked ? "why-off" : "run-key",
      "data-focus": "action",
    },
    onClick: () => actions.submit(),
  }, state.mode === "ask" ? "Ask" : "Run");
}

// Under the input: a refused submit, why the button is off, an edit in progress, a dataset that won't load.
function messages(state: State): (HTMLElement | null)[] {
  const blocked = state.mode === "ask" ? askBlock(state) : runBlock(state);
  const info = state.status?.datasets[state.dataset];
  const unavailable = info && (info.state === "missing" || info.state === "failed") ? info : null;
  const edit = state.mode === "sql" ? state.editFrom : null;
  return [
    state.submitError ? h("p", { class: "msg bad", attrs: { role: "alert" } }, state.submitError) : null,
    blocked && !unavailable ? h("p", { class: "msg", attrs: { id: "why-off" } }, blocked) : null,
    edit ? h("p", { class: "msg" }, `Editing the SQL of "${edit.asked?.question ?? edit.text}". It runs as edited SQL.`) : null,
    unavailable ? h("div", { class: "banner neutral", attrs: { id: "why-off" } },
      h("p", {}, h("strong", { class: "lead" }, `${unavailable.label} isn't available.`)),
      h("p", { class: "pre" }, unavailable.detail)) : null,
  ];
}

export function makeInput(actions: Actions): InputPanel {
  const question = inputById("question");
  const sql = textareaById("sql");
  const modeSlot = byId("mode");
  const dbSlot = byId("db");
  const actionSlot = byId("action");
  const under = byId("under");
  const editor = makeSqlEditor(sql, byId("sql-box"), byId("code-lines"), byId("splitter"), byId("query-pane"));
  const database = h("select", {
    class: "db-select",
    attrs: { "aria-label": "Database", "data-focus": "database" },
    onChange: (event) => { if (event.target instanceof HTMLSelectElement) actions.setDataset(event.target.value); },
  });
  // The select with the tree's chevron as its arrow, so it's drawn like the switch beside it.
  const select = h("span", { class: "select-wrap" }, database, chevron());
  let drawn = "";
  let choices = "";

  question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.isComposing && !event.shiftKey) {
      event.preventDefault();
      actions.submit();
    }
  });
  sql.addEventListener("keydown", (event) => {
    if (editor.onKey(event)) return;  // the names list took the key
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey) && !event.isComposing) {
      event.preventDefault();
      actions.submit();
    }
  });

  // The select's options, rebuilt only when the datasets or their labels change.
  function drawDatabases(state: State): void {
    const names = Object.keys(state.status?.datasets ?? state.schema?.datasets ?? {});
    const next = JSON.stringify(names.map((name) => [name, datasetLabel(state, name)]));
    if (next !== choices) {
      choices = next;
      fill(database, names.map((name) => h("option", { attrs: { value: name } }, datasetLabel(state, name))));
    }
    database.value = state.dataset;
    fill(dbSlot, names.length ? select : null);
  }

  return {
    draw(state) {
      const label = datasetLabel(state);
      redraw(modeSlot, choice("mode", "Ask a question or write SQL", MODES, state.mode, (mode) => actions.setMode(mode), true));
      drawDatabases(state);
      question.hidden = state.mode !== "ask";
      editor.draw(state, state.mode === "sql");
      question.placeholder = `Ask a question about ${label}. Enter asks.`;
      sql.placeholder = `Write SQL for ${label}. Cmd+Enter runs.`;
      redraw(actionSlot,
        isRunning(state.job) ? null : h("span", { class: "key-hint", attrs: { id: "run-key" } }, "Cmd+Enter"),
        actionButton(state, actions));
      // Redrawn only when its words change, so a screen reader hears an alert once, not on every status poll.
      const notes = messages(state);
      const text = notes.map((note) => note?.textContent ?? "").join("\n");
      if (text !== drawn) redraw(under, notes);
      drawn = text;
    },
    question: () => question.value,
    sql: () => sql.value,
    setQuestion: (text) => { question.value = text; },
    setSql: (text) => {
      sql.value = text;
      editor.render();
    },
    focusQuestion: () => question.focus(),
    focusSql: () => sql.focus(),
  };
}
