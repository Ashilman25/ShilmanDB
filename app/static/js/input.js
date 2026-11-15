// The input row: the Ask / SQL switch, the question line or the SQL editor, the action button,
// and the messages under it. The two text fields are in index.html, so typing is never lost to a redraw
// and the editor's attributes (no spellcheck, no autocorrect) are there before any script runs.
import { choice } from "./controls.js";
import { byId, h, redraw } from "./dom.js";
import { askBlock, datasetLabel, isRunning, runBlock, } from "./state.js";
const MODES = [{ value: "ask", text: "Ask" }, { value: "sql", text: "SQL" }];
function inputById(id) {
    const node = byId(id);
    if (!(node instanceof HTMLInputElement))
        throw new Error(`#${id} isn't an input`);
    return node;
}
function textareaById(id) {
    const node = byId(id);
    if (!(node instanceof HTMLTextAreaElement))
        throw new Error(`#${id} isn't a textarea`);
    return node;
}
function actionButton(state, actions) {
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
        attrs: { type: "button", disabled: blocked !== null || state.submitting, "aria-describedby": blocked ? "why-off" : null, "data-focus": "action" },
        onClick: () => actions.submit(),
    }, state.mode === "ask" ? "Ask" : "Run");
}
// Under the input: a refused submit, why the button is off, an edit in progress, a dataset that won't load.
function messages(state) {
    const blocked = state.mode === "ask" ? askBlock(state) : runBlock(state);
    const info = state.status?.datasets[state.dataset];
    const unavailable = info && (info.state === "missing" || info.state === "failed") ? info : null;
    const edit = state.mode === "sql" ? state.editFrom : null;
    return [
        state.submitError ? h("p", { class: "msg bad", attrs: { role: "alert" } }, state.submitError) : null,
        blocked && !unavailable ? h("p", { class: "msg", attrs: { id: "why-off" } }, blocked) : null,
        edit ? h("p", { class: "msg" }, `Editing the SQL of "${edit.asked?.question ?? edit.text}". It runs as edited SQL.`) : null,
        unavailable ? h("div", { class: "banner neutral", attrs: { id: "why-off" } }, h("p", {}, h("strong", { class: "lead" }, `${unavailable.label} isn't available.`)), h("p", { class: "pre" }, unavailable.detail)) : null,
    ];
}
export function makeInput(actions) {
    const question = inputById("question");
    const sql = textareaById("sql");
    const modeSlot = byId("mode");
    const actionSlot = byId("action");
    const under = byId("under");
    let drawn = "";
    question.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.isComposing && !event.shiftKey) {
            event.preventDefault();
            actions.submit();
        }
    });
    sql.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && (event.metaKey || event.ctrlKey) && !event.isComposing) {
            event.preventDefault();
            actions.submit();
        }
    });
    return {
        draw(state) {
            const label = datasetLabel(state);
            redraw(modeSlot, choice("mode", "Ask a question or write SQL", MODES, state.mode, (mode) => actions.setMode(mode), true));
            question.hidden = state.mode !== "ask";
            sql.hidden = state.mode !== "sql";
            question.placeholder = `Ask a question about ${label}. Enter asks.`;
            sql.placeholder = `Write SQL for ${label}. Cmd+Enter runs.`;
            redraw(actionSlot, actionButton(state, actions));
            // Redrawn only when its words change, so a screen reader hears an alert once, not on every status poll.
            const notes = messages(state);
            const text = notes.map((note) => note?.textContent ?? "").join("\n");
            if (text !== drawn)
                redraw(under, notes);
            drawn = text;
        },
        question: () => question.value,
        sql: () => sql.value,
        setQuestion: (text) => { question.value = text; },
        setSql: (text) => { sql.value = text; },
        focusQuestion: () => question.focus(),
        focusSql: () => sql.focus(),
    };
}
