// The status bar: each dataset's engine, the model and (with the live model) the prompt cache on the left;
// on the right, the picked cell's column and value, or the shown result's rows and time.
import { CONNECTING, msText, rowsText } from "./copy.js";
import { h, redraw } from "./dom.js";
import { activeTable, QUERY_TAB } from "./state.js";
const QUIET_STATES = ["loading", "ready", "off"];
// A dataset's name without its scale: "TPC-H", "Reviews".
function shortName(info) {
    return info.label.split(",", 1)[0] ?? info.label;
}
// A dataset's engine in words. Missing files or a failed load are the data's problem;
// a missing dataset with only the build hint means the engine bridge isn't built.
function datasetWords(info) {
    if (info.state === "missing")
        return info.detail.startsWith("Missing:") ? "Data missing" : "Engine missing";
    if (info.state === "failed")
        return "Data failed";
    return `Engine ${info.state}`;
}
// The prompt cache, for the live model only: on, or off with the reason in plain sight.
function cacheWords(status) {
    if (status.model.mode !== "mlx")
        return null;
    return status.cache.on ? "Prompt cache on" : `Prompt cache off: ${status.cache.reason ?? "no reason given"}`;
}
// One item: a name in ink when it has one, then its state, in red when it's a problem.
function item(name, words, bad) {
    return h("span", { class: "sb-item" }, name ? [h("span", { class: "sb-name" }, name), " "] : null, h("span", { class: bad ? "state bad" : "state" }, words));
}
function loaded(state) {
    if (state.statusError)
        return [item(null, "Not connected", true), h("span", { class: "sb-item" }, state.statusError)];
    const status = state.status;
    if (!status)
        return [item(null, CONNECTING, false)];
    const model = status.model;
    const recorded = model.mode === "recorded" && model.state === "ready" ? " (recorded)" : "";
    const cache = cacheWords(status);
    return [
        Object.values(status.datasets).map((info) => item(shortName(info), datasetWords(info), !QUIET_STATES.includes(info.state))),
        item(null, `Model ${model.state}${recorded}`, !QUIET_STATES.includes(model.state)),
        cache ? item(null, cache, false) : null,
    ];
}
// The right side: the picked cell of a grid on the shown tab, else that tab's rows and time.
function result(state) {
    const table = activeTable(state);
    const grid = state.pick?.grid ?? "";
    const shown = table ? grid === `table:${table.key}` : state.job !== null && grid.startsWith(`job:${state.job.id}:`);
    const pick = shown ? state.pick : null;
    if (pick) {
        return h("span", { class: "sb-pick", attrs: { title: pick.value } }, h("span", { class: "sb-name" }, pick.column), " ", pick.value);
    }
    if (table) {
        const data = state.previews.get(table.key)?.data;
        return data ? h("span", {}, `${rowsText(data.rows.length)} in ${msText(data.ms)}`) : null;
    }
    const first = state.active === QUERY_TAB ? state.job?.first : null;
    return first ? h("span", {}, `${rowsText(first.row_count)} in ${msText(first.ms)}`) : null;
}
export function drawStatusBar(root, state) {
    redraw(root, h("div", { class: "sb-start" }, loaded(state)), h("div", { class: "sb-end" }, result(state)));
}
