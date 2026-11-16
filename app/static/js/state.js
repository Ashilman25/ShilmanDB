// What the page holds between redraws, the actions the panels can call, and the rules
// derived from them (is the shown job running, why Ask or Run is off).
import { BUSY, CONNECTING } from "./copy.js";
export const QUERY_TAB = "query";
export const MAX_TABLE_TABS = 8;
// The table each database opens with.
export const START_TABLES = { tpch: "orders", reviews: "reviews" };
export const SECOND_ROW = "second_phrasing";
export function tableKey(dataset, table) {
    return `${dataset}.${table}`;
}
export function isRunning(job) {
    return job !== null && (job.stage === "queued" || job.stage === "writing" || job.stage === "first");
}
export function isSql(job) {
    return job.kind === "sql" || job.kind === "edit";
}
export function secondRow(job) {
    return job?.checks.find((row) => row.name === SECOND_ROW);
}
// The second answer sits beside the first only when it was shown and gave a different result.
export function showsSecond(job) {
    return job.second !== null && secondRow(job)?.word === "Differs";
}
export function datasetLabel(state, name = state.dataset) {
    return state.status?.datasets[name]?.label ?? state.schema?.datasets[name]?.label ?? name;
}
// A table's schema entry, once /api/schema is in.
export function schemaTable(state, dataset, table) {
    return state.schema?.datasets[dataset]?.tables.find((entry) => entry.name === table);
}
export function activeTable(state) {
    return state.tabs.find((tab) => tab.key === state.active);
}
// Why typed SQL can't run on a dataset (the Query tab's, unless named), or null when it can.
export function runBlock(state, dataset = state.dataset) {
    if (state.statusError)
        return state.statusError;
    const info = state.status?.datasets[dataset];
    if (!info)
        return CONNECTING;
    if (info.state === "loading")
        return `${info.label} isn't loaded yet.`;
    if (info.state !== "ready")
        return `${info.label} isn't available.`;
    return null;
}
// Why a question can't be asked now, or null when it can.
export function askBlock(state) {
    const model = state.status?.model;
    if (state.statusError || !model)
        return runBlock(state);
    if (model.state === "loading")
        return "The model is still loading.";
    if (model.state === "off")
        return "The analyst is off (started with --no-model).";
    if (model.state === "failed")
        return `The model didn't load: ${model.detail}`;
    return runBlock(state);
}
// Whether a new question or SQL run can start from the page right now.
export function canStart(state) {
    return !isRunning(state.job) && !state.submitting;
}
// Why Query this table can't run now, or null when it can: a run already under way, or the Run button's reason.
export function queryBlock(state, dataset) {
    if (!canStart(state))
        return BUSY;
    return runBlock(state, dataset);
}
