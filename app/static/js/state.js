// What the page holds between redraws, the actions the panels can call, and the rules
// derived from them (is the shown job running, why Ask or Run is off).
import { CONNECTING } from "./copy.js";
export const SECOND_ROW = "second_phrasing";
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
    return state.status?.datasets[name]?.label ?? name;
}
// Why typed SQL can't run on the current dataset, or null when it can.
export function runBlock(state) {
    if (state.statusError)
        return state.statusError;
    const info = state.status?.datasets[state.dataset];
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
