// Calls to the app's server. Every reply is checked against its type before the page sees it.
import { UNREACHABLE, UNREADABLE } from "./copy.js";
import { errorMessage, readHistory, readRowsPage, readSnapshot, readStatus, readStopReply, ShapeError, } from "./read.js";
// A call the server refused, or one that never reached it (status 0).
export class ApiError extends Error {
    status;
    constructor(message, status) {
        super(message);
        this.status = status;
    }
}
export function isAbort(error) {
    return error instanceof DOMException && error.name === "AbortError";
}
// One request and its checked reply; the server's own message when it says no.
async function call(path, read, init = {}) {
    let response;
    try {
        response = await fetch(path, { ...init, cache: "no-store", credentials: "same-origin" });
    }
    catch (error) {
        if (isAbort(error))
            throw error;
        throw new ApiError(UNREACHABLE, 0);
    }
    let body;
    try {
        body = await response.json();
    }
    catch (error) {
        if (isAbort(error))
            throw error;
        throw new ApiError(UNREADABLE, response.status);
    }
    if (!response.ok)
        throw new ApiError(errorMessage(body) ?? UNREADABLE, response.status);
    try {
        return read(body, "reply");
    }
    catch (error) {
        if (!(error instanceof ShapeError))
            throw error;
        console.error(`${path}: ${error.message}`);
        throw new ApiError(UNREADABLE, response.status);
    }
}
function post(path, body, read) {
    return call(path, read, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}
function jobPath(id) {
    return `/api/jobs/${encodeURIComponent(id)}`;
}
export function getStatus() {
    return call("/api/status", readStatus);
}
export function getHistory() {
    return call("/api/history", readHistory);
}
// The job once its version passes after; the server holds the call up to 20 s (-1 answers at once).
export function getJob(id, after, signal) {
    return call(`${jobPath(id)}?after=${after}`, readSnapshot, { signal });
}
export function ask(dataset, question) {
    return post("/api/ask", { dataset, question }, readSnapshot);
}
// Typed SQL; with fromJob it's an edit of that answer's SQL.
export function runSql(dataset, sql, fromJob) {
    return post("/api/sql", fromJob ? { dataset, sql, from_job: fromJob } : { dataset, sql }, readSnapshot);
}
export function reask(job, term, reading) {
    return post("/api/reask", { job, term, reading }, readSnapshot);
}
export function stopJob(id) {
    return post(`${jobPath(id)}/stop`, {}, readStopReply);
}
// The next page of an answer's rows, starting at offset.
export function getRows(id, answer, offset) {
    return call(`${jobPath(id)}/rows?answer=${answer}&offset=${offset}`, readRowsPage);
}
export function csvHref(id, answer) {
    return `${jobPath(id)}/csv?answer=${answer}`;
}
