// Calls to the app's server. Every reply is checked against its type before the page sees it.
import { UNREACHABLE, UNREADABLE } from "./copy.js";
import {
  errorMessage, readHistory, readRowsPage, readSnapshot, readStatus, readStopReply, ShapeError, type Read,
} from "./read.js";
import type { AnswerName, HistoryEntry, RowsPage, Snapshot, Status, StopReply } from "./types.js";

// A call the server refused, or one that never reached it (status 0).
export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

export function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

// One request and its checked reply; the server's own message when it says no.
async function call<T>(path: string, read: Read<T>, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, { ...init, cache: "no-store", credentials: "same-origin" });
  } catch (error) {
    if (isAbort(error)) throw error;
    throw new ApiError(UNREACHABLE, 0);
  }

  let body: unknown;
  try {
    body = await response.json();
  } catch (error) {
    if (isAbort(error)) throw error;
    throw new ApiError(UNREADABLE, response.status);
  }
  if (!response.ok) throw new ApiError(errorMessage(body) ?? UNREADABLE, response.status);

  try {
    return read(body, "reply");
  } catch (error) {
    if (!(error instanceof ShapeError)) throw error;
    console.error(`${path}: ${error.message}`);
    throw new ApiError(UNREADABLE, response.status);
  }
}

function post<T>(path: string, body: Record<string, string>, read: Read<T>): Promise<T> {
  return call(path, read, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

function jobPath(id: string): string {
  return `/api/jobs/${encodeURIComponent(id)}`;
}

export function getStatus(): Promise<Status> {
  return call("/api/status", readStatus);
}

export function getHistory(): Promise<HistoryEntry[]> {
  return call("/api/history", readHistory);
}

// The job once its version passes after; the server holds the call up to 20 s (-1 answers at once).
export function getJob(id: string, after: number, signal: AbortSignal): Promise<Snapshot> {
  return call(`${jobPath(id)}?after=${after}`, readSnapshot, { signal });
}

export function ask(dataset: string, question: string): Promise<Snapshot> {
  return post("/api/ask", { dataset, question }, readSnapshot);
}

// Typed SQL; with fromJob it's an edit of that answer's SQL.
export function runSql(dataset: string, sql: string, fromJob: string | null): Promise<Snapshot> {
  return post("/api/sql", fromJob ? { dataset, sql, from_job: fromJob } : { dataset, sql }, readSnapshot);
}

export function reask(job: string, term: string, reading: string): Promise<Snapshot> {
  return post("/api/reask", { job, term, reading }, readSnapshot);
}

export function stopJob(id: string): Promise<StopReply> {
  return post(`${jobPath(id)}/stop`, {}, readStopReply);
}

// The next page of an answer's rows, starting at offset.
export function getRows(id: string, answer: AnswerName, offset: number): Promise<RowsPage> {
  return call(`${jobPath(id)}/rows?answer=${answer}&offset=${offset}`, readRowsPage);
}

export function csvHref(id: string, answer: AnswerName): string {
  return `${jobPath(id)}/csv?answer=${answer}`;
}
