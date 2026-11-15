// The page's own words. Every sentence about an answer (verdict, checks, terms, the tested line,
// the meta line, the CSV label) comes from the server instead.

// A whole number with en-US grouping, the same text Python's f"{n:,}" gives.
export function count(n: number): string {
  return n.toLocaleString("en-US");
}

export function rowsText(n: number): string {
  return n === 1 ? "1 row" : `${count(n)} rows`;
}

export function aheadText(n: number): string {
  if (n <= 0) return "Starting next.";
  return n === 1 ? "1 job ahead." : `${count(n)} jobs ahead.`;
}

export const EMPTY_HISTORY = "Ask a question or run SQL. Your runs this session show here.";
export const EMPTY_RESULT = "Ask a question in plain English, or switch to SQL and run a query. "
  + "The answer shows here, with the checks beside it.";
export const EMPTY_EVIDENCE = "Checks show here once an answer is back.";
export const NO_ROWS = "No rows.";
export const STOPPED_QUEUED = "Stopped before it started.";
export const ASK_RECOVERY = "Ask again, or pick another question.";
export const SQL_RECOVERY = "Fix the SQL and run it again.";
export const NOT_TESTED = "(not part of the test)";
export const CONNECTING = "Checking what's loaded.";
export const UNREACHABLE = "Can't reach the app. Check that it's still running in its terminal, then reload.";
export const UNREADABLE = "The app sent a reply this page can't read. Reload the page.";
export const NO_SUCH_JOB = "That job isn't in this session. Runs from an earlier launch aren't kept.";
export const TESTED_LINK = "How the checks were tested";
export const RATES_HIDDEN = "The test's rates are hidden, because";
export const SETUP_DIFFERS = "this setup differs from the tested one";
