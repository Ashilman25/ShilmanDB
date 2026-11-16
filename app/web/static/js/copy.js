// The page's own words. Every sentence about an answer (verdict, checks, terms, the tested line,
// the meta line, the CSV label) comes from the server instead.
// A whole number with en-US grouping, the same text Python's f"{n:,}" gives.
export function count(n) {
    return n.toLocaleString("en-US");
}
export function rowsText(n) {
    return n === 1 ? "1 row" : `${count(n)} rows`;
}
// An engine time in words, as the server writes it: whole ms with separators, or "under 1 ms".
export function msText(ms) {
    return ms >= 1 ? `${count(Math.round(ms))} ms` : "under 1 ms";
}
export function aheadText(n) {
    if (n <= 0)
        return "Starting next.";
    return n === 1 ? "1 job ahead." : `${count(n)} jobs ahead.`;
}
// Which rows of a grid are on screen, out of all the rows there are.
export function rowsRange(shown, total) {
    return shown === 0 ? "No rows." : `Rows 1 to ${count(shown)} of ${count(total)}`;
}
export function moreText(n) {
    return `Show ${count(n)} more`;
}
// A preview that holds fewer rows than the table, once all of them are on screen.
export function previewCap(loaded, total) {
    return `Showing the first ${count(loaded)} of ${count(total)} rows. Query the table to filter or sort.`;
}
export function tableMeta(rows, columns) {
    return `${rowsText(rows)}, ${columns === 1 ? "1 column" : `${count(columns)} columns`}`;
}
export const EMPTY_HISTORY = "Your queries and questions this session show here.";
export const EMPTY_EVIDENCE = "Checks show here once an answer is back.";
export const FILTER_LABEL = "Filter tables and columns";
export const NO_MATCH = "No table or column matches.";
export const NO_TABLES = "No tables to show.";
export const QUERY_TABLE = "Query this table";
export const START_FROM_TABLE = "Start from a table";
export const START_HINT = "Click one to put it in the editor, then run it.";
export const EXAMPLES_HINT = "Each answer comes back with its checks in the Checks panel: the second answer, the engine's evidence, the terms and the SQL.";
export const NO_ROWS = "No rows.";
export const STOPPED_QUEUED = "Stopped before it started.";
export const ASK_RECOVERY = "Ask again, or pick another question.";
export const SQL_RECOVERY = "Fix the SQL and run it again.";
export const NOT_TESTED = "(not part of the test)";
export const CONNECTING = "Checking what's loaded.";
export const UNREACHABLE = "Can't reach the app. Check that it's still running in its terminal, then reload.";
export const UNREADABLE = "The app sent a reply this page can't read. Reload the page.";
export const BUSY = "A run is under way in the Query tab.";
export const SCHEMA_RETRY = "Couldn't load the tables. Retrying.";
export const NO_SUCH_JOB = "That job isn't in this session. Runs from an earlier launch aren't kept.";
export const TESTED_LINK = "How the checks were tested";
export const RATES_HIDDEN = "The test's rates are hidden, because";
export const SETUP_DIFFERS = "this setup differs from the tested one";
export function examplesTitle(label) {
    return `Example questions for ${label}`;
}
export function startSql(table) {
    return `SELECT * FROM ${table} LIMIT 100`;
}
