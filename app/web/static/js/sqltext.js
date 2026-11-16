// SQL as the editor reads it, with no page involved: the text cut into colored pieces, and the table and
// column names to offer for the word before the caret.
export const MAX_CHOICES = 8;
const ALIAS_MAX = 2; // letters right after a table name that read as its alias, not a name being typed
export const KEYWORDS = new Set(`
  select from where group by order having limit offset join inner left right full outer cross on using as and or
  not in is null like between case when then else end distinct all union intersect except exists asc desc with
  count sum avg min max cast extract interval date year month day true false integer bigint decimal varchar
`.trim().split(/\s+/));
// Comments, strings, quoted names, numbers, then bare words; everything else stays plain.
const TOKEN = /(--[^\n]*|\/\*[\s\S]*?(?:\*\/|$))|('(?:[^']|'')*'?)|("(?:[^"]|"")*"?)|(\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)|([A-Za-z_][A-Za-z0-9_$]*)/g;
// The text cut into colored pieces: comments, strings, numbers and keywords each get a class.
export function pieces(text) {
    const out = [];
    let at = 0;
    for (const found of text.matchAll(TOKEN)) {
        const start = found.index ?? 0;
        if (start > at)
            out.push([null, text.slice(at, start)]);
        const kind = found[1] ? "c" : found[2] ? "s" : found[4] ? "n" : found[5] && KEYWORDS.has(found[5].toLowerCase()) ? "k" : null;
        out.push([kind, found[0]]);
        at = start + found[0].length;
    }
    if (at < text.length)
        out.push([null, text.slice(at)]);
    return out;
}
// The names that start with the word before the caret, case aside; after "table." only that table's columns.
// Nothing inside a string or a comment, right after a digit, for a keyword or a whole name, or for one or two
// letters right after a table name, which read as its alias.
export function namesAt(text, caret, names) {
    const before = text.slice(0, caret);
    const last = pieces(before).at(-1);
    if (last && (last[0] === "c" || (last[0] === "s" && !/^'(?:[^']|'')*'$/.test(last[1]))))
        return null;
    const found = /(?:([A-Za-z_][A-Za-z0-9_]*)\.)?([A-Za-z_][A-Za-z0-9_]*)?$/.exec(before);
    const word = found?.[2] ?? "";
    const table = found?.[1]?.toLowerCase() ?? null;
    const from = caret - word.length;
    const lead = before.slice(0, from - (table === null ? 0 : table.length + 1));
    if ((!word && table === null) || /[0-9]$/.test(lead))
        return null;
    const lower = word.toLowerCase();
    const tables = new Set(names.filter((name) => name.table === null).map((name) => name.name.toLowerCase()));
    const after = /([A-Za-z_][A-Za-z0-9_]*)\s+(?:as\s+)?$/i.exec(lead)?.[1]?.toLowerCase();
    if (table === null && after !== undefined && tables.has(after) && word.length <= ALIAS_MAX)
        return null;
    const pool = table === null ? names : names.filter((name) => name.table?.toLowerCase() === table);
    if (KEYWORDS.has(lower) || pool.some((name) => name.name.toLowerCase() === lower))
        return null;
    const choices = pool.filter((name) => name.name.toLowerCase().startsWith(lower)).slice(0, MAX_CHOICES);
    return choices.length ? { from, choices } : null;
}
