"""Tests for the page: app/web/static/ and the TypeScript it's built from in app/web/src/.

They read files, and run the one compiled module that touches no page (sqltext.js) in Node when Node is
installed; the build itself (tsc) runs in CI, never here."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

from app.server import copy_rules, jobs, views
from app.tests.test_app_views import REGION_SQL, _region_reply

APP_DIR = Path(__file__).resolve().parents[1]
WEB = APP_DIR / "web"
STATIC = WEB / "static"
INDEX, CSS = STATIC / "index.html", STATIC / "app.css"
SOURCES, COMPILED = WEB / "src", STATIC / "js"
TSCONFIG = WEB / "tsconfig.json"

HTML_SINKS = ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write")
ADDRESSES = ("http://", "https://")
# The SVG namespace is a name, not an address: createElementNS needs it and nothing is fetched.
SVG_NAMESPACE = "http://www.w3.org/2000/svg"

# The dark theme's tokens, which :root must define.
DARK = {"--paper": "#131518", "--panel": "#1b1e22", "--line": "#2e3238", "--ink": "#e4e6e9",
        "--muted": "#9aa1a9", "--accent": "#7fb2dc"}
MIN_FONT_PX = 12


# The TypeScript sources; never empty once the page is built.
def _sources() -> list[Path]:
    found = sorted(SOURCES.glob("*.ts"))
    assert found, f"no TypeScript in {SOURCES}"
    return found


# The compiled modules the page loads.
def _compiled() -> list[Path]:
    found = sorted(COMPILED.glob("*.js"))
    assert found, f"no compiled JS in {COMPILED}"
    return found


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# Code with every run of whitespace as one space, so a check reads the TypeScript and tsc's output alike.
def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


# Every tag in a page with its attributes, plus the text inside each script and style element.
class _Tags(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict]] = []
        self.inline: dict[str, str] = {"script": "", "style": ""}
        self._open: str | None = None

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        self._open = tag if tag in self.inline else None

    def handle_startendtag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_endtag(self, tag):
        self._open = None

    def handle_data(self, data):
        if self._open:
            self.inline[self._open] += data


def _tags(text: str) -> _Tags:
    parser = _Tags()
    parser.feed(text)
    parser.close()
    return parser


# The body of the first CSS rule whose selector list matches exactly; comments don't count.
def _rule(css: str, selector: str) -> str:
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    for found in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        if found.group(1).strip() == selector:
            return found.group(2)
    raise AssertionError(f"no rule for {selector} in app.css")


# The text inside one @media block, so a rule can be read as it applies at that width.
def _media(css: str, query: str) -> str:
    start = css.index(f"@media {query} {{") + len(f"@media {query} {{")
    depth, i = 1, start
    while depth:
        depth += {"{": 1, "}": -1}.get(css[i], 0)
        i += 1
    return css[start:i - 1]


# The custom properties a rule body sets, name to value.
def _tokens(body: str) -> dict[str, str]:
    return {name: value.strip().lower() for name, value in re.findall(r"(--[\w-]+)\s*:\s*([^;]+);?", body)}


# The top-level fields of one TypeScript interface (nested object types are never used there).
def _interface_fields(source: str, name: str) -> set[str]:
    start = re.search(rf"\binterface\s+{name}\s*\{{", source)
    assert start, f"no interface {name} in types.ts"
    depth, fields, i = 1, set(), start.end()
    line_start = i
    while depth and i < len(source):
        char = source[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif char in "\n;" and depth == 1:
            field = re.match(r"\s*(?:readonly\s+)?(\w+)\??\s*:", source[line_start:i])
            if field:
                fields.add(field.group(1))
            line_start = i + 1
        i += 1
    return fields


def test_static_files_pass_copy_rules():
    files = [INDEX, CSS, *_sources(), *_compiled()]
    found = {path.relative_to(APP_DIR).as_posix(): copy_rules.problems(_read(path)) for path in files}
    assert {name: problems for name, problems in found.items() if problems} == {}


def test_page_never_sets_html_from_data():
    for path in (*_sources(), *_compiled()):
        text = _read(path)
        assert [sink for sink in HTML_SINKS if sink in text] == [], path.name


def test_page_is_offline():
    for path in (INDEX, CSS, *_compiled()):
        text = _read(path).replace(SVG_NAMESPACE, "")
        assert [a for a in ADDRESSES if a in text] == [], path.name


def test_index_loads_one_module():
    page = _tags(_read(INDEX))
    scripts = [attrs for tag, attrs in page.tags if tag == "script"]
    assert scripts == [{"type": "module", "src": "/static/js/main.js"}]
    assert page.inline["script"].strip() == ""


# The server's Content-Security-Policy refuses inline styles and handlers, so the page has none.
def test_index_has_no_inline_style_or_handlers():
    page = _tags(_read(INDEX))
    assert [tag for tag, _ in page.tags if tag == "style"] == []
    assert page.inline["style"] == ""
    for tag, attrs in page.tags:
        assert "style" not in attrs, tag
        assert [name for name in attrs if name.startswith("on")] == [], tag
    sheets = [attrs.get("href") for tag, attrs in page.tags if tag == "link" and attrs.get("rel") == "stylesheet"]
    assert sheets == ["/static/app.css"]


# The text fields live in the page itself: SQL is never spellchecked, capitalized or corrected.
def test_sql_editor_is_plain_text():
    fields = {attrs.get("id"): (tag, attrs) for tag, attrs in _tags(_read(INDEX)).tags if attrs.get("id") in ("question", "sql")}
    tag, sql = fields["sql"]
    assert tag == "textarea"
    assert {name: sql.get(name) for name in ("spellcheck", "autocapitalize", "autocorrect", "autocomplete")} == {
        "spellcheck": "false", "autocapitalize": "off", "autocorrect": "off", "autocomplete": "off"}
    tag, question = fields["question"]
    assert (tag, question.get("type"), question.get("autocomplete")) == ("input", "text", "off")


# A full-height app frame with no page scroll on desktop: the sidebar's tree, the results, the inspector
# and each grid scroll on their own. The results, both inspectors and the narrow layout's pane always keep
# their scrollbar's room, so an answer or a verdict that makes them scroll never rewraps what's there.
def test_page_is_an_app_shell():
    css = _read(CSS)
    assert "overflow: hidden" in _rule(css, "body")
    assert "grid-template-rows: minmax(0, 1fr)" in _rule(css, ".app")
    assert "grid-template-columns: 272px minmax(0, 1fr)" in _rule(css, ".app")
    assert "overflow-y: auto" in _rule(css, ".side-scroll")
    for selector in (".results", ".ins-body"):
        assert "overflow-y: scroll" in _rule(css, selector), selector
    assert "overflow-y: scroll" in _rule(_media(css, "(max-width: 1100px)"), ".pane, .pane.query")
    assert "overflow-y: scroll" in _rule(_media(css, "(max-width: 760px)"), "html")
    assert "overflow: auto" in _rule(css, ".grid-scroll")


# The grid holds its row numbers and its header while it scrolls both ways; rows stay one 28px line,
# and a cell past 360px is cut with an ellipsis and keeps its value in a tooltip.
def test_grid_holds_row_numbers_and_header():
    css = _read(CSS)
    head = _rule(css, ".grid-table thead th")
    assert "position: sticky" in head and "top: 0" in head and "background: var(--panel)" in head
    numbers = _rule(css, ".grid-table .rn")
    assert "position: sticky" in numbers and "left: 0" in numbers and "text-align: right" in numbers
    cells = _rule(css, ".grid-table th, .grid-table td")
    assert "white-space: nowrap" in cells and "height: 28px" in cells and "padding: 0 10px" in cells
    assert "border-right: 1px solid var(--line)" in cells and "border-bottom: 1px solid var(--line)" in cells
    cut = _rule(css, ".grid-table td .cell")
    assert "max-width: 340px" in cut and "text-overflow: ellipsis" in cut  # 340px plus 20px of padding
    assert "font: 12px/1.3 var(--mono)" in _rule(css, ".col-type")
    for path in (SOURCES / "grid.ts", COMPILED / "grid.js"):
        code = _read(path)
        assert 'class: "rn", attrs: { scope: "row" }' in code, path.name  # a row number heads each row
        assert 'role: "grid"' in code and "aria-activedescendant" in code, path.name
        assert "col-name" in code and "col-type" in code, path.name  # each header: the name, its type under it


# The Checks inspector holds its layout on a question: the verdict line keeps two lines in every stage, and the
# second answer's row keeps five lines in the 380px inspector (three when it sits under the grid, full width).
def test_question_panel_holds_its_layout():
    css = _read(CSS)
    assert "min-height: calc(2 * 1.45em)" in _rule(css, ".verdict.reserve")
    assert "min-height: calc(5 * 1.45em)" in _rule(css, ".entry.second")
    assert "min-height: calc(3 * 1.45em)" in _rule(_media(css, "(max-width: 1100px)"), ".entry.second")


# Section headings are 15 px semibold, a clear step above the 14 px verdict line and the 12 px status words.
def test_section_headings_are_15px():
    css = _read(CSS)
    for selector in (".ins-head h2", ".ins-body h2", ".pair > h2", ".sheet-head h2", ".starter h2"):
        body = _rule(css, selector)
        assert "font-size: 15px" in body and "font-weight: 600" in body, selector
    assert "font-size: 12px" in _rule(css, ".word")


# Finish details: the open table's tree row and a picked cell take the select tint (the cell with a 1px inset
# blue outline), every number is in tabular figures, no layout comes from a :has() rule, and scrollbar thumbs
# draw no paper-coloured ring on panels.
def test_finish_details():
    css = _read(CSS)
    selected = _rule(css, '.tree-item[aria-selected="true"] > .tree-row')
    assert "background: var(--select)" in selected and "color: var(--ink)" in selected
    picked = _rule(css, ".grid-table td.picked, .grid-table tbody tr:hover td.picked")
    assert "background: var(--select)" in picked and "outline: 1px solid var(--accent)" in picked and "outline-offset: -1px" in picked
    assert "font-variant-numeric: tabular-nums" in _rule(css, "body")
    plain = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    assert ":has(" not in plain
    thumb = _rule(css, "::-webkit-scrollbar-thumb")
    assert "solid transparent" in thumb and "background-clip: padding-box" in thumb


# The regions keep their sizes: the 44px top bar, toolbar and inspector header, the 272px sidebar,
# the 36px tab strip, the 28px status bar, the 32px grid footer, and the 340px and 380px inspectors.
def test_layout_sizes():
    css = _read(CSS)
    assert "height: 44px" in _rule(css, ".top")
    assert "height: 44px" in _rule(css, ".toolbar")
    assert "height: 44px" in _rule(css, ".ins-head")
    assert "height: 36px" in _rule(css, ".tabs") and "height: 36px" in _rule(css, ".side-tabs")
    assert "min-height: 28px" in _rule(css, ".statusbar")
    assert "min-height: 32px" in _rule(css, ".grid-foot")
    assert "grid-template-columns: minmax(0, 1fr) 340px" in _rule(css, ".pane")
    assert "grid-template-columns: minmax(0, 1fr) 380px" in _rule(css, ".pane.query")
    assert "height: var(--code-h, 140px)" in _rule(css, ".sql-box")  # 140px until the splitter moves it
    narrow = _rule(_media(css, "(max-width: 1100px)"), ".inspector")
    assert "border-top: 1px solid var(--line)" in narrow  # under 1100px the inspector moves under the grid


# The page also runs in Safari 16's engine (the Mac app's web view), so the stylesheet avoids what it lacks.
def test_css_uses_nothing_safari_16_lacks():
    css = re.sub(r"/\*.*?\*/", "", _read(CSS), flags=re.DOTALL)
    missing = ("scrollbar-gutter", "scrollbar-color", "scrollbar-width", "color-mix(", "light-dark(", "@starting-style",
               "text-wrap", "field-sizing", ":user-invalid", "view-transition", "@scope")
    assert [name for name in missing if name in css] == []
    assert not re.search(r"\{[^{}]*&", css)  # no nesting


def test_every_source_has_compiled_js():
    assert {p.stem for p in _sources()} == {p.stem for p in _compiled()}


def test_typescript_pinned():
    package = json.loads(_read(WEB / "package.json"))
    for field in ("dependencies", "peerDependencies", "optionalDependencies", "bundleDependencies"):
        assert field not in package, field
    assert package["devDependencies"] == {"typescript": "7.0.2"}

    lock = json.loads(_read(WEB / "package-lock.json"))
    assert lock["packages"][""]["devDependencies"] == {"typescript": "7.0.2"}
    assert lock["packages"]["node_modules/typescript"]["version"] == "7.0.2"


# tsc writes into outDir, from app/web/; CI checks COMPILED, so the two must agree.
def test_tsconfig_builds_into_compiled():
    out_dir = json.loads(_read(TSCONFIG))["compilerOptions"]["outDir"]
    assert (WEB / out_dir).resolve() == COMPILED.resolve()


def test_dark_is_default():
    css = _read(CSS)
    dark = _tokens(_rule(css, ":root"))
    assert {name: dark.get(name) for name in DARK} == DARK

    light = _tokens(_rule(css, ':root[data-theme="light"]'))
    assert set(DARK) <= set(light)
    assert all(light[name] != dark[name] for name in DARK)
    assert "prefers-color-scheme" not in css  # dark by decision, whatever the OS says

    sizes = re.findall(r"font(?:-size)?\s*:[^;{}]*?(\d+(?:\.\d+)?)px", css)
    assert sizes and min(float(s) for s in sizes) >= MIN_FONT_PX
    other_units = re.findall(r"font-size\s*:\s*([^;{}]+)", css)
    assert [u for u in other_units if not u.strip().endswith("px") and u.strip() != "inherit"] == []


def test_snapshot_type_matches_server():
    types = _read(SOURCES / "types.ts")
    assert _interface_fields(types, "Snapshot") == set(jobs.SNAPSHOT_KEYS)
    assert _interface_fields(types, "AnswerView") == set(views.answer_view(_region_reply(), REGION_SQL))


# The numbers the page shows as text come from the server: the plan's times and the job's seconds.
def test_number_types_match_server():
    types = _read(SOURCES / "types.ts")
    assert _interface_fields(types, "PlanRow") == set(views.plan_rows(_region_reply()["profile"])[0])
    timing = jobs.Job.__dataclass_fields__["timing"].default_factory()
    assert _interface_fields(types, "Timing") == set(timing)


# WCAG relative luminance of a #rrggbb color.
def _luminance(color: str) -> float:
    channels = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a: str, b: str) -> float:
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


# The text fields' border is what outlines them, so it keeps 3:1 against the paper and the panel
# in both themes (WCAG 1.4.11). Hover still goes to muted; nothing else uses the token.
def test_text_field_border_has_3_to_1():
    css = _read(CSS)
    for block in (":root", ':root[data-theme="light"]'):
        tokens = _tokens(_rule(css, block))
        for ground in ("--paper", "--panel"):
            assert _contrast(tokens["--field"], tokens[ground]) >= 3.0, (block, ground)
    assert "border: 1px solid var(--field)" in _rule(css, ".field-line, .field-sql, .filter")
    assert "border-color: var(--muted)" in _rule(css, ".field-line:hover, .field-sql:hover, .filter:hover")
    assert css.count("var(--field)") == 1


# Under 760 px the stacked sidebar is capped and scrolls on its own, so the tabs stay near the top.
def test_stacked_sidebar_is_capped():
    sidebar = _rule(_media(_read(CSS), "(max-width: 760px)"), ".sidebar")
    assert "max-height: 35vh" in sidebar and "overflow-y: auto" in sidebar
    assert "border-bottom: 1px solid var(--line)" in sidebar


# A hovered thumb keeps its clear ring (background-color leaves background-clip alone) and a 4px corner.
def test_scrollbar_thumb_keeps_its_ring():
    css = _read(CSS)
    hover = _rule(css, "::-webkit-scrollbar-thumb:hover")
    assert "background-color: var(--muted)" in hover and not re.search(r"background\s*:", hover)
    assert "border-radius: 4px" in _rule(css, "::-webkit-scrollbar-thumb")


# The banner under the input (a dataset that isn't available) keeps the lone result column's measure.
def test_under_banner_keeps_a_readable_measure():
    assert "max-width: 66ch" in _rule(_read(CSS), ".under > .banner")


# Show more stays focusable while its rows load: aria-disabled, drawn like a disabled button, no hover.
def test_loading_button_looks_disabled():
    css = _read(CSS)
    assert "color: var(--muted)" in _rule(css, '.btn2[aria-disabled="true"]')
    assert _rule(css, '.btn2:hover:not(:disabled):not([aria-disabled="true"])')


# Bars lying across take 22px rows, and 16px rows from 13 bars on, so a 50-bar chart stays near 800px.
def test_dense_bar_rows():
    chart = _read(SOURCES / "chart.ts")
    assert re.search(r"const ROW = 22;", chart) and re.search(r"const DENSE_ROW = 16;", chart)
    assert re.search(r"const DENSE_FROM = 13;", chart)
    labels = _rule(_read(CSS), ".dense .row-labels span")
    assert "height: 16px" in labels and "line-height: 16px" in labels


# In mlx mode the status bar shows the prompt cache: on, or off with its reason as visible text.
def test_status_bar_draws_the_cache_state():
    for path in (SOURCES / "statusbar.ts", COMPILED / "statusbar.js"):
        code = _read(path)
        assert re.search(r'mode [!=]== "mlx"', code) and "status.cache" in code, path.name
        assert '"Prompt cache on"' in code and "`Prompt cache off: ${" in code, path.name


# The status bar names each dataset's engine; missing files or a failed load are named as data,
# Engine missing is the bridge binary's. Items are split by a 1px rule, never a character.
def test_status_bar_names_data_problems():
    for path in (SOURCES / "statusbar.ts", COMPILED / "statusbar.js"):
        code = _read(path)
        assert '"Data missing"' in code and '"Data failed"' in code and '"Engine missing"' in code, path.name
        assert '"Missing:"' in code, path.name  # the server's prefix for missing dataset files
        assert "status.datasets" in code and "Model ${" in code, path.name
    assert "border-left: 1px solid var(--line)" in _rule(_read(CSS), ".sb-item + .sb-item")
    assert "color: var(--bad)" in _rule(_read(CSS), ".state.bad")


# The empty Query tab teaches: example questions in Ask mode, a first query per table in SQL mode;
# started with --no-model it's always the SQL one.
def test_empty_query_tab_follows_the_model():
    copy = _read(SOURCES / "copy.ts")
    assert 'export const START_FROM_TABLE = "Start from a table";' in copy
    assert "return `Example questions for ${label}`;" in copy
    assert "return `SELECT * FROM ${table} LIMIT 100`;" in copy
    for path in (SOURCES / "result.ts", COMPILED / "result.js"):
        assert re.search(r'model\.state === "off" \? "sql" : state\.mode', _read(path)), path.name


# A call that never reaches the server rechecks the status, so the status bar says Not connected.
def test_lost_server_rechecks_the_status():
    for path in (SOURCES / "main.ts", COMPILED / "main.js"):
        code = _read(path)
        found = re.search(r"function noticeGone\(error[^)]*\)[^{]*\{(.*?)\n\}", code, re.DOTALL)
        assert found and "status === 0" in found.group(1) and "loadStatus()" in found.group(1), path.name
        assert code.count("noticeGone(error)") >= 7, path.name  # follow, start, stop, show more, history, schema, preview


# The picked option of a normal switch: ink at 600 and an inset 1px muted ring, at least 3:1 on the
# paper and the panel in both themes. Hover fills with the checked panel tone, never past it.
def test_picked_switch_has_a_3_to_1_cue():
    css = _read(CSS)
    checked = _rule(css, ".seg input:checked + span")
    assert "font-weight: 600" in checked and "box-shadow: inset 0 0 0 1px var(--muted)" in checked
    assert "background: var(--panel)" in checked and "color: var(--ink)" in checked
    assert "background: var(--panel)" in _rule(css, ".seg:not(.strong) input:hover + span")
    strong = _rule(css, ".seg.strong input:checked + span")
    assert "box-shadow: none" in strong and "background: var(--btnbg)" in strong
    for block in (":root", ':root[data-theme="light"]'):
        tokens = _tokens(_rule(css, block))
        for ground in ("--paper", "--panel"):
            assert _contrast(tokens["--muted"], tokens[ground]) >= 3.0, (block, ground)


# The schema sidebar is a real tree: treeitems that open and close, groups under them, and the arrow keys.
def test_tree_is_a_real_tree():
    for path in (SOURCES / "sidebar.ts", COMPILED / "sidebar.js"):
        code = _read(path)
        assert 'role: "tree"' in code and 'role: "treeitem"' in code and 'role: "group"' in code, path.name
        assert '"aria-expanded"' in code and '"aria-selected"' in code and '"aria-level"' in code, path.name
        for key in ("ArrowDown", "ArrowUp", "ArrowRight", "ArrowLeft", "Enter"):
            assert f'"{key}"' in code, (path.name, key)
        assert "No table or column matches." in _read(SOURCES / "copy.ts")


# The tab strip is a tablist: Query first and never closable, at most eight table tabs, arrow keys between
# tabs, each close mark named for its table.
def test_tab_strip_is_a_tablist():
    tabs = [attrs for tag, attrs in _tags(_read(INDEX)).tags if attrs.get("id") == "tabs"]
    assert tabs and tabs[0]["role"] == "tablist"
    panes = {attrs.get("id"): attrs for tag, attrs in _tags(_read(INDEX)).tags if attrs.get("role") == "tabpanel"}
    assert set(panes) == {"query-pane", "table-pane"}
    for path in (SOURCES / "tabs.ts", COMPILED / "tabs.js"):
        code = _read(path)
        assert 'role: "tab"' in code and '"aria-selected"' in code and '"aria-controls"' in code, path.name
        assert '"ArrowRight"' in code and '"ArrowLeft"' in code, path.name
        assert 'cell(QUERY_TAB, "Query", false)' in code, path.name
        assert '"aria-label": `Close ${label}`' in code and '"auxclick"' in code, path.name
    assert "export const MAX_TABLE_TABS = 8;" in _read(SOURCES / "state.ts")


# The page opens on the start table of the active database: orders for TPC-H, reviews for the reviews.
def test_start_tables_are_orders_and_reviews():
    state = _read(SOURCES / "state.ts")
    assert 'START_TABLES: Record<string, string> = { tpch: "orders", reviews: "reviews" };' in state
    for name, table in (("tpch", "orders"), ("reviews", "reviews")):
        spec = json.loads((APP_DIR.parent / f"bench/analyst/datasets/{name}.json").read_text(encoding="utf-8"))
        assert table in spec["primary_keys"], name
    for path in (SOURCES / "main.ts", COMPILED / "main.js"):
        code = _read(path)
        assert "START_TABLES[state.dataset]" in code and 'START_DATASET = "tpch"' in code, path.name


# The only drawn marks: the tree's chevron and a tab's close mark, authored inline in one 1.5px stroke,
# in currentColor and hidden from screen readers. Nothing else draws an SVG except the chart.
def test_icons_are_the_two_drawn_marks():
    controls = _read(SOURCES / "controls.ts")
    assert controls.count('svg("path"') == 1 and re.findall(r'return mark\("([\w-]+)"', controls) == ["chevron", "close-mark"]
    assert '"stroke-width": 1.5' in controls and 'stroke: "currentColor"' in controls and '"aria-hidden": "true"' in controls
    drawing = [path.name for path in _sources() if "svg(" in _read(path)]
    assert sorted(drawing) == ["chart.ts", "controls.ts"]


# The schema and preview replies are read field by field into types that match the server's.
def test_schema_and_preview_types_match_server():
    types = _read(SOURCES / "types.ts")
    assert _interface_fields(types, "Preview") == {"dataset", "table", "columns", "rows", "total", "ms"}
    assert _interface_fields(types, "SchemaTable") == {"name", "rows", "primary_key", "columns"}
    assert _interface_fields(types, "SchemaColumn") == {"name", "type", "description", "key", "references", "values"}
    assert _interface_fields(types, "DatasetSchema") == {"label", "state", "detail", "about", "tables"}


# A #rrggbb color under rgba(r, g, b, a): what the eye sees where the select tint covers that ground.
def _over(rgba: str, ground: str) -> str:
    r, g, b, a = (float(x) for x in re.findall(r"[\d.]+", rgba))
    base = [int(ground[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(a * top + (1 - a) * low):02x}" for top, low in zip((r, g, b), base))


# The select tint marks the open table, a picked column and a picked cell: ink on it keeps 4.5:1 and the
# picked cell's blue outline 3:1 on paper and panel in both themes. Muted text and status words on the tint
# turn ink, since grey, red and amber there can fall under 4.5:1.
def test_select_tint_keeps_text_readable():
    css = _read(CSS)
    for block in (":root", ':root[data-theme="light"]'):
        tokens = _tokens(_rule(css, block))
        for ground in ("--paper", "--panel"):
            tinted = _over(tokens["--select"], tokens[ground])
            assert _contrast(tokens["--ink"], tinted) >= 4.5, (block, ground)
            assert _contrast(tokens["--accent"], tinted) >= 3.0, (block, ground)
    assert "color: var(--ink)" in _rule(css, ".hist.on .hist-kind, .hist.on .word, .hist.on .hist-note")
    assert "color: var(--ink)" in _rule(css, ".grid-table th.picked .col-type")
    assert "color: var(--ink)" in _rule(css, ".col-entry.picked .col-entry-key, .col-entry.picked .col-entry-type, "
                                             ".col-entry.picked .col-entry-ref, .col-entry.picked .col-entry-values")


# Query this table follows the Run button: off while a run is under way or its dataset isn't ready, with the
# reason in text beside it, and queryTable returns before it touches the editor, the mode or the dataset.
def test_query_this_table_waits_for_a_run():
    state = _read(SOURCES / "state.ts")
    assert re.search(r"export function queryBlock\(state: State, dataset: string\): string \| null \{\n"
                     r"  if \(!canStart\(state\)\) return BUSY;\n  return runBlock\(state, dataset\);", state)
    for path in (SOURCES / "main.ts", COMPILED / "main.js"):
        body = re.search(r"queryTable\(dataset, table\) \{\n(.*?)\n\s*\},", _read(path), re.DOTALL)
        assert body, path.name
        lines = [line.strip() for line in body.group(1).splitlines()]
        assert _flat(" ".join(lines[:2])).startswith("if (queryBlock(state, dataset) !== null) return;"), path.name
    for path in (SOURCES / "tabletab.ts", COMPILED / "tabletab.js"):
        code = _read(path)
        assert "queryBlock(state, tab.dataset)" in code, path.name
        assert "disabled: blocked !== null" in code and '"aria-describedby": blocked ? why : null' in code, path.name


# A run that ended without numbers to judge (failed, not run, refused, can't answer, stopped) leads the Checks
# inspector with the result banner's own words, so a Pass under it never reads as approval of a failed run.
def test_ended_runs_lead_the_checks_with_their_verdict():
    for path in (SOURCES / "evidence.ts", COMPILED / "evidence.js"):
        code = _read(path)
        ended = re.search(r"function endedWords\(job[^)]*\)[^{]*\{(.*?)\n\}", code, re.DOTALL)
        assert ended, path.name
        body = ended.group(1)
        assert 'job.stage === "error"' in body and "failedSentence(job)" in body, path.name
        assert "BANNER_IS_RESULT.includes(" in body and "STOPPED_QUEUED" in body, path.name
        line = re.search(r"function verdictLine\(job[^)]*\)[^{]*\{(.*?)\n\}", code, re.DOTALL).group(1)
        assert line.strip().startswith("const ended = endedWords(job);"), path.name
        panel = re.search(r"const parts[^=]*= \[\s*(\w+),", code)
        assert panel and panel.group(1) == "line", path.name  # the verdict line comes first
    assert "font-weight: 400" in _rule(_read(CSS), ".verdict.ended")


# The results header never repeats the SQL (it's in the editor above and the inspector): typed SQL gets only the
# meta and timing lines, a question keeps itself as the header. A one-row, one-column answer shows its value once.
def test_results_header_never_repeats_the_sql():
    for path in (SOURCES / "result.ts", COMPILED / "result.js"):
        code = _read(path)
        assert "rtitle code" not in code and "firstLine(job.text)" not in code, path.name
        assert re.search(r'h\("h1", \{ class: "rtitle" \}, job\.asked\?\.question \?\? job\.text\)', code), path.name
        assert re.search(r'view\.chart\.kind === "number" && view\.columns\.length === 1', code), path.name
        figure = re.search(r"function figureView\([^)]*\)[^{]*\{(.*?)\n\}", code, re.DOTALL)
        assert figure and "grid(" not in figure.group(1) and "csvLink(" in figure.group(1), path.name
    assert ".rtitle.code" not in _read(CSS)


# At phone width the top bar wraps with Dark / Light in view, the bars wrap, and nothing scrolls sideways.
def test_phone_width_wraps_the_bars():
    narrow = _media(_read(CSS), "(max-width: 760px)")
    top = _rule(narrow, ".top")
    assert "flex-wrap: wrap" in top and "height: auto" in top
    assert "flex-wrap: wrap" in _rule(narrow, ".top-end")
    assert "white-space: nowrap" in _rule(narrow, ".top .link")
    assert "flex-wrap: wrap" in _rule(narrow, ".editor-bar, .toolbar")


# The SQL editor stays a plain textarea; its colors are an overlay behind it built from text nodes, set in
# the same font, size, line height, padding and wrap, scrolled with it, with the line numbers in a gutter.
def test_sql_editor_draws_its_colors_behind_the_text():
    css = _read(CSS)
    shared = _rule(css, ".code-lines, .field-sql")
    for part in ("font: 13px/1.5 var(--mono)", "white-space: pre-wrap", "overflow-wrap: break-word", "tab-size: 4",
                 "letter-spacing: 0", "padding: 8px calc(12px + var(--bar, 0px)) 8px var(--gutter)"):
        assert part in shared, part
    text = _rule(css, ".field-sql")
    assert "color: transparent" in text and "background: transparent" in text and "resize: none" in text
    assert "overflow-y: scroll" in text  # the scrollbar's room is kept, so the overlay wraps where the text does
    lines = _rule(css, ".code-lines")
    assert "overflow: hidden" in lines and "pointer-events: none" in lines
    assert "color: var(--muted)" in _rule(css, ".code-lines .ln") and "text-align: right" in _rule(css, ".code-lines .ln")
    assert "font-weight: 600" in _rule(css, ".code-lines .k")
    assert "color: var(--sql-string)" in _rule(css, ".code-lines .s")
    assert "color: var(--sql-number)" in _rule(css, ".code-lines .n")
    page = _tags(_read(INDEX))
    overlay = [attrs for tag, attrs in page.tags if attrs.get("id") == "code-lines"]
    assert overlay and overlay[0].get("aria-hidden") == "true"
    for path in (SOURCES / "sqledit.ts", COMPILED / "sqledit.js"):
        code = _read(path)
        assert "lines.scrollTop = sql.scrollTop" in code and "lines.scrollLeft = sql.scrollLeft" in code, path.name
        assert 'h("span", { class: "ln" }' in code, path.name


# The strings' and numbers' new colors keep 4.5:1 on the editor's ground in both themes.
def test_sql_colors_keep_4_5_to_1():
    css = _read(CSS)
    assert "background: var(--panel)" in _rule(css, ".sql-box")
    for block in (":root", ':root[data-theme="light"]'):
        tokens = _tokens(_rule(css, block))
        for name in ("--sql-string", "--sql-number", "--muted", "--ink"):
            assert _contrast(tokens[name], tokens["--panel"]) >= 4.5, (block, name)


# Table and column names complete from the schema in the SQL editor only: at most 8, Up and Down move,
# Tab or Enter take one, Escape closes; the list is fixed to the screen so the editor never clips it.
def test_sql_editor_completes_names():
    css = _read(CSS)
    assert "position: fixed" in _rule(css, ".complete")
    assert re.search(r"MAX_CHOICES = 8\b", _read(SOURCES / "sqltext.ts"))
    for path in (SOURCES / "sqledit.ts", COMPILED / "sqledit.js"):
        code = _read(path)
        assert 'role: "listbox"' in code and 'role: "option"' in code and "aria-activedescendant" in code, path.name
        for key in ("ArrowDown", "ArrowUp", "Tab", "Enter", "Escape"):
            assert f'"{key}"' in code, (path.name, key)
        assert "schema?.datasets[state.dataset]" in code, path.name
        assert "question" not in code, path.name  # never on the Ask field
    for path in (SOURCES / "input.ts", COMPILED / "input.js"):
        assert re.search(r"event\.key === \"Enter\" && \(event\.metaKey \|\| event\.ctrlKey\)", _read(path)), path.name


# The editor's height is set by a drawn splitter: a 5px separator with a 1px hairline and a centered 24x3px grip
# in muted; on hover and focus the hairline turns 2px muted, and focus adds the blue ring. The arrow keys move it
# 16px, between 96px and 60% of the tab.
def test_editor_splitter():
    css = _read(CSS)
    split = _rule(css, ".splitter")
    assert "height: 5px" in split and "cursor: row-resize" in split
    assert "height: 1px" in _rule(css, ".splitter::before")
    grip = _rule(css, ".splitter::after")
    assert "width: 24px" in grip and "height: 3px" in grip and "background: var(--muted)" in grip and "left: 50%" in grip
    active = _rule(css, ".splitter:hover::before, .splitter.dragging::before, .splitter:focus-visible::before")
    assert "height: 2px" in active and "background: var(--muted)" in active
    assert "outline: 2px solid var(--accent)" in _rule(css, ".splitter:focus-visible")
    page = _tags(_read(INDEX))
    found = [attrs for tag, attrs in page.tags if attrs.get("id") == "splitter"]
    assert found and found[0].get("role") == "separator" and found[0].get("aria-orientation") == "horizontal"
    assert found[0].get("tabindex") == "0" and found[0].get("aria-controls") == "sql"
    for path in (SOURCES / "sqledit.ts", COMPILED / "sqledit.js"):
        code = _read(path)
        assert re.search(r"MIN_HEIGHT = 96\b", code) and re.search(r"STEP = 16\b", code) and "0.6" in code, path.name
        assert '"aria-valuenow"' in code and "setPointerCapture" in code, path.name


# The database select is drawn like the switch: no native look, the switch's border, and the tree's chevron as its arrow.
def test_database_select_matches_the_switch():
    css = _read(CSS)
    select = _rule(css, ".db-select")
    assert "appearance: none" in select and "-webkit-appearance: none" in select
    assert "border: 1px solid var(--line)" in select and "border-radius: 4px" in select
    assert "font-size: 14px" in select and "line-height: 1.45" in select
    assert "transform: rotate(90deg)" in _rule(css, ".select-wrap .chevron")
    for path in (SOURCES / "input.ts", COMPILED / "input.js"):
        assert re.search(r'h\("span", \{ class: "select-wrap" \}, database, chevron\(\)\)', _read(path)), path.name


# Each bar carries its value as text, and the value axis is labeled at 0 and at its top value, in muted tabular figures.
def test_chart_labels_values_and_axis():
    css = _read(CSS)
    for selector in (".bar-value", ".axis span"):
        body = _rule(css, selector)
        assert "color: var(--muted)" in body and "font-variant-numeric: tabular-nums" in body, selector
    for path in (SOURCES / "chart.ts", COMPILED / "chart.js"):
        code = _read(path)
        assert 'class: "bar-value"' in code and 'class: "axis-zero" }, "0")' in code and 'class: "axis-top"' in code, path.name
    assert "fill: var(--bars)" in _rule(css, ".chart .bar")  # Bar Slate


# "How the checks were tested" opens as a sheet under its link: fixed, right-aligned, at most 560px by 70vh with its
# own scroll, a hairline, 4px corners and no shadow; Close and Escape shut it, focus goes in and comes back.
def test_tested_sheet():
    css = _read(CSS)
    sheet = _rule(css, ".tested-panel")
    for part in ("position: fixed", "max-height: 70vh", "overflow-y: auto", "border: 1px solid var(--line)",
                 "border-radius: 4px", "background: var(--panel)"):
        assert part in sheet, part
    assert "560px" in sheet and "shadow" not in sheet
    assert "grid-template-columns: 96px minmax(0, 1fr)" in _rule(css, ".facts .entry")
    for path in (SOURCES / "topbar.ts", COMPILED / "topbar.js"):
        code = _read(path)
        assert '"Close"' in code and 'class: "fig"' in code, path.name
    for path in (SOURCES / "main.ts", COMPILED / "main.js"):
        code = _read(path)
        assert 'focusKey("tested-link")' in code, path.name
        # Escape shuts it only from inside the sheet or on its link, never from the editor or the filter
        escape = re.search(r'addEventListener\("keydown", \(event\) => \{(.*?)\n\}\);', code, re.DOTALL)
        assert escape, path.name
        body = _flat(escape.group(1))
        assert 'event.key === "Escape" && state.testedOpen && fromSheet()' in body, path.name
        sheet = re.search(r"function fromSheet\(\)[^{]*\{(.*?)\n\}", code, re.DOTALL)
        assert sheet and 'byId("tested").contains(' in sheet.group(1) and '"tested-link"' in sheet.group(1), path.name
    tested = [attrs for tag, attrs in _tags(_read(INDEX)).tags if attrs.get("id") == "tested"]
    assert tested and tested[0].get("tabindex") == "-1"  # a click inside the sheet keeps focus there
    assert "check_views" not in "".join(_read(p) for p in _sources())


# Opening a ninth table while the oldest is shown closes it without loading a preview for nothing.
def test_ninth_table_closes_the_oldest_quietly():
    for path in (SOURCES / "main.ts", COMPILED / "main.js"):
        code = _read(path)
        assert "closeTab(state.tabs[0].key, false)" in code, path.name
        body = re.search(r"function closeTab\(key[^)]*\)[^{]*\{(.*?)\n\}", code, re.DOTALL).group(1)
        assert "if (state.active !== key || !takeover) return;" in _flat(body), path.name


# Closing the shown tab with its mark or a middle-click puts focus on the tab shown next.
def test_closing_a_tab_keeps_focus_in_the_strip():
    for path in (SOURCES / "tabs.ts", COMPILED / "tabs.js"):
        code = _read(path)
        close = re.search(r"const close = \(key[^)]*\) => \{(.*?)\n\s*\};", code, re.DOTALL)
        assert close and "actions.closeTab(key)" in close.group(1) and "focusTab(root, state.active)" in close.group(1), path.name
        assert code.count("close(key)") >= 2, path.name  # the mark and the middle-click


# Back on the Query tab, the results, the Checks list and the narrow pane are where you left them.
def test_query_tab_keeps_its_scroll():
    for path in (SOURCES / "main.ts", COMPILED / "main.js"):
        code = _read(path)
        assert "keepQueryPlace()" in code and "restoreQueryPlace()" in code, path.name
        keep = re.search(r"function keepQueryPlace\(\)[^{]*\{(.*?)\n\}", code, re.DOTALL).group(1)
        for target in ('"result"', '"query-pane"', ".ins-body"):
            assert target in code, (path.name, target)
        assert "scrollTop" in keep, path.name


# Cmd+C copies a number column's value without its thousands separators; the grid still shows them.
def test_copy_takes_the_plain_number():
    for path in (SOURCES / "grid.ts", COMPILED / "grid.js"):
        code = _read(path)
        assert re.search(r"entry\.numeric\[[\w.]+\.col\] \? plainNumber\(", code), path.name
        assert re.search(r"function plainNumber\(text[^)]*\)[^{]*\{\s*return text\.replace\(/,/g, \"\"\);", code), path.name


# A schema load that fails with the server still there says so in the tree and backs off from 2 s to 30 s.
def test_schema_failure_says_so_and_backs_off():
    assert 'export const SCHEMA_RETRY = "Couldn\'t load the tables. Retrying.";' in _read(SOURCES / "copy.ts")
    for path in (SOURCES / "main.ts", COMPILED / "main.js"):
        code = _read(path)
        assert re.search(r"SCHEMA_MAX_MS = 30_?000", code), path.name
        assert re.search(r"Math\.min\(SCHEMA_MAX_MS, RETRY_MS \* 2 \*\* \(schemaTries - 1\)\)", code), path.name
    for path in (SOURCES / "sidebar.ts", COMPILED / "sidebar.js"):
        assert "state.schemaError" in _read(path), path.name


# History names the database when it isn't the Query tab's, says Edited SQL for an edit, and brings the shown run into view.
def test_history_keeps_the_database_and_edited_sql():
    for path in (SOURCES / "sidebar.ts", COMPILED / "sidebar.js"):
        code = _read(path)
        assert '"Edited SQL"' in code and "entry.dataset !== state.dataset" in code and "hist-note" in code, path.name
        assert "revealHistory(" in code, path.name
    assert "color: var(--ink)" in _rule(_read(CSS), ".hist.on .hist-kind, .hist.on .word, .hist.on .hist-note")


# A preview that failed because the server was gone is loaded again once the status says it's back.
def test_lost_preview_reloads_when_the_server_returns():
    assert "lost: boolean;" in _read(SOURCES / "state.ts")
    for path in (SOURCES / "main.ts", COMPILED / "main.js"):
        code = _read(path)
        assert "preview.lost = error instanceof ApiError && error.status === 0;" in code, path.name
        assert "if (wasGone && !state.statusError) forgetLostPreviews();" in _flat(code), path.name


# A model that's still loading (or any reason Ask is off) is said once, under the input; the example list
# under it doesn't say it again.
def test_ask_block_is_said_once():
    for path in (SOURCES / "result.ts", COMPILED / "result.js"):
        examples = re.search(r"function examples\(state[^)]*\)[^{]*\{(.*?)\n\}", _read(path), re.DOTALL)
        assert examples and "blocked" in examples.group(1), path.name  # the rows still turn off
        assert 'h("p", { class: "hint" }, blocked)' not in examples.group(1), path.name


NODE = shutil.which("node")

# A few tables and columns, shaped as the editor's names list holds them.
NAMES = [{"name": t, "note": "table", "table": None} for t in ("orders", "lineitem", "customer")] + [
    {"name": c, "note": f"{t} TYPE", "table": t} for t, cols in (
        ("orders", ("o_orderkey", "o_totalprice", "o_orderdate", "o_orderstatus")),
        ("lineitem", ("l_shipdate", "l_shipmode")),
        ("customer", ("c_custkey", "c_name"))) for c in cols]


# What the editor offers for each text, with the caret at its end: the names, or None when the list stays shut.
def _offers(texts: list[str]) -> list:
    if NODE is None:
        pytest.skip("Node isn't installed")
    module = (COMPILED / "sqltext.js").as_uri()
    script = (f"import {{ namesAt }} from {json.dumps(module)};\n"
              f"const names = {json.dumps(NAMES)};\n"
              f"const texts = {json.dumps(texts)};\n"
              "console.log(JSON.stringify(texts.map((t) => { const o = namesAt(t, t.length, names); "
              "return o && { from: o.from, names: o.choices.map((c) => c.name) }; })));")
    done = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


# The names list opens only for a prefix of a name: never for a word that merely sits inside one, a whole name,
# a keyword, one or two letters after a table name (an alias), or inside a string. The first two are the
# measured cases where Enter used to rewrite the SQL ("AS total" became "AS o_totalprice", "FROM orders o"
# became "FROM orders orders").
def test_names_open_only_for_a_prefix():
    shut = ["SELECT SUM(o_totalprice) AS total", "SELECT * FROM orders o", "SELECT * FROM orders AS o",
            "SELECT * FROM lineitem li", "SELECT * FROM orders", "SELECT * FROM orders ORDER", "SELECT price",
            "WHERE o_orderstatus = 'orde", "SELECT 1e", "SELECT * FROM orders -- l_sh"]
    assert _offers(shut) == [None] * len(shut)
    assert _offers(["SELECT l_ship", "SELECT O_TOT", "SELECT orders.o_ord", "SELECT * FROM ord"]) == [
        {"from": 7, "names": ["l_shipdate", "l_shipmode"]},
        {"from": 7, "names": ["o_totalprice"]},
        {"from": 14, "names": ["o_orderkey", "o_orderdate", "o_orderstatus"]},
        {"from": 14, "names": ["orders"]},
    ]


# Enter keeps its meaning (a new line) while the list is open, unless Up or Down has moved the highlight since it
# opened; Tab or a click takes a name. Only typing opens the list, never a paste.
def test_enter_takes_a_name_only_after_the_arrows():
    for path in (SOURCES / "sqledit.ts", COMPILED / "sqledit.js"):
        code = _flat(_read(path))
        assert 'event.key === "Tab" || (event.key === "Enter" && current.moved)' in code, path.name
        assert "current.moved = true;" in code, path.name
        assert 'event.inputType === "insertText"' in code, path.name
        assert "namesAt(" in code, path.name
