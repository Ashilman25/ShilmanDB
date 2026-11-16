"""Tests for the page: app/static/ and the TypeScript it's built from in app/web/.

They only read files; the build itself (tsc) runs in CI, never here."""
from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

from app import copy_rules, jobs, views
from app.tests.test_app_views import REGION_SQL, _region_reply

APP_DIR = Path(__file__).resolve().parents[1]
STATIC = APP_DIR / "static"
WEB = APP_DIR / "web"
INDEX, CSS = STATIC / "index.html", STATIC / "app.css"
SOURCES, COMPILED = WEB / "src", STATIC / "js"

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


# The top bar stays put; the rail and the main area scroll on their own, and the main area always
# keeps its scrollbar's room, so the layout never shifts when an answer makes it scroll.
def test_page_is_an_app_shell():
    css = _read(CSS)
    assert "overflow: hidden" in _rule(css, "body")
    assert "grid-template-rows: minmax(0, 1fr)" in _rule(css, ".frame")
    assert "overflow-y: scroll" in _rule(css, ".main")
    assert "overflow-y: auto" in _rule(css, ".rail")


# Rows stay one line tall, and a long table scrolls in its own box under a header that stays.
def test_tables_scroll_in_place():
    css = _read(CSS)
    scroller = _rule(css, ".scroller")
    assert "overflow: auto" in scroller and "max-height" in scroller
    assert "white-space: nowrap" in _rule(css, "th, td")
    assert "position: sticky" in _rule(css, "th")
    assert "text-overflow: ellipsis" in _rule(css, "td.text .cell")


# The panel holds its layout on a question: the verdict line keeps two lines in every stage, and the
# second answer's row keeps four lines beside the result (three when the panel sits under it, full width).
def test_question_panel_holds_its_layout():
    css = _read(CSS)
    assert "min-height: calc(2 * 1.45em)" in _rule(css, ".verdict.reserve")
    assert "min-height: calc(4 * 1.45em)" in _rule(css, ".entry.second")
    assert "min-height: calc(3 * 1.45em)" in _rule(_media(css, "(max-width: 1100px)"), ".entry.second")


# Section headings are 15 px semibold, a clear step above the 14 px verdict line and the 12 px status words.
def test_section_headings_are_15px():
    css = _read(CSS)
    for selector in (".rail h2", ".evidence h2", ".pair h2", ".tested-inner h2"):
        body = _rule(css, selector)
        assert "font-size: 15px" in body and "font-weight: 600" in body, selector
    assert "font-size: 12px" in _rule(css, ".word")


# Finish details: the shown History item has an accent hairline, an empty evidence column steps aside
# and leaves the result a readable measure, and scrollbar thumbs draw no paper-coloured ring on panels.
def test_finish_details():
    css = _read(CSS)
    assert "border-color: var(--accent)" in _rule(css, ".item.on")
    assert "grid-template-columns: minmax(0, 1fr)" in _rule(css, ".cols.alone")
    assert "display: none" in _rule(css, ".cols.alone > .evidence")
    assert "max-width: 66ch" in _rule(css, ".cols.alone > .result")
    # the columns' layout is set from the script, never by a :has() rule
    plain = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", plain):
        if ":has(" in selector:
            assert "cols" not in selector and "grid-template" not in body, selector.strip()
    thumb = _rule(css, "::-webkit-scrollbar-thumb")
    assert "solid transparent" in thumb and "background-clip: padding-box" in thumb


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
    assert "border: 1px solid var(--field)" in _rule(css, ".field-line, .field-sql")
    assert "border-color: var(--muted)" in _rule(css, ".field-line:hover, .field-sql:hover")
    assert css.count("var(--field)") == 1


# Under 760 px the stacked rail is capped and scrolls on its own, so the input row stays near the top.
def test_stacked_rail_never_buries_the_input():
    rail = _rule(_media(_read(CSS), "(max-width: 760px)"), ".rail")
    assert "max-height: 35vh" in rail and "overflow-y: auto" in rail
    assert "border-bottom: 1px solid var(--line)" in rail


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


# In mlx mode the top bar shows the prompt cache: on, or off with its reason as visible text.
def test_top_bar_draws_the_cache_state():
    for path in (SOURCES / "topbar.ts", COMPILED / "topbar.js"):
        code = _read(path)
        assert re.search(r'mode [!=]== "mlx"', code) and "status.cache" in code, path.name
        assert '"Prompt cache on"' in code and "`Prompt cache off: ${" in code, path.name


# A dataset whose files are missing or failed to load is named as data; Engine is the bridge binary's.
def test_top_bar_names_data_problems():
    for path in (SOURCES / "topbar.ts", COMPILED / "topbar.js"):
        code = _read(path)
        assert '"Data missing"' in code and '"Data failed"' in code and '"Engine missing"' in code, path.name
        assert '"Missing:"' in code, path.name  # the server's prefix for missing dataset files


# Started with --no-model, the empty result points to the SQL runner, not to asking.
def test_empty_hint_follows_the_model():
    copy = _read(SOURCES / "copy.ts")
    sql_hint = re.search(r'export const EMPTY_RESULT_SQL = "([^"]+)"', copy)
    assert sql_hint and "plain English" not in sql_hint.group(1) and "SQL" in sql_hint.group(1)
    for path in (SOURCES / "result.ts", COMPILED / "result.js"):
        assert re.search(r'model\.state === "off" \? EMPTY_RESULT_SQL : EMPTY_RESULT', _read(path)), path.name


# A call that never reaches the server rechecks the status, so the top bar says Not connected.
def test_lost_server_rechecks_the_status():
    for path in (SOURCES / "main.ts", COMPILED / "main.js"):
        code = _read(path)
        found = re.search(r"function noticeGone\(error[^)]*\)[^{]*\{(.*?)\n\}", code, re.DOTALL)
        assert found and "status === 0" in found.group(1) and "loadStatus()" in found.group(1), path.name
        assert code.count("noticeGone(error)") >= 5, path.name  # follow, start, stop, show more, history


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
