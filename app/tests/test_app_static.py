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
    assert "max-width: 75ch" in _rule(css, ".cols.alone > .result")
    assert ":has(" not in css
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
