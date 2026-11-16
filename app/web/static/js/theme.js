const KEY = "shilmandb-theme";
export function savedTheme() {
    try {
        return localStorage.getItem(KEY) === "light" ? "light" : "dark";
    }
    catch {
        return "dark"; // storage blocked: dark, as decided
    }
}
export function saveTheme(theme) {
    try {
        localStorage.setItem(KEY, theme);
    }
    catch {
        // storage blocked: the switch still works until the page reloads
    }
}
export function applyTheme(theme) {
    if (theme === "light")
        document.documentElement.setAttribute("data-theme", "light");
    else
        document.documentElement.removeAttribute("data-theme");
}
