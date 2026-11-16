// Dark or light. Dark is the default; the choice is kept in this browser when it allows storage.
export type Theme = "dark" | "light";

const KEY = "shilmandb-theme";

export function savedTheme(): Theme {
  try {
    return localStorage.getItem(KEY) === "light" ? "light" : "dark";
  } catch {
    return "dark"; // storage blocked: dark, as decided
  }
}

export function saveTheme(theme: Theme): void {
  try {
    localStorage.setItem(KEY, theme);
  } catch {
    // storage blocked: the switch still works until the page reloads
  }
}

export function applyTheme(theme: Theme): void {
  if (theme === "light") document.documentElement.setAttribute("data-theme", "light");
  else document.documentElement.removeAttribute("data-theme");
}
