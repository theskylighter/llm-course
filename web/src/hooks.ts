import { useCallback, useEffect, useState } from "react";
import { storageKey } from "./site";

const THEME_KEY = storageKey("theme");

/** Dark mode, persisted in localStorage and applied to <html class="dark">. */
export function useDarkMode(): [boolean, () => void] {
  const [dark, setDark] = useState<boolean>(() => localStorage.getItem(THEME_KEY) === "dark");

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem(THEME_KEY, dark ? "dark" : "light");
  }, [dark]);

  const toggle = useCallback(() => setDark((d) => !d), []);
  return [dark, toggle];
}

/** A boolean kept in localStorage, e.g. whether a sidebar is collapsed. */
export function useStoredBoolean(key: string, initial = false): [boolean, () => void] {
  const [value, setValue] = useState<boolean>(() => {
    const stored = localStorage.getItem(key);
    return stored === null ? initial : stored === "1";
  });

  useEffect(() => {
    localStorage.setItem(key, value ? "1" : "0");
  }, [key, value]);

  const toggle = useCallback(() => setValue((v) => !v), []);
  return [value, toggle];
}

/** Whether a CSS media query currently matches (re-evaluated when the window changes). */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState<boolean>(() => window.matchMedia(query).matches);

  useEffect(() => {
    const mql = window.matchMedia(query);
    const onChange = () => setMatches(mql.matches);
    onChange();
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, [query]);

  return matches;
}
