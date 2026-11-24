import { Fragment, useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { BookOpen, CornerDownLeft, History, Search, X } from "lucide-react";
import { storageKey } from "../site";
import { highlightTerms, runSearch, snippet, termsRegex, type HitGroup, type SearchHit } from "../search";

const RECENT_KEY = storageKey("recent-searches");
const MAX_RECENT = 5;

function readRecent(): string[] {
  try {
    const v = JSON.parse(localStorage.getItem(RECENT_KEY) ?? "[]");
    return Array.isArray(v) ? v.filter((x) => typeof x === "string").slice(0, MAX_RECENT) : [];
  } catch {
    return [];
  }
}

/** Wrap every match of `re` in <mark>. */
function highlight(text: string, re: RegExp | null): ReactNode {
  if (!re) return text;
  return text.split(re).map((part, i) => (i % 2 === 1 ? <mark key={i}>{part}</mark> : <Fragment key={i}>{part}</Fragment>));
}

interface Props {
  open: boolean;
  onClose: () => void;
}

/**
 * Ctrl/Cmd+K search window. The index lives in the browser (see search.ts), so there is no
 * server or third-party service: results are grouped by lesson, then section, with the matched
 * words highlighted in a short excerpt.
 */
export function SearchDialog({ open, onClose }: Props) {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [groups, setGroups] = useState<HitGroup[]>([]);
  const [searched, setSearched] = useState("");   // the query `groups` answers
  const [loading, setLoading] = useState(false);
  const [active, setActive] = useState(0);
  const [recent, setRecent] = useState<string[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const seq = useRef(0);

  const flat = useMemo(() => groups.flatMap((g) => g.hits), [groups]);

  // On open: remember where focus was, lock page scroll, focus the box. On close: give focus back.
  useEffect(() => {
    if (!open) return;
    const previous = document.activeElement as HTMLElement | null;
    setRecent(readRecent());
    document.body.style.overflow = "hidden";
    inputRef.current?.focus();
    return () => {
      document.body.style.overflow = "";
      previous?.focus?.();
    };
  }, [open]);

  // Search as you type. A counter drops answers that arrive after a newer keystroke.
  useEffect(() => {
    if (!open) return;
    const q = query.trim();
    if (!q) {
      setGroups([]);
      setSearched("");
      setLoading(false);
      return;
    }
    const mine = ++seq.current;
    setLoading(true);
    runSearch(q).then((g) => {
      if (mine !== seq.current) return;
      setGroups(g);
      setSearched(q);
      setActive(0);
      setLoading(false);
    });
  }, [query, open]);

  // Keep the highlighted result in view while arrowing through a long list.
  useEffect(() => {
    listRef.current?.querySelector<HTMLElement>(`[data-i="${active}"]`)?.scrollIntoView({ block: "nearest" });
  }, [active, groups]);

  if (!open) return null;

  function choose(hit: SearchHit) {
    const q = query.trim();
    if (q) {
      const next = [q, ...readRecent().filter((r) => r !== q)].slice(0, MAX_RECENT);
      localStorage.setItem(RECENT_KEY, JSON.stringify(next));
    }
    onClose();
    navigate(`/${hit.slug}${hit.anchor ? `#${hit.anchor}` : ""}`);
  }

  function onKeyDown(e: KeyboardEvent) {
    if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (!flat.length) return;
      setActive((a) => (e.key === "ArrowDown" ? (a + 1) % flat.length : (a - 1 + flat.length) % flat.length));
    } else if (e.key === "Enter" && flat[active]) {
      e.preventDefault();
      choose(flat[active]);
    }
  }

  const trimmed = query.trim();
  const settled = trimmed !== "" && searched === trimmed && !loading;
  let flatIndex = -1;

  return (
    <div className="search-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="search-dialog" role="dialog" aria-modal="true" aria-label="Search the course" onKeyDown={onKeyDown}>
        <div className="search-input-row">
          <Search className="icon" size={18} aria-hidden="true" />
          <input
            ref={inputRef}
            className="search-input"
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search lessons, concepts, papers…"
            role="combobox"
            aria-expanded={flat.length > 0}
            aria-controls="search-results"
            aria-activedescendant={flat.length ? `search-hit-${active}` : undefined}
            autoComplete="off"
            autoCorrect="off"
            spellCheck={false}
            enterKeyHint="go"
          />
          <button type="button" className="search-close" onClick={onClose} aria-label="Close search">
            <X size={18} aria-hidden="true" />
          </button>
        </div>

        <div className="search-body" id="search-results" role="listbox" ref={listRef}>
          {!trimmed && (
            <div className="search-empty">
              {recent.length > 0 ? (
                <>
                  <div className="search-group-title">Recent</div>
                  {recent.map((r) => (
                    <button type="button" key={r} className="search-recent" onClick={() => { setQuery(r); inputRef.current?.focus(); }}>
                      <History className="icon" size={15} aria-hidden="true" />
                      {r}
                    </button>
                  ))}
                </>
              ) : (
                <p>Type to search every lesson, including the deep-dive walkthroughs.</p>
              )}
            </div>
          )}

          {trimmed && loading && !groups.length && <div className="search-empty"><p>Searching…</p></div>}
          {settled && !flat.length && (
            <div className="search-empty">
              <p>No results for <strong>“{trimmed}”</strong>.</p>
              <p className="search-tip">Try a shorter or different word.</p>
            </div>
          )}

          {groups.map((g) => (
            <div key={g.slug} className="search-group" role="presentation">
              <div className="search-group-title" role="presentation">
                <span className="search-lesson">{g.number ? `${g.number} · ` : ""}{g.title}</span>
                <span className="search-chapter">{g.chapter}</span>
              </div>
              {g.hits.map((hit) => {
                flatIndex += 1;
                const i = flatIndex;
                const re = termsRegex(highlightTerms(searched, hit));
                const isLead = !hit.anchor;
                return (
                  <div
                    key={hit.id}
                    id={`search-hit-${i}`}
                    data-i={i}
                    role="option"
                    aria-selected={i === active}
                    className={`search-hit${i === active ? " active" : ""}`}
                    onMouseMove={() => i !== active && setActive(i)}
                    onClick={() => choose(hit)}
                  >
                    <div className="search-hit-head">
                      {hit.kind === "deep" && <BookOpen className="icon" size={14} aria-hidden="true" />}
                      <span className="search-hit-title">{isLead ? "Lesson overview" : highlight(hit.heading, re)}</span>
                      {i === active && <CornerDownLeft className="icon search-enter" size={14} aria-hidden="true" />}
                    </div>
                    <div className="search-snippet">{highlight(snippet(hit.text, re), re)}</div>
                  </div>
                );
              })}
            </div>
          ))}
        </div>

        <div className="search-footer" aria-hidden="true">
          <span><kbd>↑</kbd><kbd>↓</kbd> navigate</span>
          <span><kbd>↵</kbd> open</span>
          <span><kbd>esc</kbd> close</span>
        </div>
      </div>
    </div>
  );
}
