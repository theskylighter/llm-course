import { useCallback, useEffect, useState } from "react";
import { Link, Outlet, useLocation } from "react-router-dom";
import { Eye, EyeOff, GraduationCap, Menu, Moon, Search, Sun } from "lucide-react";
import { useDarkMode, useMediaQuery, useStoredBoolean } from "../hooks";
import { course, findLesson, HOME_SLUG } from "../course";
import { site, storageKey } from "../site";
import { loadSearch } from "../search";
import { ReadingProgress } from "./ReadingProgress";
import { SearchDialog } from "./SearchDialog";
import { Sidebar } from "./Sidebar";
import { TocPanel } from "./TocPanel";

/** Context passed to lesson pages through <Outlet />. */
export interface LayoutContext {
  showAll: boolean;
}

/**
 * The persistent shell: top bar, course sidebar (left) and "On this page" (right). These live
 * outside <Outlet />, so they stay mounted while the reader moves between lessons. That keeps
 * the course list's scroll position and open chapters intact.
 */
export function Layout() {
  const [dark, toggleDark] = useDarkMode();
  const [showAll, setShowAll] = useState(false);
  const [leftCollapsed, toggleLeft] = useStoredBoolean(storageKey("left-collapsed"));
  const [rightCollapsed, toggleRight] = useStoredBoolean(storageKey("right-collapsed"));
  const { pathname } = useLocation();
  const [searchOpen, setSearchOpen] = useState(false);
  const closeSearch = useCallback(() => setSearchOpen(false), []);

  // On phones the course list is an off-canvas drawer instead of a docked panel.
  const isMobile = useMediaQuery("(max-width: 900px)");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const closeDrawer = useCallback(() => setDrawerOpen(false), []);

  const slug = pathname === "/" ? HOME_SLUG : pathname.replace(/^\//, "");
  const current = findLesson(slug);
  const toc = current ? current.toc : slug === HOME_SLUG ? course.intro.toc : [];

  const toggleAnswers = useCallback(() => setShowAll((s) => !s), []);

  // Close the drawer after navigating, on Escape, and when the window grows past phone width.
  useEffect(() => closeDrawer(), [pathname, closeDrawer]);
  useEffect(() => {
    if (!isMobile) closeDrawer();
  }, [isMobile, closeDrawer]);
  useEffect(() => {
    if (!drawerOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && closeDrawer();
    window.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden"; // keep the page from scrolling behind the drawer
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [drawerOpen, closeDrawer]);

  // Ctrl+B / Cmd+B toggles the course sidebar (the drawer on phones).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "b") {
        e.preventDefault();
        if (isMobile) setDrawerOpen((o) => !o);
        else toggleLeft();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggleLeft, isMobile]);

  // Ctrl/Cmd+K opens search from anywhere; "/" does too unless the reader is typing.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      const typing = !!el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName) || !!el.closest(".excalidraw"));
      const cmdK = (e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "k";
      const slash = e.key === "/" && !e.ctrlKey && !e.metaKey && !e.altKey && !typing;
      if (cmdK || slash) {
        e.preventDefault();
        setSearchOpen(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  // Navigating (e.g. choosing a result) always closes the window.
  useEffect(() => closeSearch(), [pathname, closeSearch]);

  return (
    <>
      <header className="topbar">
        <div className="topbar-start">
          <button
            type="button"
            className="icon-btn menu-btn"
            onClick={() => setDrawerOpen((o) => !o)}
            aria-label={drawerOpen ? "Close course menu" : "Open course menu"}
            aria-expanded={drawerOpen}
            aria-controls="course-sidebar"
          >
            <Menu size={22} aria-hidden="true" />
          </button>
          <Link className="brand" to="/">
            <GraduationCap className="icon" size={20} aria-hidden="true" />
            {site.brand}
          </Link>
          {current && (
            <span className="topbar-sep" aria-hidden="true">
              /
            </span>
          )}
          <span className="ptitle">{current ? current.title : ""}</span>
        </div>
        <button
          type="button"
          className="btn-ghost search-btn topbar-search"
          onClick={() => setSearchOpen(true)}
          onPointerEnter={() => void loadSearch()}
          onFocus={() => void loadSearch()}
          aria-label="Search"
          aria-keyshortcuts="Control+K Meta+K"
        >
          <Search className="icon" size={16} aria-hidden="true" />
          <span className="label">Search</span>
          <kbd className="search-kbd">{/Mac|iPhone|iPad/.test(navigator.platform) ? "⌘K" : "Ctrl K"}</kbd>
        </button>
        <div className="topbar-actions">
          <button type="button" className="btn-ghost" onClick={toggleAnswers} aria-pressed={showAll}>
            {showAll ? (
              <EyeOff className="icon" size={16} aria-hidden="true" />
            ) : (
              <Eye className="icon" size={16} aria-hidden="true" />
            )}
            <span className="label">{showAll ? "Hide answers" : "Show answers"}</span>
          </button>
          <button
            type="button"
            className="icon-btn"
            onClick={toggleDark}
            aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
            title={dark ? "Light theme" : "Dark theme"}
          >
            {dark ? <Sun className="icon" size={18} aria-hidden="true" /> : <Moon className="icon" size={18} aria-hidden="true" />}
          </button>
          <a
            className="icon-btn github-link"
            href="https://github.com/theskylighter/llm-course"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="GitHub repository"
            title="GitHub repository"
          >
            <svg className="icon github-icon" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" />
            </svg>
          </a>
        </div>
        <ReadingProgress />
      </header>
      <div className="layout">
        <div className={`drawer-backdrop${drawerOpen ? " open" : ""}`} onClick={closeDrawer} aria-hidden="true" />
        <Sidebar
          currentSlug={slug}
          collapsed={leftCollapsed}
          onToggle={isMobile ? closeDrawer : toggleLeft}
          drawerOpen={drawerOpen}
        />
        <main>
          <div className="content">
            <Outlet context={{ showAll } satisfies LayoutContext} />
          </div>
        </main>
        <TocPanel toc={toc} pageKey={slug} collapsed={rightCollapsed} onToggle={toggleRight} />
      </div>
      <SearchDialog open={searchOpen} onClose={closeSearch} />
    </>
  );
}
