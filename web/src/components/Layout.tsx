import { useCallback, useEffect, useState } from "react";
import { Link, Outlet, useLocation } from "react-router-dom";
import { Eye, EyeOff, GraduationCap, Menu, Moon, Sun } from "lucide-react";
import { useDarkMode, useMediaQuery, useStoredBoolean } from "../hooks";
import { course, findLesson, HOME_SLUG } from "../course";
import { site, storageKey } from "../site";
import { ReadingProgress } from "./ReadingProgress";
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

  return (
    <>
      <header className="topbar">
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
    </>
  );
}
