import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ChevronRight, ListChecks, PanelLeftClose, PanelLeftOpen } from "lucide-react";
import { course, lessonPath } from "../course";
import { storageKey } from "../site";

const SCROLL_KEY = storageKey("sidebar-scroll");
const OPEN_KEY = storageKey("sidebar-open-chapters");

interface SidebarProps {
  currentSlug: string;
  /** Collapsed to a slim rail. The list stays mounted, so it keeps its scroll position. */
  collapsed: boolean;
  onToggle: () => void;
  /** Phones only: the sidebar is an off-canvas drawer, shown when true. */
  drawerOpen?: boolean;
}

/** The chapter that contains a lesson, or null for the home page. */
function chapterOf(slug: string): string | null {
  const chapter = course.chapters.find((c) => c.lessons.includes(slug));
  return chapter ? chapter.title : null;
}

/** Chapters that were open earlier in this browser session. */
function readOpenChapters(): string[] {
  try {
    const parsed: unknown = JSON.parse(sessionStorage.getItem(OPEN_KEY) ?? "[]");
    return Array.isArray(parsed) ? parsed.filter((x): x is string => typeof x === "string") : [];
  } catch {
    return [];
  }
}

/** Left sidebar: the course list. Its scroll position is kept across navigation and reloads. */
export function Sidebar({ currentSlug, collapsed, onToggle, drawerOpen = false }: SidebarProps) {
  const bodyRef = useRef<HTMLDivElement>(null);
  const activeChapter = chapterOf(currentSlug);

  // Chapters the reader has opened. The current lesson's chapter is open from the first render,
  // so the saved scroll position can be restored against the final list height.
  // They are also remembered for the browser session: restoring the scroll position after a reload
  // only works if the list is as tall as it was, i.e. the same chapters are open.
  const [openChapters, setOpenChapters] = useState<Set<string>>(() => {
    const open = new Set<string>(readOpenChapters());
    if (activeChapter) open.add(activeChapter);
    return open;
  });

  useEffect(() => {
    sessionStorage.setItem(OPEN_KEY, JSON.stringify([...openChapters]));
  }, [openChapters]);

  useEffect(() => {
    if (!activeChapter) return;
    setOpenChapters((prev) => (prev.has(activeChapter) ? prev : new Set(prev).add(activeChapter)));
  }, [activeChapter]);

  // Restore the saved scroll position on first mount (for example after a full page reload).
  useLayoutEffect(() => {
    const saved = Number(sessionStorage.getItem(SCROLL_KEY));
    if (bodyRef.current && Number.isFinite(saved)) bodyRef.current.scrollTop = saved;
  }, []);

  // Remember the scroll position as the reader scrolls.
  useEffect(() => {
    const body = bodyRef.current;
    if (!body) return;
    const onScroll = () => sessionStorage.setItem(SCROLL_KEY, String(body.scrollTop));
    body.addEventListener("scroll", onScroll, { passive: true });
    return () => body.removeEventListener("scroll", onScroll);
  }, []);

  function toggleChapter(title: string) {
    setOpenChapters((prev) => {
      const next = new Set(prev);
      if (next.has(title)) next.delete(title);
      else next.add(title);
      return next;
    });
  }

  return (
    <aside
      id="course-sidebar"
      className={`panel sidebar${collapsed ? " collapsed" : ""}${drawerOpen ? " drawer-open" : ""}`}
      aria-label="Course navigation"
    >
      <button
        type="button"
        className="icon-btn rail-btn"
        onClick={onToggle}
        aria-label="Expand course sidebar"
        title="Expand sidebar (Ctrl+B)"
      >
        <PanelLeftOpen size={18} aria-hidden="true" />
      </button>
      <div className="panel-head">
        <span>Course</span>
        <button
          type="button"
          className="icon-btn"
          onClick={onToggle}
          aria-label="Collapse course sidebar"
          title="Collapse sidebar (Ctrl+B)"
        >
          <PanelLeftClose size={18} aria-hidden="true" />
        </button>
      </div>
      <div className="panel-body course-nav" ref={bodyRef}>
        {course.chapters.map((chapter) => {
          const open = openChapters.has(chapter.title);
          return (
            <section key={chapter.title} className="chapter-nav">
              <button
                type="button"
                className="chapter-toggle"
                aria-expanded={open}
                onClick={() => toggleChapter(chapter.title)}
              >
                <ChevronRight className="chev" size={14} aria-hidden="true" />
                {chapter.title}
              </button>
              {open && (
                <ul className="pages">
                  {chapter.lessons.map((slug) => {
                    const lesson = course.lessons.find((l) => l.slug === slug);
                    if (!lesson) return null;
                    const isCurrent = slug === currentSlug;
                    return (
                      <li key={slug} className={isCurrent ? "cur" : undefined}>
                        <Link to={lessonPath(slug)} aria-current={isCurrent ? "page" : undefined}>
                          <span className="ln">
                            {lesson.number ||
                              (lesson.syllabus === "Practice" ? (
                                <ListChecks className="icon" size={14} aria-label="Practice set" />
                              ) : null)}
                          </span>
                          <span>{lesson.title}</span>
                        </Link>
                      </li>
                    );
                  })}
                </ul>
              )}
            </section>
          );
        })}
      </div>
    </aside>
  );
}
