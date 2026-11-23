import { useEffect, useState } from "react";
import { PanelRightClose, PanelRightOpen } from "lucide-react";
import type { TocEntry } from "../types";

interface TocPanelProps {
  /** Headings of the current page. */
  toc: TocEntry[];
  /** Changes on every page, so the scroll-spy re-attaches to the new headings. */
  pageKey: string;
  /** Collapsed to a slim rail. Stays mounted, so the scroll-spy keeps tracking the page. */
  collapsed: boolean;
  onToggle: () => void;
}

/** Right-hand "On this page" navigation with a scroll-spy that highlights the section in view. */
export function TocPanel({ toc, pageKey, collapsed, onToggle }: TocPanelProps) {
  const [activeId, setActiveId] = useState<string | null>(toc[0]?.id ?? null);

  useEffect(() => {
    setActiveId(toc[0]?.id ?? null);
    if (toc.length === 0) return;

    const headings = toc
      .map((entry) => document.getElementById(entry.id))
      .filter((el): el is HTMLElement => el !== null);

    // A heading becomes "current" once it crosses the upper part of the viewport.
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible.length > 0) setActiveId(visible[0].target.id);
      },
      { rootMargin: "-72px 0px -65% 0px", threshold: 0 },
    );
    headings.forEach((h) => observer.observe(h));
    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pageKey]);

  if (toc.length === 0) return null;

  return (
    <aside className={`panel toc-aside${collapsed ? " collapsed" : ""}`} aria-label="On this page">
      <button
        type="button"
        className="icon-btn rail-btn"
        onClick={onToggle}
        aria-label="Expand 'On this page'"
        title="Show 'On this page'"
      >
        <PanelRightOpen size={18} aria-hidden="true" />
      </button>
      <div className="panel-head">
        <span>On this page</span>
        <button
          type="button"
          className="icon-btn"
          onClick={onToggle}
          aria-label="Collapse 'On this page'"
          title="Hide 'On this page'"
        >
          <PanelRightClose size={18} aria-hidden="true" />
        </button>
      </div>
      <div className="panel-body">
        <ul className="toc">
          {toc.map((entry) => (
            <li key={entry.id} className={`l${entry.level}${entry.id === activeId ? " active" : ""}`}>
              <a
                href={`#${entry.id}`}
                aria-current={entry.id === activeId ? "location" : undefined}
                dangerouslySetInnerHTML={{ __html: entry.html }}
              />
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}
