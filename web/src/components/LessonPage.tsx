import { MouseEvent, useEffect, useRef } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
import { Link, Navigate, useLocation, useNavigate, useOutletContext, useParams } from "react-router-dom";
import { ArrowLeft, ArrowRight, BookMarked, Clock } from "lucide-react";
import { findLesson, HOME_SLUG, lessonOrder, lessonPath } from "../course";
import type { LayoutContext } from "./Layout";
import { NotFound } from "./NotFound";
import { ExcalidrawTimeline } from "./ExcalidrawTimeline";

/** Matches links to lesson pages inside lesson HTML, e.g. "08_1_association.html#sec". */
const PAGE_LINK = /^([\w-]+)\.html(#.*)?$/;

export function LessonPage() {
  const { slug: rawSlug = "" } = useParams();
  const navigate = useNavigate();
  const { pathname, hash, key } = useLocation();
  const { showAll } = useOutletContext<LayoutContext>();
  const bodyRef = useRef<HTMLDivElement>(null);

  // Old links look like /03_4_encapsulation.html; send them to the clean route.
  const slug = rawSlug.endsWith(".html") ? rawSlug.slice(0, -".html".length) : rawSlug;
  const lesson = findLesson(slug);
  const redirectTo = slug === HOME_SLUG ? "/" : lessonPath(slug);

  // Scroll the content back to the top on navigation, unless the URL points at an anchor.
  useEffect(() => {
    if (!hash) window.scrollTo(0, 0);
  }, [pathname, hash]);

  // A link to #something scrolls there; a deep dive (#deep-...) is opened first. `key` changes on
  // every navigation, so choosing the same search result twice still scrolls back to it.
  useEffect(() => {
    const target = hash ? document.getElementById(decodeURIComponent(hash.slice(1))) : null;
    if (!target) return;
    if (target instanceof HTMLDetailsElement) target.open = true;
    target.scrollIntoView({ block: "start" });
  }, [hash, slug, key]);

  // Open or close every answer block when the topbar toggle changes, and on each new lesson.
  useEffect(() => {
    bodyRef.current?.querySelectorAll<HTMLDetailsElement>("details.ans").forEach((d) => {
      d.open = showAll;
    });
  }, [showAll, slug]);

  // Render TeX spans (written as $...$ and $$...$$ in the lessons) with KaTeX.
  useEffect(() => {
    bodyRef.current?.querySelectorAll<HTMLElement>(".math[data-tex]").forEach((el) => {
      if (el.dataset.done) return;
      try {
        katex.render(el.dataset.tex ?? "", el, {
          displayMode: el.classList.contains("display"),
          throwOnError: true,
        });
        el.dataset.done = "1";
      } catch (e) {
        el.classList.add("math-error");
        el.dataset.done = "err";
      }
    });
  }, [slug]);

  // Add a copy button to every code block (not output or ASCII-diagram blocks).
  useEffect(() => {
    const blocks = bodyRef.current?.querySelectorAll<HTMLDivElement>(".codewrap:not(.output):not(.text):not(.uml)") ?? [];
    blocks.forEach((block) => {
      if (block.querySelector(".copybtn")) return;
      const button = document.createElement("button");
      button.className = "copybtn";
      button.type = "button";
      button.textContent = "copy";
      button.addEventListener("click", () => {
        const code = block.querySelector("pre")?.innerText ?? "";
        void navigator.clipboard.writeText(code);
        button.textContent = "copied!";
        window.setTimeout(() => (button.textContent = "copy"), 1200);
      });
      block.appendChild(button);
    });
  }, [slug]);

  if (rawSlug !== slug) return <Navigate to={redirectTo} replace />;
  if (slug === HOME_SLUG) return <Navigate to="/" replace />;
  if (!lesson) return <NotFound />;

  // Links between lessons are rendered as plain <a href="x.html">, so intercept them and
  // navigate inside the app. The sidebar stays mounted and keeps its scroll position.
  function onBodyClick(event: MouseEvent<HTMLDivElement>) {
    const anchor = (event.target as HTMLElement).closest("a");
    const href = anchor?.getAttribute("href");
    if (!href) return;
    const match = PAGE_LINK.exec(href);
    if (!match) return;
    event.preventDefault();
    const target = match[1] === HOME_SLUG ? "/" : `/${match[1]}`;
    navigate(target + (match[2] ?? ""));
  }

  const index = lessonOrder.indexOf(slug);
  const prev = index > 0 ? findLesson(lessonOrder[index - 1]) : undefined;
  const next = index < lessonOrder.length - 1 ? findLesson(lessonOrder[index + 1]) : undefined;

  // Check if this is the Research Papers Journey lesson to render interactive timeline
  const isResearchPapersJourney = slug === "11_6_research_papers_journey";

  return (
    <article className="lesson">
      <div className="crumb">
        {lesson.chapter}
        {lesson.number ? ` › Lesson ${lesson.number}` : ""}
      </div>
      <h1>{lesson.title}</h1>
      <div className="meta">
        <span>
          <Clock className="icon" size={15} aria-hidden="true" /> {lesson.minutes} min read
        </span>
        <span>
          <BookMarked className="icon" size={15} aria-hidden="true" /> Syllabus: {lesson.syllabus}
        </span>
      </div>
      {isResearchPapersJourney && <ExcalidrawTimeline />}
      <div ref={bodyRef} onClick={onBodyClick} dangerouslySetInnerHTML={{ __html: lesson.html }} />
      <nav className="pager" aria-label="Lesson navigation">
        {prev ? (
          <Link className="nav prev" to={lessonPath(prev.slug)}>
            <ArrowLeft className="icon" size={16} aria-hidden="true" />
            {prev.title}
          </Link>
        ) : (
          <Link className="nav prev" to="/">
            <ArrowLeft className="icon" size={16} aria-hidden="true" />
            Course home
          </Link>
        )}
        {next && (
          <Link className="nav next" to={lessonPath(next.slug)}>
            {next.title}
            <ArrowRight className="icon" size={16} aria-hidden="true" />
          </Link>
        )}
      </nav>
    </article>
  );
}
