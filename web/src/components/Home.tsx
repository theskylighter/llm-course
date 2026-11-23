import { Link } from "react-router-dom";
import { ListChecks } from "lucide-react";
import { course, lessonPath } from "../course";
import type { Lesson } from "../types";

const isLesson = (l: Lesson | undefined): l is Lesson => l !== undefined;

/** Course roadmap: one card per chapter, each lesson with its reading time. */
function Roadmap() {
  return (
    <div className="roadmap">
      {course.chapters
        .filter((c) => c.title !== "Start Here")
        .map((chapter) => {
          const lessons = chapter.lessons
            .map((slug) => course.lessons.find((l) => l.slug === slug))
            .filter(isLesson);
          const total = lessons.reduce((sum, l) => sum + l.minutes, 0);
          return (
            <div className="chapter-card" key={chapter.title}>
              <div className="chhead">
                <span>{chapter.title}</span>
                <span className="rt">
                  {lessons.length} lessons · ~{total} min
                </span>
              </div>
              <ul>
                {lessons.map((l) => (
                  <li key={l.slug}>
                    <Link to={lessonPath(l.slug)}>
                      <span className="ln">
                        {l.number || <ListChecks className="icon" size={14} aria-label="Practice set" />}
                      </span>
                      {l.title}
                    </Link>
                    <span className="rt">{l.minutes} min</span>
                  </li>
                ))}
              </ul>
            </div>
          );
        })}
    </div>
  );
}

/**
 * Home page: the intro Markdown (which owns the "Course Roadmap" heading), then the generated
 * roadmap cards where the <<ROADMAP>> placeholder sits, then the rest of the Markdown.
 */
export function Home() {
  const { beforeRoadmap, afterRoadmap } = course.intro;
  return (
    <>
      <div dangerouslySetInnerHTML={{ __html: beforeRoadmap }} />
      <Roadmap />
      <div dangerouslySetInnerHTML={{ __html: afterRoadmap }} />
    </>
  );
}
