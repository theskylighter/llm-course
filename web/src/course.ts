import data from "./data/lessons.json";
import type { CourseData, Lesson } from "./types";

export const course = data as CourseData;

export const HOME_SLUG = "index";

const bySlug = new Map<string, Lesson>(course.lessons.map((l) => [l.slug, l]));

export function findLesson(slug: string): Lesson | undefined {
  return bySlug.get(slug);
}

/** Lessons in reading order, used for previous/next links. */
export const lessonOrder: string[] = course.lessons.map((l) => l.slug);

export function lessonPath(slug: string): string {
  return slug === HOME_SLUG ? "/" : `/${slug}`;
}
