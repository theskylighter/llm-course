import { useEffect, useRef } from "react";

const NATIVE = typeof CSS !== "undefined" && CSS.supports("animation-timeline: scroll()");

/**
 * Thin reading-progress bar along the bottom edge of the top bar. It is only displayed below the
 * breakpoint where the "On this page" panel is hidden (see `.reading-progress` in styles.css).
 * Browsers with scroll-driven animations need no JS; others get a rAF-throttled scroll listener.
 */
export function ReadingProgress() {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const bar = ref.current;
    if (NATIVE || !bar) return;
    let frame = 0;
    const update = () => {
      frame = 0;
      const max = document.documentElement.scrollHeight - window.innerHeight;
      bar.style.transform = `scaleX(${max > 0 ? Math.min(1, window.scrollY / max) : 0})`;
    };
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(update);
    };
    update();
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    return () => {
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
      cancelAnimationFrame(frame);
    };
  }, []);

  return <div ref={ref} className="reading-progress" aria-hidden="true" />;
}
