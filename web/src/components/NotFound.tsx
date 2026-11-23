import { Link } from "react-router-dom";

export function NotFound() {
  return (
    <article className="lesson">
      <h1>Page not found</h1>
      <p>
        That lesson doesn't exist. <Link to="/">Back to the course home</Link>.
      </p>
    </article>
  );
}
