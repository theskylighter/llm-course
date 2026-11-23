import { Route, Routes } from "react-router-dom";
import { Home } from "./components/Home";
import { Layout } from "./components/Layout";
import { LessonPage } from "./components/LessonPage";
import { NotFound } from "./components/NotFound";

/** Every page shares the persistent Layout; only the <Outlet /> content changes. */
export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Home />} />
        <Route path=":slug" element={<LessonPage />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
