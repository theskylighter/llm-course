import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// content/site.json is the single source of truth for the title and storage prefix.
const site = JSON.parse(readFileSync(resolve(__dirname, "../content/site.json"), "utf-8")) as {
  title: string;
  storagePrefix: string;
};

export default defineConfig({
  plugins: [
    react(),
    {
      name: "site-config-in-html",
      transformIndexHtml: (html) =>
        html.replaceAll("%SITE_TITLE%", site.title).replaceAll("%SITE_PREFIX%", site.storagePrefix),
    },
  ],
  build: { outDir: "dist", sourcemap: false },
});
