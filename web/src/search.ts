import MiniSearch from "minisearch";

/** One searchable section of a lesson, written by tools/sitegen/search_index.py. */
export interface SearchRecord {
  id: string;
  slug: string;
  title: string;   // lesson title
  chapter: string;
  number: string;
  heading: string; // section heading, or "Deep dive: ..." for a collapsible walkthrough
  anchor: string;  // element id to scroll to; empty for the lead-in before the first heading
  kind: "section" | "deep";
  text: string;
}

export interface SearchHit extends SearchRecord {
  score: number;
  terms: string[]; // index terms that matched, used to highlight the snippet
}

export interface HitGroup {
  slug: string;
  title: string;
  chapter: string;
  number: string;
  hits: SearchHit[];
}

const STORED: (keyof SearchRecord)[] = ["slug", "title", "chapter", "number", "heading", "anchor", "kind", "text"];

const SPLIT = /[\s\-_^/.,;:()[\]{}'"“”‘’=+*<>|\\!?·→—–×≤≥≈∈~]+/u;

/** Words, plus the parts of camelCase words: "FlashAttention" is found by "flash" and by "attention". */
function tokenize(text: string): string[] {
  const out: string[] = [];
  for (const word of text.split(SPLIT)) {
    if (!word) continue;
    out.push(word);
    const parts = word.match(/[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+/g);
    if (parts && parts.length > 1) for (const p of parts) if (p.length > 1) out.push(p);
  }
  return out;
}

let loading: Promise<MiniSearch<SearchRecord>> | null = null;

/** Download and index the search data once. It is a separate chunk, so normal page loads never pay for it. */
export function loadSearch(): Promise<MiniSearch<SearchRecord>> {
  loading ??= import("./data/search.json").then((mod) => {
    const index = new MiniSearch<SearchRecord>({
      fields: ["heading", "title", "text"],
      storeFields: STORED,
      idField: "id",
      tokenize,
    });
    index.addAll(mod.default as unknown as SearchRecord[]);
    return index;
  });
  return loading;
}

const SEARCH_OPTIONS = {
  boost: { heading: 4, title: 2 }, // a word in a heading beats the same word in body text
  prefix: true,                    // "atten" finds "attention" while typing
  fuzzy: (term: string) => (term.length > 4 ? 0.2 : false), // tolerate a typo in longer words
} as const;

const MAX_HITS = 40;
const MAX_GROUPS = 8;
const MAX_PER_GROUP = 4;

export async function runSearch(query: string): Promise<HitGroup[]> {
  const q = query.trim();
  if (!q) return [];
  const index = await loadSearch();
  // Every word must match; if nothing does, fall back to any word so there is always a suggestion.
  let results = index.search(q, { ...SEARCH_OPTIONS, combineWith: "AND" });
  if (!results.length) results = index.search(q, { ...SEARCH_OPTIONS, combineWith: "OR" });

  const groups = new Map<string, HitGroup>();
  for (const r of results.slice(0, MAX_HITS)) {
    const hit = r as unknown as SearchHit;
    let g = groups.get(hit.slug);
    if (!g) {
      if (groups.size >= MAX_GROUPS) continue;
      g = { slug: hit.slug, title: hit.title, chapter: hit.chapter, number: hit.number, hits: [] };
      groups.set(hit.slug, g);
    }
    if (g.hits.length < MAX_PER_GROUP) g.hits.push(hit);
  }
  return [...groups.values()];
}

/** The words to highlight: what was typed plus the full words it matched. Longest first. */
export function highlightTerms(query: string, hit: SearchHit): string[] {
  const typed = query.toLowerCase().split(/[\s\-_/.,;:()]+/).filter(Boolean);
  return [...new Set([...typed, ...hit.terms])].sort((a, b) => b.length - a.length);
}

export function termsRegex(terms: string[]): RegExp | null {
  if (!terms.length) return null;
  const escaped = terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  return new RegExp(`(?<![\\p{L}\\p{N}])(${escaped.join("|")})`, "giu");
}

/** A short excerpt around the first match, so a result shows *why* it matched. */
export function snippet(text: string, re: RegExp | null, length = 170): string {
  let start = 0;
  if (re) {
    re.lastIndex = 0;
    const m = re.exec(text);
    if (m) start = Math.max(0, m.index - 55);
  }
  if (start > 0) {
    const space = text.indexOf(" ", start);
    if (space !== -1 && space - start < 20) start = space + 1; // don't begin mid-word
  }
  let out = text.slice(start, start + length);
  if (start + length < text.length) out = out.replace(/\s+\S*$/, "") + "…";
  return (start > 0 ? "…" : "") + out;
}
