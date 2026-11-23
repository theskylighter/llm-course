import data from "./data/lessons.json";
import type { SiteInfo } from "./types";

/** Site-wide settings from content/site.json (title, brand, storage prefix). */
export const site = (data as unknown as { site: SiteInfo }).site;

/** localStorage / sessionStorage key, namespaced per site so two study sites never share state. */
export function storageKey(name: string): string {
  return `${site.storagePrefix}-${name}`;
}
