import type { MetadataRoute } from "next";
import { getRegistryManifest } from "@/lib/registry";
import { getAbsoluteUrl, getSocialImagePath } from "@/lib/seo";

const ROUTES = [
  { path: "/", changeFrequency: "weekly", priority: 1 },
  { path: "/mission", changeFrequency: "weekly", priority: 0.92 },
  { path: "/models", changeFrequency: "daily", priority: 0.95 },
  { path: "/tools", changeFrequency: "daily", priority: 0.95 },
  { path: "/docs", changeFrequency: "weekly", priority: 0.8 },
  { path: "/submit", changeFrequency: "monthly", priority: 0.65 },
] as const;

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const manifest = await getRegistryManifest();
  const lastModified = manifest.generated_at;

  return ROUTES.map((route) => ({
    url: getAbsoluteUrl(route.path),
    lastModified,
    changeFrequency: route.changeFrequency,
    priority: route.priority,
    images: [getAbsoluteUrl(getSocialImagePath(route.path))],
  }));
}
