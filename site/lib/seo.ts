import type { Metadata } from "next";

export const SITE_NAME = "Open Tool Embeddings";
export const SITE_TITLE = "Open Tool Embeddings";
export const SITE_DESCRIPTION =
  "Registry-first infrastructure for open-source tool embedding models, downloads, contribution workflows, and training snapshots.";
export const GITHUB_REPO_URL = "https://github.com/andykr1k/NTILC";
export const HUGGING_FACE_ORG_URL = "https://huggingface.co/OpenToolEmbeddings";
export const SITE_LOCALE = "en_US";
export const DEFAULT_KEYWORDS = [
  "open source tool embeddings",
  "tool embedding model",
  "hierarchical embeddings",
  "embedding model downloads",
  "tool registry",
  "open source AI tools",
  "tool routing",
  "tool retrieval",
  "Next.js",
  "Tailwind CSS",
];

function normalizeUrl(value: string) {
  return value.startsWith("http://") || value.startsWith("https://") ? value : `https://${value}`;
}

export function getSiteUrl() {
  const configured =
    process.env.NEXT_PUBLIC_SITE_URL ||
    process.env.SITE_URL ||
    process.env.VERCEL_PROJECT_PRODUCTION_URL ||
    process.env.VERCEL_URL ||
    "http://localhost:3000";

  return new URL(normalizeUrl(configured));
}

export function getAbsoluteUrl(path = "/") {
  return new URL(path, getSiteUrl()).toString();
}

export function getSocialImagePath(path = "/") {
  return path === "/" ? "/opengraph-image" : `${path}/opengraph-image`;
}

export function getTwitterImagePath(path = "/") {
  return path === "/" ? "/twitter-image" : `${path}/twitter-image`;
}

type BuildMetadataInput = {
  title: string;
  description: string;
  path: string;
  keywords?: string[];
};

export function buildPageMetadata({
  title,
  description,
  path,
  keywords = [],
}: BuildMetadataInput): Metadata {
  const mergedKeywords = Array.from(new Set([...DEFAULT_KEYWORDS, ...keywords]));

  return {
    title,
    description,
    keywords: mergedKeywords,
    alternates: {
      canonical: path,
    },
    openGraph: {
      title,
      description,
      url: path,
      siteName: SITE_NAME,
      locale: SITE_LOCALE,
      type: "website",
      images: [
        {
          url: getSocialImagePath(path),
          width: 1200,
          height: 630,
          alt: title,
        },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
      images: [getTwitterImagePath(path)],
    },
  };
}

export function buildCollectionPageSchema({
  name,
  description,
  path,
}: {
  name: string;
  description: string;
  path: string;
}) {
  return {
    "@context": "https://schema.org",
    "@type": "CollectionPage",
    name,
    description,
    url: getAbsoluteUrl(path),
    isPartOf: {
      "@type": "WebSite",
      name: SITE_NAME,
      url: getAbsoluteUrl("/"),
    },
  };
}

export function buildHomeSchemas() {
  return [
    {
      "@context": "https://schema.org",
      "@type": "WebSite",
      name: SITE_NAME,
      description: SITE_DESCRIPTION,
      url: getAbsoluteUrl("/"),
      inLanguage: "en-US",
      sameAs: [GITHUB_REPO_URL, HUGGING_FACE_ORG_URL],
    },
    {
      "@context": "https://schema.org",
      "@type": "Organization",
      name: SITE_NAME,
      url: getAbsoluteUrl("/"),
      description: SITE_DESCRIPTION,
      sameAs: [GITHUB_REPO_URL, HUGGING_FACE_ORG_URL],
    },
  ];
}
