import { createSocialImage, ogSize, socialImageContentType } from "@/lib/social-image";

export const alt = "Open Tool Embeddings homepage";
export const size = ogSize;
export const contentType = socialImageContentType;

export default function Image() {
  return createSocialImage({
    eyebrow: "Open Tool Embeddings",
    title: "Build a public, versioned embedding stack for open-source tool use.",
    description: "Registry-backed model downloads, tool curation, and training snapshots for open-source tools.",
    chips: ["registry", "models", "hierarchical", "open source"],
  });
}
