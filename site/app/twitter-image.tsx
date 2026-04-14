import { createSocialImage, socialImageContentType, twitterSize } from "@/lib/social-image";

export const alt = "Open Tool Embeddings homepage";
export const size = twitterSize;
export const contentType = socialImageContentType;

export default function Image() {
  return createSocialImage({
    eyebrow: "Open Tool Embeddings",
    title: "Build a public, versioned embedding stack for open-source tool use.",
    description: "Registry-backed model downloads, tool curation, and training snapshots for open-source tools.",
    chips: ["registry", "downloads", "tool embeddings"],
    size: twitterSize,
  });
}
