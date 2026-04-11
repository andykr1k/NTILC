import { createSocialImage, socialImageContentType, twitterSize } from "@/lib/social-image";

export const alt = "Open-source tool registry";
export const size = twitterSize;
export const contentType = socialImageContentType;

export default function Image() {
  return createSocialImage({
    eyebrow: "Tools",
    title: "Curate the source-of-truth registry before you train.",
    description: "Searchable open-source tool manifests, examples, categories, and interface metadata.",
    chips: ["tool registry", "schemas", "examples"],
    size: twitterSize,
  });
}
