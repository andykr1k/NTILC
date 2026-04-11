import { createSocialImage, socialImageContentType, twitterSize } from "@/lib/social-image";

export const alt = "Embedding model downloads";
export const size = twitterSize;
export const contentType = socialImageContentType;

export default function Image() {
  return createSocialImage({
    eyebrow: "Models",
    title: "Download normal and hierarchical embedding variants.",
    description: "Compare losses, checkpoints, dataset versions, and release status from one public registry.",
    chips: ["downloads", "checkpoints", "registry"],
    size: twitterSize,
  });
}
