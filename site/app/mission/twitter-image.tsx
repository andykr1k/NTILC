import { createSocialImage, socialImageContentType, twitterSize } from "@/lib/social-image";

export const alt = "Open Tool Embeddings mission";
export const size = twitterSize;
export const contentType = socialImageContentType;

export default function Image() {
  return createSocialImage({
    eyebrow: "Mission",
    title: "A community-built open-source tool embedding set that keeps evolving.",
    description: "Open submissions, explicit hierarchy labels, public training snapshots, and public model releases.",
    chips: ["mission", "registry", "community"],
    size: twitterSize,
  });
}
