import { createSocialImage, ogSize, socialImageContentType } from "@/lib/social-image";

export const alt = "Open Tool Embeddings mission";
export const size = ogSize;
export const contentType = socialImageContentType;

export default function Image() {
  return createSocialImage({
    eyebrow: "Mission",
    title: "A community-built open-source tool embedding set that keeps evolving.",
    description: "Open submissions, explicit hierarchy labels, public training snapshots, and public model releases.",
    chips: ["community built", "parent id", "open source", "evolving registry"],
  });
}
