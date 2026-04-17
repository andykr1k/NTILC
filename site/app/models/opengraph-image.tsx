import { createSocialImage, ogSize, socialImageContentType } from "@/lib/social-image";

export const alt = "Embedding model downloads";
export const size = ogSize;
export const contentType = socialImageContentType;

export default function Image() {
  return createSocialImage({
    eyebrow: "Models",
    title: "Download normal and hierarchical embedding variants.",
    description: "Compare losses, checkpoints, dataset versions, and release status from one public registry.",
    chips: ["normal", "hierarchical", "prototype_ce", "contrastive", "functional_margin"],
  });
}
