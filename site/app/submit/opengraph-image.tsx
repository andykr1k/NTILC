import { createSocialImage, ogSize, socialImageContentType } from "@/lib/social-image";

export const alt = "Submit a tool";
export const size = ogSize;
export const contentType = socialImageContentType;

export default function Image() {
  return createSocialImage({
    eyebrow: "Submit",
    title: "Accept new tools through review, not through direct uploads.",
    description: "Contribute manifests and examples through a transparent issue and pull request workflow.",
    chips: ["PR workflow", "tool submission", "review"],
  });
}
