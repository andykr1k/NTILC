import { createSocialImage, socialImageContentType, twitterSize } from "@/lib/social-image";

export const alt = "Training and publishing docs";
export const size = twitterSize;
export const contentType = socialImageContentType;

export default function Image() {
  return createSocialImage({
    eyebrow: "Docs",
    title: "Build the registry, train the models, and publish releases.",
    description: "Step-by-step commands for compiling manifests, training embeddings, and shipping public checkpoints.",
    chips: ["registry", "training", "releases"],
    size: twitterSize,
  });
}
