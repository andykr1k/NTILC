import type { Metadata } from "next";
import { PageHeader } from "@/components/page-header";
import { GITHUB_REPO_URL, HUGGING_FACE_ORG_URL, buildPageMetadata } from "@/lib/seo";

const buildCommand = `python3 scripts/build_registry.py`;

const trainAllCommand = `bash scripts/train_registry_embedding_spaces.sh`;

const trainNormalCommand = `python3 -m training.train_embedding_space \\
  --dataset-path registry/generated/tool_embedding_dataset.jsonl \\
  --output-dir output/registry_embeddings \\
  --loss-type contrastive`;

const trainHierarchicalCommand = `python3 -m training.train_hierarchical_embedding_space \\
  --dataset-path registry/generated/tool_embedding_dataset.jsonl \\
  --hierarchy-path registry/generated/hierarchy.json \\
  --output-dir output/registry_embeddings \\
  --loss-type contrastive`;

const publishNotes = [
  "Upload the finished checkpoint and model card to Hugging Face or your preferred artifact host.",
  "Add the public URL, checksum, dataset version, and metrics to registry/models/releases.yaml.",
  "Re-run the registry builder so the downloads page exposes the new release.",
];

export const metadata: Metadata = buildPageMetadata({
  title: "Training and Publishing Docs",
  description:
    "Build the registry, train normal and hierarchical embedding variants, and publish model releases with generated manifests.",
  path: "/docs",
  keywords: ["training docs", "embedding training", "registry build", "model release process"],
});

export default function DocsPage() {
  return (
    <div className="space-y-10 pb-16">
      <PageHeader
        eyebrow="Docs"
        title="Train from registry snapshots and publish to Hugging Face."
        summary="The workflow mirrors the repo structure: contributors add tools in GitHub, include a valid `parent_id`, build the registry snapshot, train from generated artifacts, then publish downloadable checkpoints to the OpenToolEmbeddings Hugging Face organization."
        meta={
          <div className="space-y-3">
            <a
              href={GITHUB_REPO_URL}
              target="_blank"
              rel="noreferrer"
              className="ghost-button inline-flex rounded-none px-4 py-2 text-sm font-semibold uppercase tracking-[0.14em]"
            >
              GitHub workflow
            </a>
            <a
              href={HUGGING_FACE_ORG_URL}
              target="_blank"
              rel="noreferrer"
              className="primary-button inline-flex rounded-none px-4 py-2 text-sm font-semibold uppercase tracking-[0.14em]"
            >
              Hugging Face org
            </a>
          </div>
        }
      />

      <section className="grid gap-4 lg:grid-cols-2 xl:grid-cols-4">
        <article className="panel steel-card reveal delay-1 rounded-none px-6 py-6">
          <p className="eyebrow">1. Build</p>
          <h2 className="mt-3 text-2xl font-semibold">Compile the registry</h2>
          <p className="mt-4 text-sm leading-7 text-[color:var(--muted)]">
            This validates manifests and writes `tools.json`, `models.json`, `hierarchy.json`, and the flat JSONL
            dataset used for training.
          </p>
          <pre className="mt-5 overflow-x-auto border border-[color:var(--line)] bg-[rgba(5,10,16,0.95)] px-4 py-4 text-sm text-stone-100">
            <code>{buildCommand}</code>
          </pre>
        </article>

        <article className="panel steel-card reveal delay-2 rounded-none px-6 py-6">
          <p className="eyebrow">2. Easy Path</p>
          <h2 className="mt-3 text-2xl font-semibold">Train every variant</h2>
          <p className="mt-4 text-sm leading-7 text-[color:var(--muted)]">
            The wrapper script uses the generated registry snapshot and trains normal plus hierarchical variants across
            all supported losses.
          </p>
          <pre className="mt-5 overflow-x-auto border border-[color:var(--line)] bg-[rgba(5,10,16,0.95)] px-4 py-4 text-sm text-stone-100">
            <code>{trainAllCommand}</code>
          </pre>
        </article>

        <article className="panel steel-card reveal delay-3 rounded-none px-6 py-6">
          <p className="eyebrow">3. Advanced</p>
          <h2 className="mt-3 text-2xl font-semibold">Normal embeddings</h2>
          <p className="mt-4 text-sm leading-7 text-[color:var(--muted)]">
            Use the flat registry dataset to train the baseline architecture with your chosen loss.
          </p>
          <pre className="mt-5 overflow-x-auto border border-[color:var(--line)] bg-[rgba(5,10,16,0.95)] px-4 py-4 text-sm text-stone-100">
            <code>{trainNormalCommand}</code>
          </pre>
        </article>

        <article className="panel steel-card reveal delay-4 rounded-none px-6 py-6">
          <p className="eyebrow">4. Advanced</p>
          <h2 className="mt-3 text-2xl font-semibold">Hierarchical embeddings</h2>
          <p className="mt-4 text-sm leading-7 text-[color:var(--muted)]">
            The hierarchy mapping is generated from each tool's `parent_id`, so the site and the trainer stay in sync.
          </p>
          <pre className="mt-5 overflow-x-auto border border-[color:var(--line)] bg-[rgba(5,10,16,0.95)] px-4 py-4 text-sm text-stone-100">
            <code>{trainHierarchicalCommand}</code>
          </pre>
        </article>
      </section>

      <section className="panel reveal delay-4 rounded-none px-6 py-7">
        <p className="eyebrow">Release Process</p>
        <h2 className="mt-3 text-3xl font-semibold">Publishing a downloadable model</h2>
        <div className="mt-5 grid gap-4 md:grid-cols-3">
          {publishNotes.map((note, index) => (
            <div key={note} className="metric-chip rounded-none px-5 py-5">
              <div className="text-sm font-semibold uppercase tracking-[0.16em] text-[color:var(--accent)]">
                Step 0{index + 1}
              </div>
              <p className="mt-3 text-sm leading-7">{note}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
