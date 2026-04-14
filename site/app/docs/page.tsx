import type { Metadata } from "next";
import { PageHeader } from "@/components/page-header";
import { GITHUB_REPO_URL, HUGGING_FACE_ORG_URL, buildPageMetadata } from "@/lib/seo";

const importCommand = `python3 scripts/import_oss_registry.py`;

const buildCommand = `python3 scripts/build_registry.py`;

const trainAllCommand = `bash scripts/train_registry_embedding_spaces.sh`;

const syncReleasesCommand = `python3 scripts/sync_model_releases.py`;

const publishCommand = `python3 scripts/publish_huggingface_models.py`;

const publishNotes = [
  "Run the release sync to compute checksums, summarize metrics, and generate model cards from local checkpoints.",
  "Publish the prepared bundle to the OpenToolEmbeddings Hugging Face organization once `HF_TOKEN` and `huggingface_hub` are available.",
  "Re-run the registry builder so the downloads page exposes the published URL and status.",
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
        summary="The workflow mirrors the repo structure: import or add tools into `registry/`, include a valid `parent_id`, build the generated snapshot, train from those artifacts, sync local release metadata, then publish downloadable checkpoints to the OpenToolEmbeddings Hugging Face organization."
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
          <p className="eyebrow">1. Import</p>
          <h2 className="mt-3 text-2xl font-semibold">Materialize the OSS baseline</h2>
          <p className="mt-4 text-sm leading-7 text-[color:var(--muted)]">
            Convert the current OSS tool JSON and synthetic examples into `registry/tools/*` manifests that the site
            and builder can read.
          </p>
          <pre className="mt-5 overflow-x-auto border border-[color:var(--line)] bg-[rgba(5,10,16,0.95)] px-4 py-4 text-sm text-stone-100">
            <code>{importCommand}</code>
          </pre>
        </article>

        <article className="panel steel-card reveal delay-2 rounded-none px-6 py-6">
          <p className="eyebrow">2. Build</p>
          <h2 className="mt-3 text-2xl font-semibold">Compile the registry</h2>
          <p className="mt-4 text-sm leading-7 text-[color:var(--muted)]">
            Validate manifests and write `tools.json`, `models.json`, `hierarchy.json`, and the flat JSONL dataset
            used for training and the site.
          </p>
          <pre className="mt-5 overflow-x-auto border border-[color:var(--line)] bg-[rgba(5,10,16,0.95)] px-4 py-4 text-sm text-stone-100">
            <code>{buildCommand}</code>
          </pre>
        </article>

        <article className="panel steel-card reveal delay-3 rounded-none px-6 py-6">
          <p className="eyebrow">3. Easy Path</p>
          <h2 className="mt-3 text-2xl font-semibold">Train every variant</h2>
          <p className="mt-4 text-sm leading-7 text-[color:var(--muted)]">
            The wrapper script uses the generated registry snapshot and trains normal plus hierarchical variants across
            all supported losses.
          </p>
          <pre className="mt-5 overflow-x-auto border border-[color:var(--line)] bg-[rgba(5,10,16,0.95)] px-4 py-4 text-sm text-stone-100">
            <code>{trainAllCommand}</code>
          </pre>
        </article>

        <article className="panel steel-card reveal delay-4 rounded-none px-6 py-6">
          <p className="eyebrow">4. Release</p>
          <h2 className="mt-3 text-2xl font-semibold">Sync local release metadata</h2>
          <p className="mt-4 text-sm leading-7 text-[color:var(--muted)]">
            Compute checksums, summarize metrics, and generate model cards plus a publish manifest for the current
            checkpoint set.
          </p>
          <pre className="mt-5 overflow-x-auto border border-[color:var(--line)] bg-[rgba(5,10,16,0.95)] px-4 py-4 text-sm text-stone-100">
            <code>{syncReleasesCommand}</code>
          </pre>
        </article>
      </section>

      <section className="panel reveal delay-4 rounded-none px-6 py-7">
        <p className="eyebrow">Release Process</p>
        <h2 className="mt-3 text-3xl font-semibold">Publishing a downloadable model</h2>
        <pre className="mt-5 overflow-x-auto border border-[color:var(--line)] bg-[rgba(5,10,16,0.95)] px-4 py-4 text-sm text-stone-100">
          <code>{publishCommand}</code>
        </pre>
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
