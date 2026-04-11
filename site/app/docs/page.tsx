import { PageHeader } from "@/components/page-header";

const buildCommand = `python3 scripts/build_registry.py`;

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

export default function DocsPage() {
  return (
    <div className="space-y-10 pb-16">
      <PageHeader
        eyebrow="Docs"
        title="Train from registry snapshots, not from scattered files."
        summary="The operational rule is simple: contributors edit `registry/`, the build step compiles the source of truth into generated manifests, and training consumes those generated artifacts."
      />

      <section className="grid gap-4 lg:grid-cols-3">
        <article className="panel rounded-[1.7rem] px-6 py-6">
          <p className="eyebrow">1. Build</p>
          <h2 className="mt-3 text-2xl font-semibold">Compile the registry</h2>
          <p className="mt-4 text-sm leading-7 text-[color:var(--muted)]">
            This validates manifests and writes `tools.json`, `models.json`, `hierarchy.json`, and the flat JSONL
            dataset used for training.
          </p>
          <pre className="mt-5 overflow-x-auto rounded-[1.4rem] bg-stone-950 px-4 py-4 text-sm text-stone-100">
            <code>{buildCommand}</code>
          </pre>
        </article>

        <article className="panel rounded-[1.7rem] px-6 py-6">
          <p className="eyebrow">2. Train</p>
          <h2 className="mt-3 text-2xl font-semibold">Normal embeddings</h2>
          <p className="mt-4 text-sm leading-7 text-[color:var(--muted)]">
            Use the flat registry dataset to train the baseline architecture with your chosen loss.
          </p>
          <pre className="mt-5 overflow-x-auto rounded-[1.4rem] bg-stone-950 px-4 py-4 text-sm text-stone-100">
            <code>{trainNormalCommand}</code>
          </pre>
        </article>

        <article className="panel rounded-[1.7rem] px-6 py-6">
          <p className="eyebrow">3. Train</p>
          <h2 className="mt-3 text-2xl font-semibold">Hierarchical embeddings</h2>
          <p className="mt-4 text-sm leading-7 text-[color:var(--muted)]">
            The hierarchy mapping is generated from each tool's `parent_category`, so the site and the trainer stay in
            sync.
          </p>
          <pre className="mt-5 overflow-x-auto rounded-[1.4rem] bg-stone-950 px-4 py-4 text-sm text-stone-100">
            <code>{trainHierarchicalCommand}</code>
          </pre>
        </article>
      </section>

      <section className="panel rounded-[1.8rem] px-6 py-7">
        <p className="eyebrow">Release Process</p>
        <h2 className="mt-3 text-3xl font-semibold">Publishing a downloadable model</h2>
        <div className="mt-5 grid gap-4 md:grid-cols-3">
          {publishNotes.map((note, index) => (
            <div key={note} className="metric-chip rounded-[1.4rem] px-5 py-5">
              <div className="text-sm font-semibold uppercase tracking-[0.16em] text-[color:var(--accent-strong)]">
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
