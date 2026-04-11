import Link from "next/link";
import { getModelsRegistry, getRegistryManifest, getToolRegistry } from "@/lib/registry";

const workflow = [
  {
    step: "Registry",
    text: "Public tool manifests, example prompts, and category labels live in one reviewable place.",
  },
  {
    step: "Compile",
    text: "A build step turns contributor files into a flat dataset, hierarchy mapping, and site manifests.",
  },
  {
    step: "Train",
    text: "Normal and hierarchical variants train from the same snapshot with prototype-CE, contrastive, or circle loss.",
  },
  {
    step: "Publish",
    text: "Model release metadata points the downloads page at the newest public checkpoints without redeploying code.",
  },
];

const promises = [
  "Keep the registry public and reviewable.",
  "Separate submission, dataset generation, and training release cycles.",
  "Publish every model with dataset version, encoder, loss, and checksum metadata.",
];

export default async function HomePage() {
  const [toolRegistry, modelRegistry, manifest] = await Promise.all([
    getToolRegistry(),
    getModelsRegistry(),
    getRegistryManifest(),
  ]);

  const featuredTools = toolRegistry.tools.slice(0, 3);
  const publishedCount = modelRegistry.releases.filter((item) => item.status === "published").length;

  return (
    <div className="space-y-20 pb-20">
      <section className="grid gap-6 lg:grid-cols-[1.35fr_0.9fr]">
        <div className="panel-strong card-enter rounded-[2rem] px-6 py-8 sm:px-8 sm:py-10">
          <p className="eyebrow">Open Tool Embeddings</p>
          <h1 className="mt-4 max-w-4xl text-5xl leading-[0.95] font-semibold sm:text-6xl">
            Build a public, versioned embedding stack for open-source tool use.
          </h1>
          <p className="mt-6 max-w-2xl text-lg leading-8 text-[color:var(--muted)]">
            The site is the interface, not the source of truth. The source of truth lives in a central registry of
            tool manifests, training examples, hierarchy labels, and model releases.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link
              href="/models"
              className="rounded-full bg-[color:var(--accent-strong)] px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-orange-900/20 hover:-translate-y-0.5"
            >
              Browse model variants
            </Link>
            <Link
              href="/tools"
              className="rounded-full border border-[color:var(--line)] bg-white/65 px-5 py-3 text-sm font-semibold text-[color:var(--ink)] hover:-translate-y-0.5"
            >
              Inspect the tool registry
            </Link>
            <Link
              href="/submit"
              className="rounded-full border border-transparent px-5 py-3 text-sm font-semibold text-[color:var(--accent-strong)] hover:bg-white/50"
            >
              Submit a tool
            </Link>
          </div>
          <div className="mt-10 grid gap-3 sm:grid-cols-3">
            <div className="metric-chip rounded-[1.4rem] px-4 py-4">
              <div className="text-3xl font-semibold">{manifest.tool_count}</div>
              <div className="mt-1 text-sm text-[color:var(--muted)]">tools in the registry</div>
            </div>
            <div className="metric-chip rounded-[1.4rem] px-4 py-4">
              <div className="text-3xl font-semibold">{manifest.example_count}</div>
              <div className="mt-1 text-sm text-[color:var(--muted)]">training examples generated</div>
            </div>
            <div className="metric-chip rounded-[1.4rem] px-4 py-4">
              <div className="text-3xl font-semibold">{publishedCount}</div>
              <div className="mt-1 text-sm text-[color:var(--muted)]">published model releases</div>
            </div>
          </div>
        </div>

        <aside className="panel card-enter rounded-[2rem] px-6 py-8" style={{ animationDelay: "80ms" }}>
          <p className="eyebrow">Operating Model</p>
          <div className="mt-4 space-y-5">
            <div>
              <div className="text-sm font-semibold uppercase tracking-[0.18em] text-[color:var(--muted)]">
                Canonical home
              </div>
              <p className="mt-2 text-base leading-7">A GitHub-backed registry for manifests, examples, and review.</p>
            </div>
            <div>
              <div className="text-sm font-semibold uppercase tracking-[0.18em] text-[color:var(--muted)]">
                Distribution
              </div>
              <p className="mt-2 text-base leading-7">
                Hugging Face or another model host for weights, dataset snapshots, cards, and checksums.
              </p>
            </div>
            <div>
              <div className="text-sm font-semibold uppercase tracking-[0.18em] text-[color:var(--muted)]">
                Frontend role
              </div>
              <p className="mt-2 text-base leading-7">
                Next App Router pages expose models, registry state, docs, and contribution instructions from generated
                manifests.
              </p>
            </div>
          </div>
        </aside>
      </section>

      <section className="grid gap-5 lg:grid-cols-3">
        {promises.map((promise, index) => (
          <div key={promise} className="panel card-enter rounded-[1.8rem] px-6 py-6" style={{ animationDelay: `${120 + index * 70}ms` }}>
            <p className="eyebrow">Principle {index + 1}</p>
            <p className="mt-4 text-lg leading-8">{promise}</p>
          </div>
        ))}
      </section>

      <section>
        <div className="flex items-end justify-between gap-4">
          <div>
            <p className="eyebrow">Pipeline</p>
            <h2 className="mt-3 text-4xl font-semibold">Registry-first training flow</h2>
          </div>
          <Link href="/docs" className="text-sm font-semibold text-[color:var(--accent-strong)]">
            Full docs
          </Link>
        </div>
        <div className="mt-8 grid gap-4 lg:grid-cols-4">
          {workflow.map((item, index) => (
            <article key={item.step} className="panel card-enter rounded-[1.7rem] px-5 py-6" style={{ animationDelay: `${160 + index * 60}ms` }}>
              <div className="flex items-center justify-between">
                <p className="eyebrow">{item.step}</p>
                <span className="text-sm font-semibold text-[color:var(--muted)]">0{index + 1}</span>
              </div>
              <p className="mt-4 text-base leading-7 text-[color:var(--ink)]">{item.text}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="grid gap-6 lg:grid-cols-[1fr_1.15fr]">
        <div className="panel rounded-[1.8rem] px-6 py-7">
          <p className="eyebrow">What Ships</p>
          <h2 className="mt-3 text-4xl font-semibold">Two architectures, three losses, one registry snapshot.</h2>
          <p className="mt-5 max-w-xl text-base leading-8 text-[color:var(--muted)]">
            Your existing training code already supports `normal` and `hierarchical` variants across
            `prototype_ce`, `contrastive`, and `circle`. This site keeps those choices visible instead of burying them
            inside filenames.
          </p>
          <div className="mt-6 flex flex-wrap gap-2 text-sm">
            {["normal", "hierarchical", "prototype_ce", "contrastive", "circle"].map((label) => (
              <span key={label} className="metric-chip rounded-full px-3 py-1.5">
                {label}
              </span>
            ))}
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-3">
          {featuredTools.map((tool, index) => (
            <article key={tool.id} className="panel rounded-[1.6rem] px-5 py-6">
              <p className="eyebrow">Featured Tool {index + 1}</p>
              <h3 className="mt-3 text-2xl font-semibold">{tool.display_name}</h3>
              <p className="mt-3 text-sm leading-7 text-[color:var(--muted)]">{tool.description}</p>
              <div className="mt-5 flex flex-wrap gap-2">
                {tool.tags.slice(0, 3).map((tag) => (
                  <span key={tag} className="metric-chip rounded-full px-3 py-1 text-xs uppercase tracking-[0.12em]">
                    {tag}
                  </span>
                ))}
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
