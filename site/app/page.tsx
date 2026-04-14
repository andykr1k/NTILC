import Link from "next/link";
import { StructuredData } from "@/components/structured-data";
import { getModelsRegistry, getRegistryManifest, getToolRegistry } from "@/lib/registry";
import { GITHUB_REPO_URL, HUGGING_FACE_ORG_URL, buildHomeSchemas } from "@/lib/seo";

const principles = [
  {
    id: "01",
    title: "Public registry",
    text: "The ontology, examples, parent ids, and model release metadata stay in one open-source surface instead of being buried in private infrastructure.",
  },
  {
    id: "02",
    title: "Hierarchy-aware",
    text: "Tools are not only flat labels. The project treats parent ids as first-class training structure for hierarchical retrieval and broader tool families.",
  },
  {
    id: "03",
    title: "Always live",
    text: "The dataset is meant to evolve with open-source tooling, not freeze around a one-time benchmark snapshot or a vendor-specific schema.",
  },
];

export default async function HomePage() {
  const [manifest, modelRegistry, toolRegistry] = await Promise.all([
    getRegistryManifest(),
    getModelsRegistry(),
    getToolRegistry(),
  ]);

  const publishedCount = modelRegistry.releases.filter((item) => item.status === "published").length;

  return (
    <div className="space-y-[4.5rem] pb-20 xl:space-y-20">
      <StructuredData data={buildHomeSchemas()} />

      <section className="grid items-start gap-6 xl:grid-cols-[1.18fr_0.82fr]">
        <div className="panel-strong reveal rounded-none px-6 py-9 sm:px-8 sm:py-11 lg:px-10">
          <div className="hero-mark absolute left-[-0.08em] top-[-0.16em] text-[6.5rem] sm:text-[10rem] lg:text-[12rem]">
            Open
          </div>
          <div className="relative z-10 max-w-[min(100%,72rem)] pr-1 sm:pr-4 lg:pr-10">
            <p className="eyebrow">Cold Industrial Tech</p>
            <h1 className="mt-6 max-w-[11.8ch] text-[clamp(3.75rem,8vw,7rem)] leading-[0.9]">
              The open-source layer for tool embeddings.
            </h1>
            <p className="mt-7 max-w-3xl text-lg leading-8 text-[color:var(--muted)]">
              Open Tool Embeddings is building a public, community-evolving embedding set for open-source tools, with
              GitHub as the collaboration surface and Hugging Face as the model distribution surface.
            </p>

            <div className="mt-9 flex flex-wrap gap-3">
              <Link href="/mission" className="primary-button rounded-none px-5 py-3 text-sm font-semibold uppercase tracking-[0.16em]">
                Enter Mission
              </Link>
              <a
                href={GITHUB_REPO_URL}
                target="_blank"
                rel="noreferrer"
                className="ghost-button rounded-none px-5 py-3 text-sm font-semibold uppercase tracking-[0.16em]"
              >
                View GitHub
              </a>
              <a
                href={HUGGING_FACE_ORG_URL}
                target="_blank"
                rel="noreferrer"
                className="ghost-button rounded-none px-5 py-3 text-sm font-semibold uppercase tracking-[0.16em]"
              >
                View Hugging Face
              </a>
            </div>

            <div className="mt-12 grid gap-3 sm:grid-cols-3 lg:mt-14">
              <div className="metric-chip reveal delay-1 rounded-none px-4 py-4">
                <div className="eyebrow">Registry</div>
                <div className="mt-3 text-4xl font-semibold text-[color:var(--ink)]">{manifest.tool_count}</div>
                <div className="mt-2 text-sm uppercase tracking-[0.14em] text-[color:var(--muted)]">tools tracked</div>
              </div>
              <div className="metric-chip reveal delay-2 rounded-none px-4 py-4">
                <div className="eyebrow">Snapshot</div>
                <div className="mt-3 text-4xl font-semibold text-[color:var(--ink)]">{manifest.example_count}</div>
                <div className="mt-2 text-sm uppercase tracking-[0.14em] text-[color:var(--muted)]">example prompts</div>
              </div>
              <div className="metric-chip reveal delay-3 rounded-none px-4 py-4">
                <div className="eyebrow">Releases</div>
                <div className="mt-3 text-4xl font-semibold text-[color:var(--ink)]">{modelRegistry.releases.length}</div>
                <div className="mt-2 text-sm uppercase tracking-[0.14em] text-[color:var(--muted)]">model variants</div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid gap-5 xl:pt-6">
          <aside className="signal-grid reveal delay-1 min-h-[29rem] rounded-none px-6 py-6">
            <div className="relative z-10 flex h-full flex-col justify-between">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="eyebrow">Signal Map</p>
                  <div className="mt-3 max-w-xs text-3xl font-semibold text-[color:var(--ink)]">
                    Registry activity rendered like machine telemetry.
                  </div>
                </div>
                <div className="text-right font-mono text-xs uppercase tracking-[0.18em] text-[color:var(--muted)]">
                  <div>axis: tool</div>
                  <div>axis: parent_id</div>
                  <div>axis: release</div>
                </div>
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                <div className="metric-chip rounded-none px-4 py-4">
                  <div className="text-xs uppercase tracking-[0.16em] text-[color:var(--muted)]">Categories</div>
                  <div className="mt-2 text-3xl font-semibold text-[color:var(--ink)]">{toolRegistry.categories.length}</div>
                </div>
                <div className="metric-chip rounded-none px-4 py-4">
                  <div className="text-xs uppercase tracking-[0.16em] text-[color:var(--muted)]">Published</div>
                  <div className="mt-2 text-3xl font-semibold text-[color:var(--ink)]">{publishedCount}</div>
                </div>
              </div>
            </div>
          </aside>

          <div className="panel steel-card reveal delay-2 ml-0 rounded-none px-6 py-6">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="eyebrow">Distribution</p>
                <h2 className="mt-4 text-4xl">GitHub for source. Hugging Face for models.</h2>
              </div>
            </div>
            <p className="mt-5 max-w-lg text-base leading-8 text-[color:var(--muted)]">
              The repo explains the system and accepts tool contributions. The OpenToolEmbeddings organization is where
              checkpoints should live so downloads stay obvious and centralized.
            </p>
          </div>
        </div>
      </section>

      <section className="grid gap-5 xl:grid-cols-[0.85fr_1.15fr]">
        <div className="panel reveal delay-1 rounded-none px-6 py-7 xl:translate-y-10">
          <p className="eyebrow">What This Project Is</p>
          <h2 className="mt-5 text-5xl">A shared representation layer for tools.</h2>
          <p className="mt-5 text-base leading-8 text-[color:var(--muted)]">
            Instead of mapping each query directly to a closed tool list, the project builds a public embedding space
            over tool metadata and examples. That space can support routing, retrieval, clustering, and hierarchical
            reasoning across open-source tooling.
          </p>
          <div className="mt-6 flex flex-wrap gap-2">
            {["registry-first", "hierarchical", "retrieval", "public checkpoints"].map((item) => (
              <span key={item} className="metric-chip rounded-none px-3 py-2 text-xs uppercase tracking-[0.14em] text-[color:var(--ink)]">
                {item}
              </span>
            ))}
          </div>
        </div>

        <div className="grid gap-4">
          {principles.map((item, index) => (
            <article
              key={item.id}
              className={`panel steel-card reveal rounded-none px-6 py-6 ${index === 1 ? "xl:ml-12" : index === 2 ? "xl:ml-24" : ""}`}
              style={{ animationDelay: `${180 + index * 80}ms` }}
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="eyebrow">{item.id}</p>
                  <h3 className="mt-4 text-4xl">{item.title}</h3>
                </div>
                <div className="steel-index text-sm font-semibold uppercase tracking-[0.18em] text-[color:var(--muted)]">
                  signal
                </div>
              </div>
              <p className="mt-5 max-w-2xl text-base leading-8 text-[color:var(--muted)]">{item.text}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="grid gap-5 lg:grid-cols-[1.2fr_0.8fr]">
        <div className="panel-strong reveal delay-2 rounded-none px-6 py-8 sm:px-8">
          <p className="eyebrow">Workflow</p>
          <h2 className="mt-5 text-5xl sm:text-6xl">Contribute the tool. Build the snapshot. Train the release.</h2>
          <div className="mt-8 grid gap-4 lg:grid-cols-3">
            {[
              "Add a tool manifest and examples in GitHub.",
              "Set a valid parent_id for the hierarchical model.",
              "Train from generated registry artifacts and publish to Hugging Face.",
            ].map((step, index) => (
              <div key={step} className="metric-chip rounded-none px-4 py-4">
                <div className="eyebrow">0{index + 1}</div>
                <p className="mt-3 text-base leading-7 text-[color:var(--ink)]">{step}</p>
              </div>
            ))}
          </div>
        </div>
        <div className="grid gap-4">
          <Link href="/docs" className="panel steel-card reveal delay-3 rounded-none px-6 py-6">
            <div className="eyebrow">Training</div>
            <h3 className="mt-4 text-4xl">Read the training path</h3>
            <p className="mt-4 text-base leading-8 text-[color:var(--muted)]">
              Start from the registry build and use the wrapper script for all model variants.
            </p>
          </Link>
          <Link href="/submit" className="panel steel-card reveal delay-4 rounded-none px-6 py-6 lg:ml-12">
            <div className="eyebrow">Submission</div>
            <h3 className="mt-4 text-4xl">Add the next tool to the public set</h3>
            <p className="mt-4 text-base leading-8 text-[color:var(--muted)]">
              The registry is intentionally empty until the community begins filling it with real tools.
            </p>
          </Link>
        </div>
      </section>
    </div>
  );
}
