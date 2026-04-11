import type { Metadata } from "next";
import Link from "next/link";
import { PageHeader } from "@/components/page-header";
import { StructuredData } from "@/components/structured-data";
import { GITHUB_REPO_URL, HUGGING_FACE_ORG_URL, buildPageMetadata, getAbsoluteUrl } from "@/lib/seo";

const manifesto = [
  {
    id: "01",
    title: "The registry should stay open.",
    text: "The definitions, examples, hierarchy, and releases should be inspectable and editable by the community, not trapped inside a private orchestration layer.",
  },
  {
    id: "02",
    title: "The hierarchy should stay explicit.",
    text: "A tool embedding project needs parent ids and broader families of behavior, not only flat labels with no structure for the model to exploit.",
  },
  {
    id: "03",
    title: "The dataset should stay alive.",
    text: "Open-source tooling changes too quickly for a frozen benchmark to remain relevant. The set has to keep evolving through new submissions and new snapshots.",
  },
  {
    id: "04",
    title: "The releases should stay public.",
    text: "Builders should not have to guess where the latest checkpoints live. The Hugging Face organization should be the obvious download destination.",
  },
];

export const metadata: Metadata = buildPageMetadata({
  title: "Mission",
  description:
    "A community-built, hierarchy-aware, always-evolving open-source tool embedding set with public releases.",
  path: "/mission",
  keywords: ["open-source mission", "tool embedding manifesto", "public hierarchy", "community registry"],
});

export default function MissionPage() {
  return (
    <div className="space-y-12 pb-16">
      <StructuredData
        data={{
          "@context": "https://schema.org",
          "@type": "AboutPage",
          name: "Open Tool Embeddings mission",
          description:
            "Manifesto page for a public, community-built, hierarchy-aware tool embedding project.",
          url: getAbsoluteUrl("/mission"),
        }}
      />

      <PageHeader
        eyebrow="Mission"
        title="A community-built open-source tool embedding set that keeps moving."
        summary="The mission is not to publish one static model and stop. The mission is to keep building a public tool embedding layer that grows with open-source tooling, public submissions, explicit hierarchy, and public releases."
        meta={
          <div className="space-y-4">
            <a
              href={GITHUB_REPO_URL}
              target="_blank"
              rel="noreferrer"
              className="panel steel-card block rounded-none px-5 py-5"
            >
              <div className="eyebrow">GitHub</div>
              <div className="mt-3 text-3xl font-semibold text-[color:var(--ink)]">Contribute the registry</div>
            </a>
            <a
              href={HUGGING_FACE_ORG_URL}
              target="_blank"
              rel="noreferrer"
              className="panel steel-card block rounded-none px-5 py-5"
            >
              <div className="eyebrow">Hugging Face</div>
              <div className="mt-3 text-3xl font-semibold text-[color:var(--ink)]">Distribute the models</div>
            </a>
          </div>
        }
      />

      <section className="grid gap-5 xl:grid-cols-[0.8fr_1.2fr]">
        <div className="panel reveal delay-1 rounded-none px-6 py-7">
          <p className="eyebrow">Why This Exists</p>
          <h2 className="mt-5 text-5xl">Closed tool layers go stale.</h2>
          <p className="mt-5 text-base leading-8 text-[color:var(--muted)]">
            New open-source tools appear constantly, interfaces shift, categories blur, and useful training examples
            accumulate over time. A private or static tool set cannot represent that reality. An open registry can.
          </p>
        </div>

        <div className="grid gap-4">
          {manifesto.map((item, index) => (
            <article
              key={item.id}
              className={`panel steel-card reveal rounded-none px-6 py-6 ${index % 2 === 1 ? "xl:ml-16" : ""}`}
              style={{ animationDelay: `${160 + index * 70}ms` }}
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="eyebrow">{item.id}</p>
                  <h3 className="mt-4 text-4xl">{item.title}</h3>
                </div>
                <div className="steel-index text-sm font-semibold uppercase tracking-[0.18em] text-[color:var(--muted)]">
                  manifesto
                </div>
              </div>
              <p className="mt-5 max-w-3xl text-base leading-8 text-[color:var(--muted)]">{item.text}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="grid gap-5 lg:grid-cols-[1.15fr_0.85fr]">
        <div className="panel-strong reveal delay-2 rounded-none px-6 py-8 sm:px-8">
          <p className="eyebrow">Commitment</p>
          <h2 className="mt-5 text-5xl sm:text-6xl">The community should be able to shape the ontology itself.</h2>
          <p className="mt-6 max-w-3xl text-lg leading-8 text-[color:var(--muted)]">
            That means adding tools, improving examples, refining parent ids, challenging categories, and releasing
            better checkpoints over time. The registry should behave like living infrastructure.
          </p>
        </div>
        <div className="signal-grid reveal delay-3 min-h-[22rem] rounded-none px-6 py-6">
          <div className="relative z-10 flex h-full flex-col justify-between">
            <div>
              <p className="eyebrow">Directive</p>
              <div className="mt-4 text-4xl font-semibold text-[color:var(--ink)]">
                Keep the tool layer public, inspectable, and reproducible.
              </div>
            </div>
            <div className="space-y-3 text-sm uppercase tracking-[0.16em] text-[color:var(--muted)]">
              <div>registry on github</div>
              <div>models on hugging face</div>
              <div>parent_id required for hierarchy</div>
            </div>
          </div>
        </div>
      </section>

      <section className="flex flex-wrap gap-3">
        <Link href="/submit" className="primary-button reveal delay-4 rounded-none px-5 py-3 text-sm font-semibold uppercase tracking-[0.16em]">
          Submit A Tool
        </Link>
        <Link href="/docs" className="ghost-button reveal delay-5 rounded-none px-5 py-3 text-sm font-semibold uppercase tracking-[0.16em]">
          Read Training Docs
        </Link>
      </section>
    </div>
  );
}
