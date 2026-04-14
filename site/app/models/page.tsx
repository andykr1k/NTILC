import type { Metadata } from "next";
import { ModelBrowser } from "@/components/model-browser";
import { PageHeader } from "@/components/page-header";
import { StructuredData } from "@/components/structured-data";
import { getModelsRegistry } from "@/lib/registry";
import { HUGGING_FACE_ORG_URL, buildCollectionPageSchema, buildPageMetadata } from "@/lib/seo";

export async function generateMetadata(): Promise<Metadata> {
  const modelRegistry = await getModelsRegistry();
  const publishedCount = modelRegistry.releases.filter((item) => item.status === "published").length;
  const readyCount = modelRegistry.releases.filter((item) => item.status === "ready").length;

  return buildPageMetadata({
    title: "Embedding Model Downloads",
    description: `Browse ${modelRegistry.releases.length} tracked tool embedding variants across normal and hierarchical architectures, including ${readyCount} ready artifacts and ${publishedCount} published releases.`,
    path: "/models",
    keywords: ["model downloads", "embedding checkpoints", "hierarchical model", "contrastive loss"],
  });
}

export default async function ModelsPage() {
  const modelRegistry = await getModelsRegistry();
  const readyCount = modelRegistry.releases.filter((item) => item.status === "ready").length;
  const publishedCount = modelRegistry.releases.filter((item) => item.status === "published").length;

  return (
    <div className="space-y-10 pb-16">
      <StructuredData
        data={buildCollectionPageSchema({
          name: "Open Tool Embeddings model downloads",
          description: "Filterable directory of normal and hierarchical tool embedding checkpoints.",
          path: "/models",
        })}
      />
      <PageHeader
        eyebrow="Models"
        title="Track packaged releases and live downloads."
        summary="The OpenToolEmbeddings Hugging Face organization is the canonical distribution surface. Releases here move from planned to ready once local artifacts and checksums exist, then to published once the checkpoint is live on Hugging Face."
        meta={
          <div className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-3">
              <div className="metric-chip rounded-none px-4 py-4">
                <div className="text-3xl font-semibold">{modelRegistry.releases.length}</div>
                <div className="mt-1 text-sm text-[color:var(--muted)]">tracked variants</div>
              </div>
              <div className="metric-chip rounded-none px-4 py-4">
                <div className="text-3xl font-semibold">{readyCount}</div>
                <div className="mt-1 text-sm text-[color:var(--muted)]">ready artifacts</div>
              </div>
              <div className="metric-chip rounded-none px-4 py-4">
                <div className="text-3xl font-semibold">{publishedCount}</div>
                <div className="mt-1 text-sm text-[color:var(--muted)]">published releases</div>
              </div>
            </div>
            <a
              href={HUGGING_FACE_ORG_URL}
              target="_blank"
              rel="noreferrer"
              className="primary-button inline-flex rounded-none px-4 py-2 text-sm font-semibold uppercase tracking-[0.14em]"
            >
              Open Hugging Face
            </a>
          </div>
        }
      />
      <ModelBrowser releases={modelRegistry.releases} />
    </div>
  );
}
