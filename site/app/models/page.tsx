import { ModelBrowser } from "@/components/model-browser";
import { PageHeader } from "@/components/page-header";
import { getModelsRegistry } from "@/lib/registry";

export default async function ModelsPage() {
  const modelRegistry = await getModelsRegistry();

  return (
    <div className="space-y-10 pb-16">
      <PageHeader
        eyebrow="Models"
        title="Download the latest embedding variants."
        summary="Every model release is keyed by architecture, loss, encoder, dataset version, and publication status so users can compare options without guessing from directory names."
        meta={
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="metric-chip rounded-[1.3rem] px-4 py-4">
              <div className="text-3xl font-semibold">{modelRegistry.releases.length}</div>
              <div className="mt-1 text-sm text-[color:var(--muted)]">tracked variants</div>
            </div>
            <div className="metric-chip rounded-[1.3rem] px-4 py-4">
              <div className="text-3xl font-semibold">
                {modelRegistry.releases.filter((item) => item.architecture === "normal").length}
              </div>
              <div className="mt-1 text-sm text-[color:var(--muted)]">normal releases</div>
            </div>
            <div className="metric-chip rounded-[1.3rem] px-4 py-4">
              <div className="text-3xl font-semibold">
                {modelRegistry.releases.filter((item) => item.architecture === "hierarchical").length}
              </div>
              <div className="mt-1 text-sm text-[color:var(--muted)]">hierarchical releases</div>
            </div>
          </div>
        }
      />
      <ModelBrowser releases={modelRegistry.releases} />
    </div>
  );
}
