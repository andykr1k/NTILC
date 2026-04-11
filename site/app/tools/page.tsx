import { PageHeader } from "@/components/page-header";
import { ToolBrowser } from "@/components/tool-browser";
import { getToolRegistry } from "@/lib/registry";

export default async function ToolsPage() {
  const toolRegistry = await getToolRegistry();

  return (
    <div className="space-y-10 pb-16">
      <PageHeader
        eyebrow="Tools"
        title="Curate the central registry before you train."
        summary="A tool embedding model is only as good as the tool inventory behind it. This browser is backed by generated manifests from `registry/`, which means contributors can submit tools without touching frontend code."
        meta={
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="metric-chip rounded-[1.3rem] px-4 py-4">
              <div className="text-3xl font-semibold">{toolRegistry.tools.length}</div>
              <div className="mt-1 text-sm text-[color:var(--muted)]">tools tracked</div>
            </div>
            <div className="metric-chip rounded-[1.3rem] px-4 py-4">
              <div className="text-3xl font-semibold">{toolRegistry.categories.length}</div>
              <div className="mt-1 text-sm text-[color:var(--muted)]">parent categories</div>
            </div>
            <div className="metric-chip rounded-[1.3rem] px-4 py-4">
              <div className="text-3xl font-semibold">
                {toolRegistry.tools.reduce((total, tool) => total + tool.example_count, 0)}
              </div>
              <div className="mt-1 text-sm text-[color:var(--muted)]">example prompts</div>
            </div>
          </div>
        }
      />
      <ToolBrowser categories={toolRegistry.categories} tools={toolRegistry.tools} />
    </div>
  );
}
