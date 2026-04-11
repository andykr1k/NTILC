import type { Metadata } from "next";
import { PageHeader } from "@/components/page-header";
import { StructuredData } from "@/components/structured-data";
import { ToolBrowser } from "@/components/tool-browser";
import { getToolRegistry } from "@/lib/registry";
import { GITHUB_REPO_URL, buildCollectionPageSchema, buildPageMetadata } from "@/lib/seo";

export async function generateMetadata(): Promise<Metadata> {
  const toolRegistry = await getToolRegistry();

  return buildPageMetadata({
    title: "Open-Source Tool Registry",
    description: `Explore ${toolRegistry.tools.length} curated open-source tools across ${toolRegistry.categories.length} categories for tool embedding training and retrieval.`,
    path: "/tools",
    keywords: ["tool registry", "open source tools", "tool metadata", "tool schemas"],
  });
}

export default async function ToolsPage() {
  const toolRegistry = await getToolRegistry();
  const promptCount = toolRegistry.tools.reduce((total, tool) => total + tool.example_count, 0);

  return (
    <div className="space-y-10 pb-16">
      <StructuredData
        data={buildCollectionPageSchema({
          name: "Open Tool Embeddings registry",
          description: "Curated source-of-truth registry for tools, categories, and training prompts.",
          path: "/tools",
        })}
      />
      <PageHeader
        eyebrow="Tools"
        title="The central registry for community-submitted open-source tools."
        summary="This page is generated from `registry/` and starts empty on purpose. The inventory should grow through public submissions, explicit parent ids for the hierarchical model, and reviewable example prompts."
        meta={
          <div className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-3">
              <div className="metric-chip rounded-none px-4 py-4">
                <div className="text-3xl font-semibold">{toolRegistry.tools.length}</div>
                <div className="mt-1 text-sm text-[color:var(--muted)]">tools tracked</div>
              </div>
              <div className="metric-chip rounded-none px-4 py-4">
                <div className="text-3xl font-semibold">{toolRegistry.categories.length}</div>
                <div className="mt-1 text-sm text-[color:var(--muted)]">parent ids available</div>
              </div>
              <div className="metric-chip rounded-none px-4 py-4">
                <div className="text-3xl font-semibold">{promptCount}</div>
                <div className="mt-1 text-sm text-[color:var(--muted)]">example prompts</div>
              </div>
            </div>
            <a
              href={GITHUB_REPO_URL}
              target="_blank"
              rel="noreferrer"
              className="ghost-button inline-flex rounded-none px-4 py-2 text-sm font-semibold uppercase tracking-[0.14em]"
            >
              Open the GitHub registry
            </a>
          </div>
        }
      />
      <ToolBrowser categories={toolRegistry.categories} tools={toolRegistry.tools} />
    </div>
  );
}
