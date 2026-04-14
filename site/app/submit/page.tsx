import type { Metadata } from "next";
import { PageHeader } from "@/components/page-header";
import { GITHUB_REPO_URL, buildPageMetadata } from "@/lib/seo";

const toolManifestExample = `id: ripgrep.search
display_name: Ripgrep Search
description: Search file contents recursively with fast regex-aware text matching.
interface_type: cli
source_repo: https://github.com/BurntSushi/ripgrep
license: MIT OR Unlicense
maintainers:
  - your-handle
parent_id: filesystem.search
tags:
  - search
  - grep
parameters:
  type: object
  properties:
    pattern:
      type: string
  required:
    - pattern`;

const examplesExample = `{"query":"Find every TODO under src","split":"train","language":"en"}
{"query":"Search recursively for CUDA references in this repo","split":"train","language":"en"}
{"query":"Look for postgres mentions in markdown files","split":"train","language":"en"}`;

const checklist = [
  "Public upstream repository and a clear license.",
  "A category that already exists or a strong case for a new one.",
  "A valid parent_id for the hierarchical model.",
  "At least three distinct natural-language examples.",
  "Parameters that are practical, minimal, and not overfit to one benchmark.",
];

export const metadata: Metadata = buildPageMetadata({
  title: "Submit a Tool",
  description:
    "Learn how to propose or contribute a new open-source tool to the central registry with manifests, examples, and reviewable metadata.",
  path: "/submit",
  keywords: ["submit tool", "open source contribution", "tool manifest", "registry contribution"],
});

export default function SubmitPage() {
  return (
    <div className="space-y-10 pb-16">
      <PageHeader
        eyebrow="Submit"
        title="Accept new tools through review, not through direct uploads."
        summary="The safest path is still GitHub: open an issue, add `tool.yaml` and `examples.jsonl` in a PR, include a valid `parent_id` for the hierarchical model, then run the registry builder before merge."
        meta={
          <a
            href={GITHUB_REPO_URL}
            target="_blank"
            rel="noreferrer"
            className="primary-button inline-flex rounded-none px-4 py-2 text-sm font-semibold uppercase tracking-[0.14em]"
          >
            Open GitHub
          </a>
        }
      />

      <section className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
        <article className="panel steel-card reveal delay-1 rounded-none px-6 py-7">
          <p className="eyebrow">Submission Checklist</p>
          <div className="mt-5 space-y-4">
            {checklist.map((item) => (
              <div key={item} className="metric-chip rounded-none px-4 py-4 text-sm leading-7">
                {item}
              </div>
            ))}
          </div>
        </article>

        <article className="panel steel-card reveal delay-2 rounded-none px-6 py-7">
          <p className="eyebrow">Contribution Flow</p>
          <ol className="mt-5 space-y-4 text-sm leading-7">
            <li>1. Open the tool submission issue template and describe the open-source project.</li>
            <li>2. Add `registry/tools/&lt;tool-id&gt;/tool.yaml` and `examples.jsonl` in a pull request.</li>
            <li>3. Set a valid `parent_id` so the hierarchical model has the right parent label.</li>
            <li>4. Run `python3 scripts/build_registry.py` and include the generated diff.</li>
            <li>5. After merge, the next training snapshot and Hugging Face release can include the tool.</li>
          </ol>
        </article>
      </section>

      <section className="grid gap-6 xl:grid-cols-2">
        <article className="panel steel-card reveal delay-3 rounded-none px-6 py-7">
          <p className="eyebrow">tool.yaml</p>
          <pre className="mt-5 overflow-x-auto border border-[color:var(--line)] bg-[rgba(5,10,16,0.95)] px-4 py-4 text-sm text-stone-100">
            <code>{toolManifestExample}</code>
          </pre>
        </article>
        <article className="panel steel-card reveal delay-4 rounded-none px-6 py-7">
          <p className="eyebrow">examples.jsonl</p>
          <pre className="mt-5 overflow-x-auto border border-[color:var(--line)] bg-[rgba(5,10,16,0.95)] px-4 py-4 text-sm text-stone-100">
            <code>{examplesExample}</code>
          </pre>
        </article>
      </section>
    </div>
  );
}
