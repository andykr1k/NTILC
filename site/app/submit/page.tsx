import { PageHeader } from "@/components/page-header";

const toolManifestExample = `id: ripgrep.search
display_name: Ripgrep Search
description: Search file contents recursively with fast regex-aware text matching.
interface_type: cli
source_repo: https://github.com/BurntSushi/ripgrep
license: MIT OR Unlicense
maintainers:
  - your-handle
parent_category: filesystem.search
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
  "At least three distinct natural-language examples.",
  "Parameters that are practical, minimal, and not overfit to one benchmark.",
];

export default function SubmitPage() {
  return (
    <div className="space-y-10 pb-16">
      <PageHeader
        eyebrow="Submit"
        title="Accept new tools through review, not through direct uploads."
        summary="The safest contribution path is a GitHub issue for proposals and a PR for the actual manifest. That preserves provenance, makes moderation explicit, and keeps bad data out of training snapshots."
      />

      <section className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
        <article className="panel rounded-[1.8rem] px-6 py-7">
          <p className="eyebrow">Submission Checklist</p>
          <div className="mt-5 space-y-4">
            {checklist.map((item) => (
              <div key={item} className="metric-chip rounded-[1.2rem] px-4 py-4 text-sm leading-7">
                {item}
              </div>
            ))}
          </div>
        </article>

        <article className="panel rounded-[1.8rem] px-6 py-7">
          <p className="eyebrow">Contribution Flow</p>
          <ol className="mt-5 space-y-4 text-sm leading-7">
            <li>1. Open the tool submission issue template and describe the open-source project.</li>
            <li>2. Add `registry/tools/&lt;tool-id&gt;/tool.yaml` and `examples.jsonl` in a pull request.</li>
            <li>3. Run `python3 scripts/build_registry.py` and include the generated diff.</li>
            <li>4. After merge, include the tool in the next registry snapshot and training release.</li>
          </ol>
        </article>
      </section>

      <section className="grid gap-6 xl:grid-cols-2">
        <article className="panel rounded-[1.8rem] px-6 py-7">
          <p className="eyebrow">tool.yaml</p>
          <pre className="mt-5 overflow-x-auto rounded-[1.4rem] bg-stone-950 px-4 py-4 text-sm text-stone-100">
            <code>{toolManifestExample}</code>
          </pre>
        </article>
        <article className="panel rounded-[1.8rem] px-6 py-7">
          <p className="eyebrow">examples.jsonl</p>
          <pre className="mt-5 overflow-x-auto rounded-[1.4rem] bg-stone-950 px-4 py-4 text-sm text-stone-100">
            <code>{examplesExample}</code>
          </pre>
        </article>
      </section>
    </div>
  );
}
