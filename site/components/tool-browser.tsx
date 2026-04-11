"use client";

import { useDeferredValue, useState } from "react";
import type { Category, Tool } from "@/lib/types";

type ToolBrowserProps = {
  categories: Category[];
  tools: Tool[];
};

export function ToolBrowser({ categories, tools }: ToolBrowserProps) {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("all");
  const [interfaceType, setInterfaceType] = useState("all");
  const deferredQuery = useDeferredValue(query);

  const normalizedQuery = deferredQuery.trim().toLowerCase();
  const interfaceTypes = Array.from(new Set(tools.map((tool) => tool.interface_type))).sort();

  const filteredTools = tools.filter((tool) => {
    if (category !== "all" && tool.parent_category !== category) {
      return false;
    }
    if (interfaceType !== "all" && tool.interface_type !== interfaceType) {
      return false;
    }
    if (!normalizedQuery) {
      return true;
    }

    const searchBlob = [
      tool.id,
      tool.display_name,
      tool.description,
      tool.license,
      tool.parent_category,
      ...tool.tags,
    ]
      .join(" ")
      .toLowerCase();

    return searchBlob.includes(normalizedQuery);
  });

  return (
    <div className="space-y-6">
      <section className="panel rounded-[1.8rem] px-5 py-5 sm:px-6">
        <div className="grid gap-4 lg:grid-cols-[1.5fr_0.8fr_0.8fr]">
          <label className="space-y-2">
            <span className="text-sm font-semibold uppercase tracking-[0.15em] text-[color:var(--muted)]">Search</span>
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="search tools, tags, or categories"
              className="w-full rounded-2xl border border-[color:var(--line)] bg-white/80 px-4 py-3 outline-none focus:border-[color:var(--accent)]"
            />
          </label>
          <label className="space-y-2">
            <span className="text-sm font-semibold uppercase tracking-[0.15em] text-[color:var(--muted)]">Category</span>
            <select
              value={category}
              onChange={(event) => setCategory(event.target.value)}
              className="w-full rounded-2xl border border-[color:var(--line)] bg-white/80 px-4 py-3 outline-none focus:border-[color:var(--accent)]"
            >
              <option value="all">All categories</option>
              {categories.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name}
                </option>
              ))}
            </select>
          </label>
          <label className="space-y-2">
            <span className="text-sm font-semibold uppercase tracking-[0.15em] text-[color:var(--muted)]">Interface</span>
            <select
              value={interfaceType}
              onChange={(event) => setInterfaceType(event.target.value)}
              className="w-full rounded-2xl border border-[color:var(--line)] bg-white/80 px-4 py-3 outline-none focus:border-[color:var(--accent)]"
            >
              <option value="all">All interfaces</option>
              {interfaceTypes.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
        </div>
        <p className="mt-4 text-sm text-[color:var(--muted)]">
          Showing {filteredTools.length} of {tools.length} tools.
        </p>
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        {filteredTools.map((tool) => (
          <article key={tool.id} className="panel rounded-[1.7rem] px-5 py-6">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <p className="eyebrow">{tool.parent_category}</p>
                <h2 className="mt-3 text-2xl font-semibold">{tool.display_name}</h2>
              </div>
              <span className="metric-chip rounded-full px-3 py-1 text-xs uppercase tracking-[0.16em]">
                {tool.interface_type}
              </span>
            </div>
            <p className="mt-4 text-sm leading-7 text-[color:var(--muted)]">{tool.description}</p>
            <div className="mt-5 flex flex-wrap gap-2">
              {tool.tags.map((tag) => (
                <span key={tag} className="metric-chip rounded-full px-3 py-1 text-xs uppercase tracking-[0.12em]">
                  {tag}
                </span>
              ))}
            </div>
            <div className="mt-6 grid gap-3 sm:grid-cols-3">
              <div className="metric-chip rounded-[1.2rem] px-4 py-3">
                <div className="text-xs uppercase tracking-[0.14em] text-[color:var(--muted)]">Examples</div>
                <div className="mt-1 text-lg font-semibold">{tool.example_count}</div>
              </div>
              <div className="metric-chip rounded-[1.2rem] px-4 py-3">
                <div className="text-xs uppercase tracking-[0.14em] text-[color:var(--muted)]">Parameters</div>
                <div className="mt-1 text-lg font-semibold">{Object.keys(tool.parameters.properties).length}</div>
              </div>
              <div className="metric-chip rounded-[1.2rem] px-4 py-3">
                <div className="text-xs uppercase tracking-[0.14em] text-[color:var(--muted)]">License</div>
                <div className="mt-1 text-sm font-semibold">{tool.license}</div>
              </div>
            </div>
            <div className="mt-6 flex flex-wrap items-center gap-4 text-sm">
              <a
                href={tool.source_repo}
                target="_blank"
                rel="noreferrer"
                className="font-semibold text-[color:var(--accent-strong)]"
              >
                Source repository
              </a>
              <span className="text-[color:var(--muted)]">{tool.id}</span>
              {tool.has_tests ? <span className="text-[color:var(--muted)]">tests attached</span> : null}
            </div>
          </article>
        ))}
      </section>

      {!filteredTools.length ? (
        <section className="panel rounded-[1.8rem] px-6 py-8 text-center text-[color:var(--muted)]">
          No tools matched the current filters.
        </section>
      ) : null}
    </div>
  );
}
