"use client";

import { useDeferredValue, useState } from "react";
import type { ModelRelease } from "@/lib/types";

type ModelBrowserProps = {
  releases: ModelRelease[];
};

export function ModelBrowser({ releases }: ModelBrowserProps) {
  const [query, setQuery] = useState("");
  const [architecture, setArchitecture] = useState("all");
  const [loss, setLoss] = useState("all");
  const [status, setStatus] = useState("all");
  const deferredQuery = useDeferredValue(query);

  const normalizedQuery = deferredQuery.trim().toLowerCase();
  const filteredReleases = releases.filter((release) => {
    if (architecture !== "all" && release.architecture !== architecture) {
      return false;
    }
    if (loss !== "all" && release.loss !== loss) {
      return false;
    }
    if (status !== "all" && release.status !== status) {
      return false;
    }
    if (!normalizedQuery) {
      return true;
    }
    const searchBlob = [release.id, release.title, release.encoder, release.dataset_version, release.notes]
      .join(" ")
      .toLowerCase();
    return searchBlob.includes(normalizedQuery);
  });

  return (
    <div className="space-y-6">
      <section className="panel rounded-[1.8rem] px-5 py-5 sm:px-6">
        <div className="grid gap-4 xl:grid-cols-[1.5fr_0.9fr_0.8fr_0.8fr]">
          <label className="space-y-2">
            <span className="text-sm font-semibold uppercase tracking-[0.15em] text-[color:var(--muted)]">Search</span>
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="search variants, encoders, or dataset versions"
              className="w-full rounded-2xl border border-[color:var(--line)] bg-white/80 px-4 py-3 outline-none focus:border-[color:var(--accent)]"
            />
          </label>
          <label className="space-y-2">
            <span className="text-sm font-semibold uppercase tracking-[0.15em] text-[color:var(--muted)]">Architecture</span>
            <select
              value={architecture}
              onChange={(event) => setArchitecture(event.target.value)}
              className="w-full rounded-2xl border border-[color:var(--line)] bg-white/80 px-4 py-3 outline-none focus:border-[color:var(--accent)]"
            >
              <option value="all">All architectures</option>
              <option value="normal">Normal</option>
              <option value="hierarchical">Hierarchical</option>
            </select>
          </label>
          <label className="space-y-2">
            <span className="text-sm font-semibold uppercase tracking-[0.15em] text-[color:var(--muted)]">Loss</span>
            <select
              value={loss}
              onChange={(event) => setLoss(event.target.value)}
              className="w-full rounded-2xl border border-[color:var(--line)] bg-white/80 px-4 py-3 outline-none focus:border-[color:var(--accent)]"
            >
              <option value="all">All losses</option>
              <option value="prototype_ce">prototype_ce</option>
              <option value="contrastive">contrastive</option>
              <option value="circle">circle</option>
            </select>
          </label>
          <label className="space-y-2">
            <span className="text-sm font-semibold uppercase tracking-[0.15em] text-[color:var(--muted)]">Status</span>
            <select
              value={status}
              onChange={(event) => setStatus(event.target.value)}
              className="w-full rounded-2xl border border-[color:var(--line)] bg-white/80 px-4 py-3 outline-none focus:border-[color:var(--accent)]"
            >
              <option value="all">All statuses</option>
              <option value="published">Published</option>
              <option value="planned">Planned</option>
              <option value="deprecated">Deprecated</option>
            </select>
          </label>
        </div>
        <p className="mt-4 text-sm text-[color:var(--muted)]">
          Showing {filteredReleases.length} of {releases.length} releases.
        </p>
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        {filteredReleases.map((release) => {
          const metricEntries = Object.entries(release.metrics);
          return (
            <article key={release.id} className="panel rounded-[1.7rem] px-5 py-6">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="eyebrow">{release.architecture}</p>
                  <h2 className="mt-3 text-2xl font-semibold">{release.title}</h2>
                </div>
                <span className="metric-chip rounded-full px-3 py-1 text-xs uppercase tracking-[0.16em]">
                  {release.status}
                </span>
              </div>

              <div className="mt-4 flex flex-wrap gap-2">
                {[release.loss, `${release.embedding_dim}d`, release.dataset_version].map((item) => (
                  <span key={item} className="metric-chip rounded-full px-3 py-1 text-xs uppercase tracking-[0.12em]">
                    {item}
                  </span>
                ))}
              </div>

              <p className="mt-4 text-sm leading-7 text-[color:var(--muted)]">{release.notes || "Release notes pending."}</p>

              <div className="mt-6 grid gap-3 sm:grid-cols-2">
                <div className="metric-chip rounded-[1.2rem] px-4 py-3">
                  <div className="text-xs uppercase tracking-[0.14em] text-[color:var(--muted)]">Encoder</div>
                  <div className="mt-1 text-sm font-semibold">{release.encoder}</div>
                </div>
                <div className="metric-chip rounded-[1.2rem] px-4 py-3">
                  <div className="text-xs uppercase tracking-[0.14em] text-[color:var(--muted)]">Checksum</div>
                  <div className="mt-1 text-sm font-semibold">{release.sha256 || "pending"}</div>
                </div>
              </div>

              <div className="mt-6">
                <div className="text-xs font-semibold uppercase tracking-[0.16em] text-[color:var(--muted)]">Metrics</div>
                {metricEntries.length ? (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {metricEntries.map(([key, value]) => (
                      <span key={key} className="metric-chip rounded-full px-3 py-1 text-xs uppercase tracking-[0.12em]">
                        {key}: {String(value)}
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="mt-3 text-sm text-[color:var(--muted)]">Metrics will appear here once the release is published.</p>
                )}
              </div>

              <div className="mt-6 flex flex-wrap items-center gap-4 text-sm">
                {release.download_url ? (
                  <a
                    href={release.download_url}
                    target="_blank"
                    rel="noreferrer"
                    className="rounded-full bg-[color:var(--accent-strong)] px-4 py-2 font-semibold text-white shadow-lg shadow-orange-900/15"
                  >
                    Download checkpoint
                  </a>
                ) : (
                  <span className="rounded-full border border-dashed border-[color:var(--line)] px-4 py-2 font-semibold text-[color:var(--muted)]">
                    Awaiting artifact
                  </span>
                )}
                {release.repository_url ? (
                  <a
                    href={release.repository_url}
                    target="_blank"
                    rel="noreferrer"
                    className="font-semibold text-[color:var(--accent-strong)]"
                  >
                    Model repository
                  </a>
                ) : null}
              </div>
            </article>
          );
        })}
      </section>

      {!filteredReleases.length ? (
        <section className="panel rounded-[1.8rem] px-6 py-8 text-center text-[color:var(--muted)]">
          No model releases matched the current filters.
        </section>
      ) : null}
    </div>
  );
}
