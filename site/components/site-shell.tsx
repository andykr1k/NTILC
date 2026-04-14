import type { ReactNode } from "react";
import Link from "next/link";
import { GITHUB_REPO_URL, HUGGING_FACE_ORG_URL } from "@/lib/seo";

const navigation = [
  { href: "/", label: "Home" },
  { href: "/mission", label: "Mission" },
  { href: "/models", label: "Models" },
  { href: "/tools", label: "Registry" },
  { href: "/docs", label: "Training" },
  { href: "/submit", label: "Submit" },
];

export function SiteShell({ children }: { children: ReactNode }) {
  return (
    <div className="relative min-h-screen overflow-x-hidden">
      <header className="sticky top-0 z-30 border-b border-[color:var(--line)] bg-[rgba(6,11,18,0.82)] backdrop-blur-2xl">
        <div className="mx-auto max-w-7xl px-4 py-3 sm:px-6 lg:px-8">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <Link href="/" className="flex min-w-0 items-center gap-3">
              <div className="flex h-11 w-11 shrink-0 items-center justify-center border border-[rgba(130,182,255,0.24)] bg-[rgba(130,182,255,0.08)] text-sm font-bold text-[color:var(--dominant)]">
                OE
              </div>
              <div className="min-w-0 text-lg font-semibold uppercase tracking-[0.1em] text-[color:var(--ink)] sm:text-xl">
                Open Tool Embeddings
              </div>
            </Link>

            <div className="flex flex-wrap items-center gap-2 lg:justify-end">
              <details className="menu-shell relative">
                <summary className="ghost-button flex cursor-pointer list-none items-center gap-2 rounded-none px-4 py-2.5 text-xs font-semibold uppercase tracking-[0.14em]">
                  Menu
                  <span className="menu-caret text-[10px]">+</span>
                </summary>

                <div className="menu-panel absolute left-0 top-[calc(100%+0.55rem)] z-40 min-w-56 border border-[color:var(--line-strong)] bg-[rgba(10,18,29,0.96)] p-2 shadow-[0_28px_60px_rgba(0,0,0,0.45)] backdrop-blur-2xl">
                  <div className="mb-1 px-2 py-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-[color:var(--muted)]">
                    Sections
                  </div>
                  <nav className="flex flex-col">
                    {navigation.map((item) => (
                      <Link
                        key={item.href}
                        href={item.href}
                        className="menu-link px-3 py-3 text-xs font-semibold uppercase tracking-[0.16em] text-[color:var(--ink)]"
                      >
                        {item.label}
                      </Link>
                    ))}
                  </nav>
                </div>
              </details>

              <div className="flex flex-wrap gap-2">
                <a
                  href={GITHUB_REPO_URL}
                  target="_blank"
                  rel="noreferrer"
                  className="ghost-button rounded-none px-4 py-2.5 text-xs font-semibold uppercase tracking-[0.14em]"
                >
                  GitHub
                </a>
                <a
                  href={HUGGING_FACE_ORG_URL}
                  target="_blank"
                  rel="noreferrer"
                  className="primary-button rounded-none px-4 py-2.5 text-xs font-semibold uppercase tracking-[0.14em]"
                >
                  Hugging Face
                </a>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-10 sm:px-6 sm:py-12 lg:px-8">{children}</main>

      <footer className="border-t border-[color:var(--line)] py-12">
        <div className="mx-auto grid max-w-7xl gap-8 px-4 sm:px-6 lg:grid-cols-[1.2fr_0.8fr_0.8fr] lg:px-8">
          <div className="panel px-6 py-6">
            <div className="eyebrow">Open Tool Embeddings</div>
            <div className="mt-4 max-w-xl text-4xl font-semibold text-[color:var(--ink)]">An industrial-grade public layer for tool understanding.</div>
            <p className="mt-4 max-w-xl text-base leading-8 text-[color:var(--muted)]">
              The registry stays open, the hierarchy stays explicit, and ready checkpoints move through the
              OpenToolEmbeddings Hugging Face organization.
            </p>
          </div>

          <div className="panel px-6 py-6">
            <div className="eyebrow">Community</div>
            <div className="mt-4 space-y-3 text-sm uppercase tracking-[0.14em] text-[color:var(--ink)]">
              <a href={GITHUB_REPO_URL} target="_blank" rel="noreferrer" className="block hover:text-[color:var(--accent)]">
                GitHub repository
              </a>
              <Link href="/mission" className="block hover:text-[color:var(--accent)]">
                Mission
              </Link>
              <Link href="/submit" className="block hover:text-[color:var(--accent)]">
                Submit a tool
              </Link>
            </div>
          </div>

          <div className="panel px-6 py-6">
            <div className="eyebrow">Distribution</div>
            <div className="mt-4 space-y-3 text-sm uppercase tracking-[0.14em] text-[color:var(--ink)]">
              <a href={HUGGING_FACE_ORG_URL} target="_blank" rel="noreferrer" className="block hover:text-[color:var(--accent)]">
                Hugging Face organization
              </a>
              <Link href="/models" className="block hover:text-[color:var(--accent)]">
                Model variants
              </Link>
              <Link href="/docs" className="block hover:text-[color:var(--accent)]">
                Training docs
              </Link>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
