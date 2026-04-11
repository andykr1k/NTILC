import type { ReactNode } from "react";
import Link from "next/link";

const navigation = [
  { href: "/", label: "Mission" },
  { href: "/models", label: "Models" },
  { href: "/tools", label: "Tools" },
  { href: "/docs", label: "Docs" },
  { href: "/submit", label: "Submit" },
];

export function SiteShell({ children }: { children: ReactNode }) {
  return (
    <div className="relative min-h-screen overflow-x-hidden">
      <div className="pointer-events-none fixed inset-0 -z-10">
        <div className="absolute left-[-8rem] top-[5rem] h-64 w-64 rounded-full bg-orange-300/30 blur-3xl" />
        <div className="absolute right-[-6rem] top-[18rem] h-72 w-72 rounded-full bg-sky-300/25 blur-3xl" />
        <div className="absolute bottom-[-4rem] left-1/3 h-56 w-56 rounded-full bg-amber-200/25 blur-3xl" />
      </div>

      <header className="sticky top-0 z-20 border-b border-white/30 bg-white/45 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-4 sm:px-6 lg:px-8">
          <Link href="/" className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-[color:var(--accent-strong)] text-sm font-bold text-white">
              OE
            </div>
            <div>
              <div className="text-sm font-semibold uppercase tracking-[0.18em] text-[color:var(--accent-strong)]">
                Open Tool Embeddings
              </div>
              <div className="text-sm text-[color:var(--muted)]">registry-backed model distribution</div>
            </div>
          </Link>

          <nav className="flex max-w-[55vw] items-center gap-1 overflow-x-auto md:max-w-none md:gap-2">
            {navigation.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="shrink-0 rounded-full px-3 py-2 text-sm font-semibold text-[color:var(--ink)] hover:bg-white/60 md:px-4"
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 sm:py-10 lg:px-8">{children}</main>

      <footer className="border-t border-[color:var(--line)] py-8">
        <div className="mx-auto flex max-w-7xl flex-col gap-3 px-4 text-sm text-[color:var(--muted)] sm:px-6 lg:px-8 md:flex-row md:items-center md:justify-between">
          <p>Registry-first infrastructure for open-source tool embeddings.</p>
          <p>Update the registry, regenerate manifests, publish releases, and the site follows.</p>
        </div>
      </footer>
    </div>
  );
}
