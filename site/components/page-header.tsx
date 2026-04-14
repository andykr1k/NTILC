import type { ReactNode } from "react";

type PageHeaderProps = {
  eyebrow: string;
  title: string;
  summary: string;
  meta?: ReactNode;
};

export function PageHeader({ eyebrow, title, summary, meta }: PageHeaderProps) {
  return (
    <section className="grid gap-5 xl:grid-cols-[1.3fr_0.7fr]">
      <div className="panel-strong reveal rounded-none px-6 py-8 sm:px-8 sm:py-10">
        <div className="hero-mark absolute right-[-1.5rem] top-[-1rem] text-[7rem] sm:text-[10rem]">{eyebrow}</div>
        <div className="relative z-10 max-w-5xl">
          <p className="eyebrow">{eyebrow}</p>
          <h1 className="mt-5 text-6xl sm:text-7xl lg:text-8xl">{title}</h1>
          <p className="mt-6 max-w-3xl text-lg leading-8 text-[color:var(--muted)]">{summary}</p>
        </div>
      </div>
      {meta ? (
        <aside className="panel steel-card reveal delay-1 rounded-none px-6 py-8">
          {meta}
        </aside>
      ) : null}
    </section>
  );
}
