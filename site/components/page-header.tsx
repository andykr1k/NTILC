import type { ReactNode } from "react";

type PageHeaderProps = {
  eyebrow: string;
  title: string;
  summary: string;
  meta?: ReactNode;
};

export function PageHeader({ eyebrow, title, summary, meta }: PageHeaderProps) {
  return (
    <section className="grid gap-6 lg:grid-cols-[1.2fr_0.95fr]">
      <div className="panel-strong rounded-[2rem] px-6 py-8 sm:px-8 sm:py-9">
        <p className="eyebrow">{eyebrow}</p>
        <h1 className="mt-4 max-w-4xl text-5xl leading-[0.98] font-semibold">{title}</h1>
        <p className="mt-5 max-w-3xl text-lg leading-8 text-[color:var(--muted)]">{summary}</p>
      </div>
      {meta ? <aside className="panel rounded-[2rem] px-6 py-8">{meta}</aside> : null}
    </section>
  );
}
