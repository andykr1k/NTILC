from __future__ import annotations

import argparse
import shutil
import subprocess
from html import escape
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
except ModuleNotFoundError:
    plt = None
    FuncFormatter = None

try:
    if plt is not None:
        import seaborn as sns
    else:
        sns = None
except ModuleNotFoundError:
    sns = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "analysis" / "context_length_by_tools.png"

SYSTEM_PROMPT_AND_QUERY_TOKENS = 1_000
AVERAGE_CONTEXT_WINDOW_TOKENS = 1_000_000
Y_AXIS_MAX_TOKENS = 1_200_000
ICT_TOKENS_PER_TOOL = 120
NTILC_TOKENS_PER_TOOL = 1
SESSION_TOOL_RESULT_TOKEN_SIZES = (
    (200, "Small result: 200 tokens/call", "#009e73"),
    (1_000, "Medium result: 1,000 tokens/call", "#56b4e9"),
    (5_000, "Large result: 5,000 tokens/call", "#cc79a7"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot context length growth as the number of tools increases."
    )
    parser.add_argument(
        "--max-tools",
        type=int,
        default=15_000,
        help="Maximum number of tools shown on the x axis.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=1_000,
        help="Number of x-axis samples used to draw each method line.",
    )
    parser.add_argument(
        "--max-session-tool-calls",
        type=int,
        default=1_000,
        help="Maximum number of tool calls shown for the session-growth panel.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the plot image will be saved.",
    )
    return parser.parse_args()


def sample_counts(max_count: int, num_points: int) -> list[float]:
    step = max_count / (num_points - 1)
    return [index * step for index in range(num_points)]


def context_length(counts: list[float], tokens_per_unit: int) -> list[float]:
    return [SYSTEM_PROMPT_AND_QUERY_TOKENS + tokens_per_unit * count for count in counts]


def threshold_crossing_count(tokens_per_unit: int) -> float:
    return (AVERAGE_CONTEXT_WINDOW_TOKENS - SYSTEM_PROMPT_AND_QUERY_TOKENS) / tokens_per_unit


def set_plot_style() -> None:
    if plt is None:
        return
    if sns:
        sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    else:
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            plt.style.use("default")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def add_average_context_window(ax) -> None:
    ax.axhspan(
        AVERAGE_CONTEXT_WINDOW_TOKENS,
        Y_AXIS_MAX_TOKENS,
        color="#d62728",
        alpha=0.08,
        zorder=0,
    )
    ax.axhline(
        AVERAGE_CONTEXT_WINDOW_TOKENS,
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label="_nolegend_",
    )
    ax.text(
        0.98,
        AVERAGE_CONTEXT_WINDOW_TOKENS * 1.01,
        "Average context window",
        color="#d62728",
        ha="right",
        va="bottom",
        fontsize=11,
        transform=ax.get_yaxis_transform(),
    )


def format_axis(ax, x_max: int) -> None:
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, Y_AXIS_MAX_TOKENS)
    ax.grid(True, axis="both", alpha=0.25)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))


def plot_catalog_context(ax, max_tools: int, num_points: int) -> None:
    tool_counts = sample_counts(max_tools, num_points)
    ict_tokens = context_length(tool_counts, ICT_TOKENS_PER_TOOL)
    ntilc_tokens = context_length(tool_counts, NTILC_TOKENS_PER_TOOL)
    ict_crossing = threshold_crossing_count(ICT_TOKENS_PER_TOOL)
    ntilc_crossing = threshold_crossing_count(NTILC_TOKENS_PER_TOOL)

    add_average_context_window(ax)
    ax.plot(
        tool_counts,
        ict_tokens,
        color="#e6b800",
        linewidth=2.8,
        label=f"ICT: {ICT_TOKENS_PER_TOOL} tokens/tool",
    )
    ax.plot(
        tool_counts,
        ntilc_tokens,
        color="#0072b2",
        linewidth=2.8,
        label=f"NTILC: {NTILC_TOKENS_PER_TOOL} token/tool",
    )

    if ict_crossing <= max_tools:
        ax.axvline(
            ict_crossing,
            color="#b38f00",
            linestyle=":",
            linewidth=2,
        )
        ax.annotate(
            f"ICT hits 1M at\n~{ict_crossing:,.0f} tools",
            xy=(ict_crossing, AVERAGE_CONTEXT_WINDOW_TOKENS),
            xytext=(max_tools * 0.43, AVERAGE_CONTEXT_WINDOW_TOKENS * 0.54),
            arrowprops={"arrowstyle": "->", "color": "#8a6d00", "linewidth": 1.5},
            color="#6f5700",
            fontsize=11,
        )

    ax.text(
        0.04,
        0.08,
        f"NTILC reaches 1M at ~{ntilc_crossing:,.0f} tools",
        transform=ax.transAxes,
        fontsize=10,
        color="#00507d",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d9d9d9", "alpha": 0.9},
    )
    ax.set_title("Available Tools Consume Context")
    ax.set_xlabel("Number of available tools")
    ax.set_ylabel("Context length (tokens)")
    format_axis(ax, max_tools)
    ax.legend(loc="upper left", frameon=True, fontsize=10)


def plot_session_context(ax, max_tool_calls: int, num_points: int) -> None:
    tool_calls = sample_counts(max_tool_calls, num_points)

    add_average_context_window(ax)
    for tokens_per_call, label, color in SESSION_TOOL_RESULT_TOKEN_SIZES:
        ax.plot(
            tool_calls,
            context_length(tool_calls, tokens_per_call),
            color=color,
            linewidth=2.8,
            label=label,
        )

    large_result_crossing = threshold_crossing_count(5_000)
    medium_result_crossing = threshold_crossing_count(1_000)
    if large_result_crossing <= max_tool_calls:
        ax.axvline(
            large_result_crossing,
            color="#cc79a7",
            linestyle=":",
            linewidth=2,
        )
        ax.annotate(
            f"5,000-token results\nhit 1M at ~{large_result_crossing:,.0f} calls",
            xy=(large_result_crossing, AVERAGE_CONTEXT_WINDOW_TOKENS),
            xytext=(max_tool_calls * 0.28, AVERAGE_CONTEXT_WINDOW_TOKENS * 0.48),
            arrowprops={"arrowstyle": "->", "color": "#9b4d7b", "linewidth": 1.5},
            color="#7a315f",
            fontsize=11,
        )

    ax.text(
        0.04,
        0.08,
        f"1,000-token results reach 1M at ~{medium_result_crossing:,.0f} calls",
        transform=ax.transAxes,
        fontsize=10,
        color="#27627d",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d9d9d9", "alpha": 0.9},
    )
    ax.set_title("Tool Use During a Session Also Grows Context")
    ax.set_xlabel("Tool calls used in session")
    format_axis(ax, max_tool_calls)
    ax.legend(loc="upper left", frameon=True, fontsize=10)


def render_matplotlib(args: argparse.Namespace) -> Path:
    set_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    fig.suptitle(
        "Context Window Pressure Grows with Tool Scale and Tool Use",
        fontsize=18,
        fontweight="bold",
    )
    plot_catalog_context(axes[0], args.max_tools, args.num_points)
    plot_session_context(axes[1], args.max_session_tool_calls, args.num_points)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, bbox_inches="tight")
    plt.close(fig)
    return args.output_path


def svg_x(value: float, x_max: int, left: float, width: float) -> float:
    return left + (value / x_max) * width


def svg_y(value: float, top: float, height: float) -> float:
    return top + height - (value / Y_AXIS_MAX_TOKENS) * height


def svg_polyline(
    counts: list[float],
    values: list[float],
    *,
    x_max: int,
    left: float,
    top: float,
    width: float,
    height: float,
    color: str,
) -> str:
    points = " ".join(
        f"{svg_x(count, x_max, left, width):.1f},{svg_y(value, top, height):.1f}"
        for count, value in zip(counts, values)
    )
    return (
        f'<polyline points="{points}" fill="none" stroke="{color}" '
        'stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />'
    )


def clipped_context_series(counts: list[float], tokens_per_unit: int) -> tuple[list[float], list[float]]:
    clipped_counts: list[float] = []
    clipped_values: list[float] = []
    max_visible_count = (Y_AXIS_MAX_TOKENS - SYSTEM_PROMPT_AND_QUERY_TOKENS) / tokens_per_unit
    for count in counts:
        value = SYSTEM_PROMPT_AND_QUERY_TOKENS + tokens_per_unit * count
        if value <= Y_AXIS_MAX_TOKENS:
            clipped_counts.append(count)
            clipped_values.append(value)
            continue
        if not clipped_counts or clipped_counts[-1] < max_visible_count:
            clipped_counts.append(max_visible_count)
            clipped_values.append(Y_AXIS_MAX_TOKENS)
        break
    return clipped_counts, clipped_values


def svg_text(
    text: str,
    *,
    x: float,
    y: float,
    size: int = 15,
    fill: str = "#222222",
    anchor: str = "start",
    weight: str = "400",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        'font-family="DejaVu Sans, Arial, Helvetica, sans-serif" font-style="normal" '
        f'font-weight="{weight}" text-anchor="{anchor}">{escape(text)}</text>'
    )


def svg_tick_values(max_value: int) -> list[int]:
    return [round(max_value * index / 5) for index in range(6)]


def svg_clip_path(
    *,
    panel_id: str,
    left: float,
    top: float,
    width: float,
    height: float,
) -> str:
    return (
        f'<clipPath id="{panel_id}-clip"><rect x="{left:.1f}" y="{top:.1f}" '
        f'width="{width:.1f}" height="{height:.1f}" /></clipPath>'
    )


def svg_panel_base(
    *,
    title: str,
    x_label: str,
    x_max: int,
    left: float,
    top: float,
    width: float,
    height: float,
    y_label: str | None = None,
) -> list[str]:
    elements: list[str] = []
    bottom = top + height
    avg_y = svg_y(AVERAGE_CONTEXT_WINDOW_TOKENS, top, height)

    elements.append(
        f'<rect x="{left:.1f}" y="{top:.1f}" width="{width:.1f}" '
        f'height="{avg_y - top:.1f}" fill="#fde8e8" />'
    )

    for tick in range(0, Y_AXIS_MAX_TOKENS + 1, 200_000):
        y = svg_y(tick, top, height)
        elements.append(
            f'<line x1="{left:.1f}" x2="{left + width:.1f}" y1="{y:.1f}" '
            f'y2="{y:.1f}" stroke="#d0d0d0" stroke-width="1" opacity="0.65" />'
        )
        elements.append(
            svg_text(f"{tick:,.0f}", x=left - 12, y=y + 5, size=12, fill="#555555", anchor="end")
        )

    for tick in svg_tick_values(x_max):
        x = svg_x(tick, x_max, left, width)
        elements.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top:.1f}" '
            f'y2="{bottom:.1f}" stroke="#d0d0d0" stroke-width="1" opacity="0.45" />'
        )
        elements.append(
            svg_text(f"{tick:,.0f}", x=x, y=bottom + 28, size=12, fill="#555555", anchor="middle")
        )

    elements.append(
        f'<line x1="{left:.1f}" x2="{left + width:.1f}" y1="{avg_y:.1f}" '
        f'y2="{avg_y:.1f}" stroke="#d62728" stroke-width="2.5" stroke-dasharray="9 7" />'
    )
    elements.append(
        svg_text(
            "Average context window",
            x=left + width - 8,
            y=avg_y - 10,
            size=13,
            fill="#d62728",
            anchor="end",
            weight="700",
        )
    )
    elements.append(
        f'<rect x="{left:.1f}" y="{top:.1f}" width="{width:.1f}" height="{height:.1f}" '
        'fill="none" stroke="#555555" stroke-width="1.2" />'
    )
    elements.append(svg_text(title, x=left, y=top - 24, size=18, weight="700"))
    elements.append(svg_text(x_label, x=left + width / 2, y=bottom + 62, size=15, anchor="middle"))
    if y_label:
        label_x = left - 72
        label_y = top + height / 2
        elements.append(
            f'<text x="{label_x:.1f}" y="{label_y:.1f}" font-size="15" fill="#222222" '
            'font-family="DejaVu Sans, Arial, Helvetica, sans-serif" font-style="normal" '
            f'text-anchor="middle" transform="rotate(-90 {label_x:.1f} {label_y:.1f})">'
            f"{escape(y_label)}</text>"
        )
    return elements


def svg_legend(items: list[tuple[str, str]], *, x: float, y: float) -> list[str]:
    row_height = 24
    width = 265
    height = 18 + row_height * len(items)
    elements = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        'rx="5" fill="white" stroke="#dddddd" opacity="0.95" />'
    ]
    for index, (label, color) in enumerate(items):
        row_y = y + 22 + index * row_height
        elements.append(
            f'<line x1="{x + 14:.1f}" x2="{x + 46:.1f}" y1="{row_y:.1f}" '
            f'y2="{row_y:.1f}" stroke="{color}" stroke-width="3" />'
        )
        elements.append(svg_text(label, x=x + 56, y=row_y + 5, size=12, fill="#333333"))
    return elements


def render_svg(args: argparse.Namespace, output_path: Path) -> Path:
    canvas_width = 1500
    canvas_height = 650
    top = 128
    chart_height = 390
    left_a = 105
    chart_width = 610
    gap = 90
    left_b = left_a + chart_width + gap

    elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" '
        f'viewBox="0 0 {canvas_width} {canvas_height}">',
        "<style>text { font-family: Arial, Helvetica, sans-serif; font-style: normal; }</style>",
        "<defs>",
    ]
    elements.append(
        svg_clip_path(
            panel_id="catalog",
            left=left_a,
            top=top,
            width=chart_width,
            height=chart_height,
        )
    )
    elements.append(
        svg_clip_path(
            panel_id="session",
            left=left_b,
            top=top,
            width=chart_width,
            height=chart_height,
        )
    )
    elements.append("</defs>")
    elements.append('<rect width="100%" height="100%" fill="white" />')
    elements.append(
        svg_text(
            "Context Window Pressure Grows with Tool Scale and Tool Use",
            x=canvas_width / 2,
            y=48,
            size=24,
            anchor="middle",
            weight="700",
        )
    )

    elements.extend(
        svg_panel_base(
            title="Available Tools Consume Context",
            x_label="Number of available tools",
            x_max=args.max_tools,
            left=left_a,
            top=top,
            width=chart_width,
            height=chart_height,
            y_label="Context length (tokens)",
        )
    )
    elements.extend(
        svg_panel_base(
            title="Tool Use During a Session Also Grows Context",
            x_label="Tool calls used in session",
            x_max=args.max_session_tool_calls,
            left=left_b,
            top=top,
            width=chart_width,
            height=chart_height,
        )
    )

    tool_counts = sample_counts(args.max_tools, args.num_points)
    ict_crossing = threshold_crossing_count(ICT_TOKENS_PER_TOOL)
    ntilc_crossing = threshold_crossing_count(NTILC_TOKENS_PER_TOOL)
    elements.append('<g clip-path="url(#catalog-clip)">')
    ict_counts, ict_values = clipped_context_series(tool_counts, ICT_TOKENS_PER_TOOL)
    elements.append(
        svg_polyline(
            ict_counts,
            ict_values,
            x_max=args.max_tools,
            left=left_a,
            top=top,
            width=chart_width,
            height=chart_height,
            color="#e6b800",
        )
    )
    ntilc_counts, ntilc_values = clipped_context_series(tool_counts, NTILC_TOKENS_PER_TOOL)
    elements.append(
        svg_polyline(
            ntilc_counts,
            ntilc_values,
            x_max=args.max_tools,
            left=left_a,
            top=top,
            width=chart_width,
            height=chart_height,
            color="#0072b2",
        )
    )
    elements.append("</g>")
    if ict_crossing <= args.max_tools:
        x = svg_x(ict_crossing, args.max_tools, left_a, chart_width)
        elements.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top:.1f}" y2="{top + chart_height:.1f}" '
            'stroke="#b38f00" stroke-width="2" stroke-dasharray="4 5" />'
        )
        elements.append(svg_text(f"ICT hits 1M at ~{ict_crossing:,.0f} tools", x=x + 12, y=335, size=14, fill="#6f5700"))
    elements.extend(
        svg_legend(
            [
                (f"ICT: {ICT_TOKENS_PER_TOOL} tokens/tool", "#e6b800"),
                (f"NTILC: {NTILC_TOKENS_PER_TOOL} token/tool", "#0072b2"),
            ],
            x=left_a + 16,
            y=top + 16,
        )
    )
    elements.append(
        svg_text(
            f"NTILC reaches 1M at ~{ntilc_crossing:,.0f} tools",
            x=left_a + 20,
            y=top + chart_height - 18,
            size=13,
            fill="#00507d",
        )
    )

    tool_calls = sample_counts(args.max_session_tool_calls, args.num_points)
    elements.append('<g clip-path="url(#session-clip)">')
    for tokens_per_call, _label, color in SESSION_TOOL_RESULT_TOKEN_SIZES:
        session_counts, session_values = clipped_context_series(tool_calls, tokens_per_call)
        elements.append(
            svg_polyline(
                session_counts,
                session_values,
                x_max=args.max_session_tool_calls,
                left=left_b,
                top=top,
                width=chart_width,
                height=chart_height,
                color=color,
            )
        )
    elements.append("</g>")
    large_result_crossing = threshold_crossing_count(5_000)
    medium_result_crossing = threshold_crossing_count(1_000)
    if large_result_crossing <= args.max_session_tool_calls:
        x = svg_x(large_result_crossing, args.max_session_tool_calls, left_b, chart_width)
        elements.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top:.1f}" y2="{top + chart_height:.1f}" '
            'stroke="#cc79a7" stroke-width="2" stroke-dasharray="4 5" />'
        )
        elements.append(
            svg_text(
                f"5,000-token results hit 1M at ~{large_result_crossing:,.0f} calls",
                x=x + 12,
                y=335,
                size=14,
                fill="#7a315f",
            )
        )
    elements.extend(
        svg_legend(
            [(label, color) for _tokens_per_call, label, color in SESSION_TOOL_RESULT_TOKEN_SIZES],
            x=left_b + 16,
            y=top + 16,
        )
    )
    elements.append(
        svg_text(
            f"1,000-token results reach 1M at ~{medium_result_crossing:,.0f} calls",
            x=left_b + 20,
            y=top + chart_height - 18,
            size=13,
            fill="#27627d",
        )
    )
    elements.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements), encoding="utf-8")
    return output_path


def convert_svg_to_png(svg_path: Path, png_path: Path) -> Path:
    converter = shutil.which("convert") or shutil.which("magick")
    if converter is None:
        raise SystemExit(
            "matplotlib is unavailable and no SVG-to-PNG converter was found. "
            f"Saved SVG to {svg_path}. Install matplotlib or ImageMagick to write PNG output."
        )

    png_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([converter, str(svg_path), str(png_path)], check=True)
    return png_path


def render_fallback(args: argparse.Namespace) -> Path:
    if args.output_path.suffix.lower() == ".svg":
        return render_svg(args, args.output_path)

    svg_path = args.output_path.with_suffix(".svg")
    render_svg(args, svg_path)
    return convert_svg_to_png(svg_path, args.output_path)


def main() -> None:
    args = parse_args()
    if args.max_tools < 1:
        raise ValueError("--max-tools must be at least 1")
    if args.max_session_tool_calls < 1:
        raise ValueError("--max-session-tool-calls must be at least 1")
    if args.num_points < 2:
        raise ValueError("--num-points must be at least 2")

    saved_path = render_fallback(args) if plt is None else render_matplotlib(args)
    print(f"Saved plot to {saved_path}")


if __name__ == "__main__":
    main()
