"""Pareto Frontier Analysis and Optimization Curves for VLM Serving."""

from __future__ import annotations

import os
from typing import Any
from pydantic import BaseModel


class ParetoPoint(BaseModel):
    """Configuration candidate point in multi-objective optimization space."""

    name: str
    latency_ms: float      # Lower is better
    quality_score: float   # Higher is better (e.g. F1 or 1.0 - CER)
    vram_mb: float         # Lower is better
    throughput_rps: float  # Higher is better
    is_pareto: bool = False
    details: dict[str, Any] = {}


def compute_pareto_frontier(
    points: list[ParetoPoint],
    minimize_keys: list[str] = ["latency_ms", "vram_mb"],
    maximize_keys: list[str] = ["quality_score", "throughput_rps"],
) -> list[ParetoPoint]:
    """Identify non-dominated solutions in multi-dimensional objective space.

    A point P dominates point Q if P is no worse than Q on all objectives
    and strictly better on at least one.
    """
    pareto_points: list[ParetoPoint] = []

    for p in points:
        dominated = False
        for other in points:
            if p is other:
                continue

            # Check if 'other' dominates 'p'
            better_or_equal = True
            strictly_better = False

            # Check minimized objectives (latency, vram)
            for k in minimize_keys:
                val_other = getattr(other, k)
                val_p = getattr(p, k)
                if val_other > val_p:
                    better_or_equal = False
                    break
                if val_other < val_p:
                    strictly_better = True

            if not better_or_equal:
                continue

            # Check maximized objectives (quality, throughput)
            for k in maximize_keys:
                val_other = getattr(other, k)
                val_p = getattr(p, k)
                if val_other < val_p:
                    better_or_equal = False
                    break
                if val_other > val_p:
                    strictly_better = True

            if better_or_equal and strictly_better:
                dominated = True
                break

        p.is_pareto = not dominated
        if not dominated:
            pareto_points.append(p)

    return pareto_points


def generate_pareto_ascii_chart(points: list[ParetoPoint]) -> str:
    """Generate a clean ASCII trade-off visualization for documentation."""
    lines = [
        "### Multi-Objective Trade-Off (Latency vs Quality vs VRAM)",
        "",
        "| Configuration | Latency p50 (ms) | Quality (Score) | Peak VRAM (MB) | Throughput (req/s) | Pareto Frontier |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for pt in points:
        status = "**Yes (Optimal)**" if pt.is_pareto else "Dominated"
        lines.append(
            f"| `{pt.name}` | {pt.latency_ms:.1f} | {pt.quality_score:.3f} | {pt.vram_mb:.0f} | {pt.throughput_rps:.1f} | {status} |"
        )
    return "\n".join(lines)


def render_pareto_image(points: list[ParetoPoint], output_path: str) -> bool:
    """Render a graphical Pareto chart using matplotlib or PIL fallback."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

        for pt in points:
            color = "#10b981" if pt.is_pareto else "#94a3b8"
            marker = "*" if pt.is_pareto else "o"
            size = 140 if pt.is_pareto else 70
            ax.scatter(pt.latency_ms, pt.quality_score, color=color, marker=marker, s=size, zorder=3)
            ax.annotate(
                pt.name,
                (pt.latency_ms, pt.quality_score),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                weight="bold" if pt.is_pareto else "normal",
            )

        ax.set_title("VLM Pareto Frontier: Quality Score vs. Latency", fontsize=12, pad=12)
        ax.set_xlabel("Latency (ms) [Lower is better]", fontsize=10)
        ax.set_ylabel("Quality Score [Higher is better]", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return True
    except Exception:
        # Fallback to minimal PIL image if matplotlib is not available
        try:
            from PIL import Image, ImageDraw

            img = Image.new("RGB", (600, 350), color="#1e293b")
            draw = ImageDraw.Draw(img)
            draw.text((20, 20), "VLM Serving Pareto Frontier", fill="#f8fafc")
            y = 60
            for pt in points:
                status = "[PARETO]" if pt.is_pareto else ""
                draw.text((20, y), f"{pt.name}: {pt.latency_ms:.1f}ms, Q={pt.quality_score:.2f} {status}", fill="#38bdf8")
                y += 30
            img.save(output_path)
            return True
        except Exception:
            return False
