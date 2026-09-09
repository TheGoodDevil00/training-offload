"""
reporter.py
===========
Automated report and leaderboard generator for the Overnight Drone Tournament.
Generates:
  - LEADERBOARD.md (Comprehensive markdown report, finalist badges, winner rationale)
  - leaderboard.json (Full machine-readable metrics)
  - leaderboard.csv (Tabular metrics for spreadsheet analysis)
  - training_curves.png (Visual comparison plots via matplotlib)
  - tournament_summary.txt (Terse plain-text summary for terminal/logs)
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import TournamentConfig
from .metrics import CandidateMetrics


def format_time_duration(seconds: float) -> str:
    """Formats seconds into human-readable hours, minutes, and seconds."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hrs > 0:
        return f"{hrs}h {mins:02d}m {secs:02d}s"
    elif mins > 0:
        return f"{mins}m {secs:02d}s"
    return f"{secs}s"


def generate_leaderboard_markdown(
    leaderboard: List[CandidateMetrics],
    winner: Optional[CandidateMetrics],
    top_finalists: List[CandidateMetrics],
    total_elapsed_s: float,
    config: TournamentConfig,
    checkpoints_dir: Path,
) -> str:
    """Generates the comprehensive LEADERBOARD.md documentation."""
    lines: List[str] = []
    lines.append("# 🏆 Overnight Drone-Footage Model Tournament — Final Report")
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**Total GPU Time:** {format_time_duration(total_elapsed_s)}  ")
    lines.append(f"**Hardware Target:** NVIDIA RTX 4050 Laptop GPU (6 GB VRAM)  ")
    lines.append(f"**Primary Objective:** Offline Drone-Footage Annotation & Curation (Recall & Small-Object Focused)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. Executive Summary & Recommended Winner
    lines.append("## 🥇 Recommended Annotation Model")
    lines.append("")
    if winner:
        lines.append(f"### **`{winner.candidate_id}` — {winner.name}**")
        lines.append("")
        lines.append(f"- **Drone Annotation Score:** **`{winner.composite_drone_score:.2f} / 100`**")
        lines.append(f"- **Recall (Priority #1):** **`{winner.recall * 100:.2f}%`** (Missed Object Rate: `{winner.missed_object_rate:.2f}%`)")
        lines.append(f"- **Small-Object Recall (Priority #2):** **`{winner.small_object_recall * 100:.2f}%`**")
        lines.append(f"- **mAP50-95:** `{winner.mAP50_95 * 100:.2f}%` | **mAP50:** `{winner.mAP50 * 100:.2f}%`")
        lines.append(f"- **Precision:** `{winner.precision * 100:.2f}%`")
        lines.append(f"- **Hard Validation Set Recall:** `{winner.hard_set_recall * 100:.2f}%`")
        lines.append(f"- **Resolution:** `{winner.imgsz}x{winner.imgsz}px` | **Batch:** `{winner.batch}`")
        lines.append(f"- **Model Size:** `{winner.model_size_mb:.1f} MB` (`{winner.parameters_m:.1f}M params`)")
        lines.append(f"- **Inference Latency:** `{winner.inference_latency_ms:.1f} ms` (~`{winner.fps:.1f} FPS`)")
        lines.append(f"- **Checkpoint Path:** `{winner.weights_path}`")
        lines.append("")
        lines.append("#### **Selection Rationale:**")
        lines.append(
            "> For offline drone annotation, **recall on small and distant objects is the critical metric** "
            "because missed human detections cannot be reviewed or curated by human annotators. "
            f"`{winner.candidate_id}` achieved the tournament's highest composite score by minimizing missed targets "
            f"({winner.missed_object_rate:.1f}% missed object rate) while maintaining strong localization fidelity "
            f"({winner.mAP50_95 * 100:.1f}% mAP50-95) and robust handling of challenging frames on the hard validation set."
        )
    else:
        lines.append("*No winning candidate selected.*")

    lines.append("")
    lines.append("---")
    lines.append("")

    # 2. Top 3 Finalists
    lines.append("## 🥈 Top 3 Tournament Finalists")
    lines.append("")
    lines.append("| Rank | Candidate ID | Model & Config | Recall | Small-Obj Rec | mAP50-95 | Drone Score | Checkpoint |")
    lines.append("|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|")
    for idx, f in enumerate(top_finalists[:3], 1):
        medal = "🥇" if idx == 1 else ("🥈" if idx == 2 else "🥉")
        lines.append(
            f"| {medal} **#{idx}** | `{f.candidate_id}` | {f.name} ({f.imgsz}px) | "
            f"**{f.recall * 100:.1f}%** | **{f.small_object_recall * 100:.1f}%** | "
            f"{f.mAP50_95 * 100:.1f}% | **{f.composite_drone_score:.2f}** | `{Path(f.weights_path).name}` |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    # 3. Full Ranked Leaderboard Table
    lines.append("## 📊 Complete Ranked Tournament Leaderboard")
    lines.append("")
    lines.append(
        "Ranked by **Drone Annotation Score** (Recall: 35%, Small-Obj Recall: 25%, mAP50-95: 20%, Precision: 15%, Hard Set: 5%):"
    )
    lines.append("")
    lines.append(
        "| Rank | Status | Candidate ID | Phase | ImgSz | Recall (P1) | Small-Obj Rec | mAP50-95 | Prec | Hard Rec | Latency | Size | Drone Score |"
    )
    lines.append(
        "|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    )

    for idx, m in enumerate(leaderboard, 1):
        status_badge = "🏆 Winner" if m.status == "winner" else ("🥈 Finalist" if m.phase == 3 else ("🎯 Shortlist" if m.phase == 2 else "Screening"))
        lines.append(
            f"| {idx} | {status_badge} | `{m.candidate_id}` | P{m.phase} | {m.imgsz}px | "
            f"**{m.recall * 100:.1f}%** | **{m.small_object_recall * 100:.1f}%** | "
            f"{m.mAP50_95 * 100:.1f}% | {m.precision * 100:.1f}% | {m.hard_set_recall * 100:.1f}% | "
            f"{m.inference_latency_ms:.1f}ms | {m.model_size_mb:.1f}MB | **{m.composite_drone_score:.2f}** |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    # 4. Phase Progression & Successive Halving Dynamics
    lines.append("## 🔄 Successive-Halving Tournament Progression")
    lines.append("")
    lines.append("```")
    lines.append("Phase 1: Screening     [ 8 Candidates @ 640px  ] -> Fast exploration across YOLO11/26 & augmentations")
    lines.append("                              ↓ (Top ~50% kept)")
    lines.append("Phase 2: Shortlist     [ 4 Candidates @ 960px  ] -> Warm-started deep tuning with aerial perspective")
    lines.append("                              ↓ (Top 2-3 kept)")
    lines.append("Phase 3: Finalists     [ 2 Candidates @ HighRes] -> Maximum GPU time allocated to top contenders")
    lines.append("                              ↓")
    lines.append(f"Winner Selected        [{winner.candidate_id if winner else 'N/A'}]")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 5. Next Steps Workflow
    lines.append("## 🚀 Recommended Next Steps")
    lines.append("")
    lines.append("1. **Offline Drone Footage Annotation:**")
    lines.append(f"   - Use `{winner.weights_path if winner else 'best.pt'}` to generate candidate detections across raw drone video files.")
    lines.append("2. **Human Verification / Rapid Curation:**")
    lines.append("   - Review predicted bounding boxes; since recall is optimized, minimal objects are missed.")
    lines.append("3. **Downstream Lightweight Deployment Model:**")
    lines.append("   - Use the newly curated high-quality dataset to train an edge-optimized model (e.g. YOLO11n / NCNN for RPi 5).")
    lines.append("")

    return "\n".join(lines)


def generate_training_curves_plot(
    leaderboard: List[CandidateMetrics],
    output_path: Path,
):
    """
    Generates multi-panel comparison visualization using matplotlib:
      1. Drone Annotation Quality Score by Candidate
      2. Recall vs mAP50-95 trade-off
      3. Small-Object Recall comparison
      4. Training Time per Model
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive headless backend
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Overnight Drone Model Tournament — Comparative Analysis", fontsize=16, fontweight="bold")

        # Color palette by phase
        phase_colors = {1: "#3498db", 2: "#f39c12", 3: "#2ecc71"}

        # 1. Bar chart: Composite Drone Score
        names = [m.candidate_id.replace("P1-", "").replace("P2-", "").replace("P3-", "") for m in leaderboard[:8]]
        scores = [m.composite_drone_score for m in leaderboard[:8]]
        colors = [phase_colors.get(m.phase, "#95a5a6") for m in leaderboard[:8]]

        axs[0, 0].barh(names[::-1], scores[::-1], color=colors[::-1], edgecolor="black", alpha=0.85)
        axs[0, 0].set_title("Drone Annotation Score (Higher = Better)", fontweight="bold")
        axs[0, 0].set_xlabel("Score (0-100)")
        axs[0, 0].set_xlim(0, 100)
        for i, v in enumerate(scores[::-1]):
            axs[0, 0].text(v + 1.5, i, f"{v:.1f}", va="center", fontweight="bold", fontsize=9)

        # 2. Scatter: Recall vs mAP50-95
        recalls = [m.recall * 100 for m in leaderboard]
        map95s = [m.mAP50_95 * 100 for m in leaderboard]
        scatter_colors = [phase_colors.get(m.phase, "#95a5a6") for m in leaderboard]

        scatter = axs[0, 1].scatter(recalls, map95s, c=scatter_colors, s=120, edgecolors="black", alpha=0.8)
        axs[0, 1].set_title("Recall vs mAP50-95 (Drone Annotation Frontier)", fontweight="bold")
        axs[0, 1].set_xlabel("Recall (%) [Priority #1]")
        axs[0, 1].set_ylabel("mAP50-95 (%)")
        axs[0, 1].grid(True, linestyle="--", alpha=0.5)

        for m in leaderboard[:4]:
            axs[0, 1].annotate(
                m.candidate_id.split("-")[-1],
                (m.recall * 100, m.mAP50_95 * 100),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                fontweight="bold"
            )

        # 3. Bar chart: Small-Object Recall
        small_recs = [m.small_object_recall * 100 for m in leaderboard[:8]]
        axs[1, 0].bar(names, small_recs, color="#9b59b6", edgecolor="black", alpha=0.85)
        axs[1, 0].set_title("Small-Object Recall (%) [Priority #2]", fontweight="bold")
        axs[1, 0].set_ylabel("Recall on Distant Targets (%)")
        axs[1, 0].tick_params(axis="x", rotation=30)
        axs[1, 0].grid(axis="y", linestyle="--", alpha=0.5)

        # 4. Training Time per Model
        train_times = [m.training_time_s / 60.0 for m in leaderboard[:8]]
        axs[1, 1].bar(names, train_times, color="#34495e", edgecolor="black", alpha=0.85)
        axs[1, 1].set_title("Training Time per Candidate (Minutes)", fontweight="bold")
        axs[1, 1].set_ylabel("Minutes")
        axs[1, 1].tick_params(axis="x", rotation=30)
        axs[1, 1].grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        print(f"[reporter] Training curves plot saved to {output_path}")
    except Exception as e:
        print(f"[reporter] Plot generation skipped: {e}")


def save_tournament_reports(
    leaderboard: List[CandidateMetrics],
    winner: Optional[CandidateMetrics],
    top_finalists: List[CandidateMetrics],
    total_elapsed_s: float,
    config: TournamentConfig,
    output_dir: Path,
    checkpoints_dir: Path,
):
    """
    Saves all tournament outputs (LEADERBOARD.md, leaderboard.json, leaderboard.csv,
    tournament_summary.txt, and training_curves.png).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. LEADERBOARD.md
    md_content = generate_leaderboard_markdown(
        leaderboard=leaderboard,
        winner=winner,
        top_finalists=top_finalists,
        total_elapsed_s=total_elapsed_s,
        config=config,
        checkpoints_dir=checkpoints_dir,
    )
    (output_dir / "LEADERBOARD.md").write_text(md_content, encoding="utf-8")

    # 2. leaderboard.json
    json_data = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_seconds": total_elapsed_s,
        "winner": winner.to_dict() if winner else None,
        "finalists": [f.to_dict() for f in top_finalists],
        "leaderboard": [m.to_dict() for m in leaderboard],
    }
    (output_dir / "leaderboard.json").write_text(
        json.dumps(json_data, indent=2), encoding="utf-8"
    )

    # 3. leaderboard.csv
    csv_file = output_dir / "leaderboard.csv"
    if leaderboard:
        with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "rank", "candidate_id", "name", "phase", "status", "imgsz", "batch",
                "composite_drone_score", "recall", "missed_object_rate",
                "small_object_recall", "mAP50", "mAP50_95", "precision",
                "hard_set_recall", "hard_set_mAP50", "inference_latency_ms", "fps",
                "model_size_mb", "parameters_m", "training_time_s", "epochs_completed",
                "weights_path"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for m in leaderboard:
                d = m.to_dict()
                d["missed_object_rate"] = round(m.missed_object_rate, 2)
                writer.writerow(d)

    # 4. Plain-Text Summary
    summary_lines = [
        "=" * 68,
        "   OVERNIGHT DRONE MODEL TOURNAMENT — SUMMARY",
        "=" * 68,
        f"Total Time: {format_time_duration(total_elapsed_s)}",
        f"Candidates Evaluated: {len(leaderboard)}",
        "",
        f"WINNER: {winner.candidate_id if winner else 'N/A'} ({winner.name if winner else ''})",
        f"  Drone Score:      {winner.composite_drone_score:.2f} / 100" if winner else "",
        f"  Recall:           {winner.recall * 100:.2f}% (Missed: {winner.missed_object_rate:.2f}%)" if winner else "",
        f"  Small-Obj Recall: {winner.small_object_recall * 100:.2f}%" if winner else "",
        f"  mAP50-95:         {winner.mAP50_95 * 100:.2f}%" if winner else "",
        f"  Weights:          {winner.weights_path}" if winner else "",
        "=" * 68,
    ]
    (output_dir / "tournament_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    # 5. Training Curves PNG Plot
    generate_training_curves_plot(
        leaderboard=leaderboard,
        output_path=output_dir / "training_curves.png",
    )

    print(f"\n[reporter] All tournament artifacts generated under {output_dir}")
    print(f"  • LEADERBOARD.md       (Detailed analysis)")
    print(f"  • leaderboard.json     (Structured data)")
    print(f"  • leaderboard.csv      (Spreadsheet table)")
    print(f"  • training_curves.png  (Visual comparison chart)")
    print(f"  • tournament_summary.txt")
