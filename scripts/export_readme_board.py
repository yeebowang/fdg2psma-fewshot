#!/usr/bin/env python3
"""Export a README-facing mean-metric summary table (no 9-fold grids / ETA / queue)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METHOD_ORDER = (
    "proto_retrieval",
    "nnunet_mim",
    "nnunet",
    "monai_swinvit",
    "monai_scratch",
    "dpdnet_dualenc",
    "dpdnet",
    "seganypet",
    "seganypet_scratch",
    "mae_swinunetr",
    "mae_scratch",
)
METHOD_FDG_INIT = {
    "nnunet_mim": "PET+CT MIM",
    "nnunet": "scratch",
    "mae_swinunetr": "PET/CT MAE SSL",
    "mae_scratch": "scratch",
    "monai_swinvit": "Tang SSL",
    "monai_scratch": "scratch",
    "seganypet": "SegAnyPET-Lesion",
    "seganypet_scratch": "scratch",
    "dpdnet_dualenc": "PET+CT dual-enc",
    "dpdnet": "scratch",
    "proto_retrieval": "none (retrieval)",
}
STAGE_COLS = (
    ("psma_fs50_f258", "PSMA fs50"),
    ("psma_fs10_f258", "PSMA fs10"),
    ("psma_fs5_f258", "PSMA fs5"),
    ("psma_fs0", "PSMA fs0"),
    ("psma_fc70", "PSMA fc70%"),
    ("fdg_test20", "FDG TEST"),
)
METRIC_HEADER = "Dice / FP / FN"


def _pct(x: Any, digits: int = 2) -> str:
    if not isinstance(x, (int, float)) or x != x:
        return "—"
    return f"{100.0 * float(x):.{digits}f}%"


def _metric_triple(st: dict) -> str:
    status = (st.get("status") or "pending").lower()
    if status not in ("done", "running") and st.get("mean") is None:
        return status.upper() if status != "pending" else "—"
    mean, fp, fn = st.get("mean"), st.get("mean_fp"), st.get("mean_fn")
    if mean is None:
        return status.upper()
    return f"{_pct(mean)}\n{_pct(fp)}\n{_pct(fn)}"


def _fdg_cell(st: dict) -> str:
    status = (st.get("status") or "pending").lower()
    if status != "done":
        return status.upper()
    bits = ["DONE"]
    if st.get("train_time"):
        bits.append(str(st["train_time"]))
    ep = st.get("total_epochs") or st.get("epoch")
    if ep:
        bits.append(f"{int(ep)}ep")
    return "\n".join(bits)


def sanitize_board(board: dict) -> dict:
    """Public board: means + status only; strip paths, folds, ETA, queue."""
    out: dict[str, Any] = {
        "protocol": board.get("protocol"),
        "updated_at": board.get("updated_at"),
        "updated_note": "README summary · mean Dice/FP/FN only",
        "metrics": "Dice / FP / FN (%) · Dice excludes empty-GT · FP=FP/Neg · FN=FN/Pos (voxel micro-avg)",
        "methods": {},
    }
    methods = board.get("methods") or {}
    for key in METHOD_ORDER:
        m = methods.get(key)
        if not isinstance(m, dict):
            continue
        row: dict[str, Any] = {
            "label": m.get("label", key),
            "pretrained": METHOD_FDG_INIT.get(key, ""),
        }
        fdg = m.get("fdg_pretrain") or {}
        row["fdg_pretrain"] = {
            "status": fdg.get("status"),
            "train_time": fdg.get("train_time"),
            "total_epochs": fdg.get("total_epochs") or fdg.get("epoch"),
        }
        for sk, _ in STAGE_COLS:
            st = m.get(sk) or {}
            row[sk] = {
                "status": st.get("status"),
                "mean": st.get("mean"),
                "mean_fp": st.get("mean_fp"),
                "mean_fn": st.get("mean_fn"),
                "train_time": st.get("train_time"),
            }
        out["methods"][key] = row
    return out


def render_png(board: dict, png: Path) -> None:
    """Optional PNG board (not used in README; kept for paper / offline use)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    status_color = {
        "done": "#2e7d32",
        "running": "#ef6c00",
        "waiting": "#0277bd",
        "pending": "#757575",
        "queued": "#757575",
        "failed": "#c62828",
    }
    methods = [(k, board["methods"][k]) for k in METHOD_ORDER if k in board.get("methods", {})]
    n = len(methods)
    row_h = 0.92
    header_y = 0.55 + n * row_h + 0.25
    cols = [("fdg", "① FDG\n(shared)", 1.35)] + [
        (sk, f"{hdr}\n{METRIC_HEADER}", 1.35) for sk, hdr in STAGE_COLS
    ]
    x0_method, x0 = 0.08, 1.55
    xs: list[tuple[float, str, str, float]] = []
    x = x0
    for key, hdr, w in cols:
        xs.append((x, key, hdr, w))
        x += w + 0.08
    fig_w = max(14.5, x + 0.35)
    fig_h = max(7.5, 1.1 + n * row_h + 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, header_y + 0.55)
    ax.axis("off")
    ax.set_title(
        f"FDG → PSMA few-shot · mean TEST metrics ({n} methods)\n"
        f"{board.get('updated_at', '')} · {METRIC_HEADER} (%)  ·  no per-fold grids",
        fontsize=11,
        pad=8,
    )
    ax.text(x0_method, header_y + 0.12, "Method\n(pretrained)", fontsize=8.5, fontweight="bold", va="center")
    for gx, _k, hdr, w in xs:
        ax.text(gx + w * 0.5, header_y + 0.12, hdr, fontsize=7.5, fontweight="bold", va="center", ha="center")

    y0 = header_y - 0.55
    for i, (key, m) in enumerate(methods):
        y = y0 - i * row_h
        ax.add_patch(
            FancyBboxPatch(
                (0.05, y - row_h * 0.42),
                fig_w - 0.1,
                row_h * 0.84,
                boxstyle="round,pad=0.012,rounding_size=0.05",
                linewidth=0.5,
                edgecolor="#bdbdbd",
                facecolor="#fafafa" if i % 2 == 0 else "#f5f5f5",
            )
        )
        label = str(m.get("label", key))
        if " (" in label:
            head, tail = label.split(" (", 1)
            label = f"{head}\n({tail}"
        init = METHOD_FDG_INIT.get(key, m.get("pretrained", ""))
        ax.text(
            x0_method,
            y,
            f"{label}\n{init}" if init else label,
            fontsize=7.2,
            fontweight="bold",
            va="center",
            linespacing=1.05,
        )
        for gx, sk, _hdr, w in xs:
            if sk == "fdg":
                st = m.get("fdg_pretrain") or {}
                txt = _fdg_cell(st)
                color = status_color.get((st.get("status") or "pending").lower(), "#424242")
            else:
                st = m.get(sk) or {}
                txt = _metric_triple(st)
                color = status_color.get((st.get("status") or "pending").lower(), "#424242")
            ax.text(
                gx + w * 0.5,
                y,
                txt,
                fontsize=7.0,
                color=color,
                fontweight="bold",
                va="center",
                ha="center",
                linespacing=1.02,
                family="DejaVu Sans Mono",
            )

    png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def markdown_table(board: dict) -> str:
    # Metric columns carry Dice / FP / FN in the header so cell triples are self-explanatory.
    headers = ["Method", "Pretrained", "FDG"] + [
        f"{hdr}<br>{METRIC_HEADER}" for _, hdr in STAGE_COLS
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    methods = board.get("methods") or {}
    for key in METHOD_ORDER:
        m = methods.get(key)
        if not isinstance(m, dict):
            continue
        cells = [
            str(m.get("label", key)).replace("|", "/"),
            METHOD_FDG_INIT.get(key, m.get("pretrained", "")),
            _fdg_cell(m.get("fdg_pretrain") or {}).replace("\n", "<br>"),
        ]
        for sk, _ in STAGE_COLS:
            cells.append(_metric_triple(m.get(sk) or {}).replace("\n", "<br>"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--board",
        type=Path,
        default=Path("/media/ybwang/data1/PSMA-CTRL/ICLR2026/vis/iclr2026_aligned_fdg_fs50_f258_board.json"),
    )
    ap.add_argument("--png", type=Path, default=None, help="Optional PNG (not embedded in README)")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args()
    raw = json.loads(args.board.read_text())
    summary = sanitize_board(raw)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    if args.png is not None:
        render_png(summary, args.png)
        print(f"[export] png={args.png}")
    md = markdown_table(summary)
    if args.out_md:
        args.out_md.write_text(md + "\n")
    print(md)
    print(f"[export] json={args.out_json}")


if __name__ == "__main__":
    main()
