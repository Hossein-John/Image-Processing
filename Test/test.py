import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def _box(ax, x, y, w, h, label, fc="#FFA43B", ec="#1b3a5a", text="#0B1A39", r=0.15, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.2,rounding_size={r}",
                                fc=fc, ec=ec, lw=1.8))
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=fs, color=text, weight="bold" if bold else "normal")

def _arrow(ax, xy1, xy2, lw=1.6):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle="-|>", lw=lw, ec="#1b3a5a", fc="#1b3a5a"))

def draw_yolov8_shape(dark=False, show=True, save_path=None, title="YOLOv8 Architecture"):
    # Palette
    bg   = "#0A0A0A" if dark else "white"
    text = "white"   if dark else "#0B1A39"
    edge = "white"   if dark else "#1b3a5a"
    c_backbone = "#FFD48A" if not dark else "#3f4b57"
    c_c2f      = "#FFC164" if not dark else "#2f6a99"
    c_sppf     = "#F2A65A" if not dark else "#7a4c1f"
    c_neck     = "#B7E1FF" if not dark else "#244e7a"
    c_head     = "#A6FFBC" if not dark else "#2e6a4d"
    c_detect   = "#7CE8A6" if not dark else "#2a8b62"

    fig, ax = plt.subplots(figsize=(12,6))
    ax.set_xlim(0, 24); ax.set_ylim(0, 14); ax.axis("off")
    fig.patch.set_facecolor(bg)

    ax.text(12, 13.4, title, ha="center", va="center", fontsize=20, color=text, weight="bold")

    # --- Backbone (left to right, downsampling) ---
    # Input
    _box(ax, 1, 10, 3, 2, "Input\n(B×3×H×W)", fc=c_backbone, ec=edge, text=text, bold=True)

    # Stem + C2f stages (P2..P5 features)
    _arrow(ax, (4.2, 11), (5.2, 11))
    _box(ax, 5.2, 10, 3.2, 2, "Conv (stride=2)\n↓", fc=c_backbone, ec=edge, text=text)

    _arrow(ax, (8.5, 11), (9.7, 11))
    _box(ax, 9.7, 10, 3.6, 2, "C2f x N\n(P2)", fc=c_c2f, ec=edge, text=text, bold=True)

    _arrow(ax, (13.35, 11), (14.35, 11))
    _box(ax, 14.35, 10, 3.2, 2, "Conv s=2\n↓", fc=c_backbone, ec=edge, text=text)

    _arrow(ax, (17.6, 11), (18.8, 11))
    _box(ax, 18.8, 10, 3.6, 2, "C2f x N\n(P3)", fc=c_c2f, ec=edge, text=text, bold=True)

    # Downsample to P4
    _arrow(ax, (20.6, 10), (20.6, 8.8))
    _box(ax, 19.0, 7.8, 3.2, 2, "Conv s=2\n↓", fc=c_backbone, ec=edge, text=text)
    _arrow(ax, (20.6, 7.8), (20.6, 6.8))
    _box(ax, 18.8, 5.8, 3.6, 2, "C2f x N\n(P4)", fc=c_c2f, ec=edge, text=text, bold=True)

    # Downsample to P5 + SPPF
    _arrow(ax, (20.6, 5.8), (20.6, 4.6))
    _box(ax, 19.0, 3.6, 3.2, 2, "Conv s=2\n↓", fc=c_backbone, ec=edge, text=text)
    _arrow(ax, (20.6, 3.6), (20.6, 2.6))
    _box(ax, 18.6, 1.6, 4.0, 2, "C2f x N\n+ SPPF\n(P5)", fc=c_sppf, ec=edge, text=text, bold=True)

    # --- Neck: PAN (upsample + concat + C2f) ---
    # Up P5->P4
    _arrow(ax, (18.6, 2.6), (15.0, 6.8))
    _box(ax, 14.0, 6.0, 3.0, 1.6, "Upsample", fc=c_neck, ec=edge, text=text)
    _arrow(ax, (17.0, 6.8), (16.0, 8.6))
    _box(ax, 13.8, 8.6, 4.4, 1.8, "Concat(P4)", fc=c_neck, ec=edge, text=text)
    _arrow(ax, (16.0, 8.6), (12.6, 8.6))
    _box(ax, 9.8, 8.0, 5.6, 2.0, "C2f\n(P4')", fc=c_neck, ec=edge, text=text, bold=True)

    # Up P4'->P3
    _arrow(ax, (9.8, 8.6), (8.6, 10.8))
    _box(ax, 7.6, 10.2, 3.0, 1.6, "Upsample", fc=c_neck, ec=edge, text=text)
    _arrow(ax, (10.6, 11), (10.6, 12.0))
    _box(ax, 9.2, 12.0, 2.8, 1.6, "Concat(P3)", fc=c_neck, ec=edge, text=text)
    _arrow(ax, (10.6, 12.0), (7.0, 12.0))
    _box(ax, 4.2, 11.4, 5.6, 2.0, "C2f\n(P3')", fc=c_neck, ec=edge, text=text, bold=True)

    # Down P3'->P4'' and P4'->P5''
    _arrow(ax, (7.0, 11.4), (7.0, 9.4))
    _box(ax, 5.6, 8.6, 2.8, 1.6, "Downsample", fc=c_neck, ec=edge, text=text)
    _arrow(ax, (8.4, 9.4), (10.0, 9.4))
    _box(ax, 10.0, 8.6, 4.6, 1.8, "Concat(P4')", fc=c_neck, ec=edge, text=text)
    _arrow(ax, (12.3, 8.6), (12.3, 7.0))
    _box(ax, 9.8, 6.3, 5.0, 1.9, "C2f\n(P4'')", fc=c_neck, ec=edge, text=text, bold=True)

    _arrow(ax, (12.3, 6.3), (12.3, 4.4))
    _box(ax, 10.8, 3.6, 2.8, 1.6, "Downsample", fc=c_neck, ec=edge, text=text)
    _arrow(ax, (13.6, 4.4), (16.0, 4.4))
    _box(ax, 16.0, 3.6, 4.2, 1.9, "Concat(P5)", fc=c_neck, ec=edge, text=text)
    _arrow(ax, (18.1, 3.6), (18.1, 2.2))
    _box(ax, 15.8, 1.5, 4.6, 2.0, "C2f\n(P5'')", fc=c_neck, ec=edge, text=text, bold=True)

    # --- Head: decoupled Detect at P3', P4'', P5'' ---
    _box(ax, 4.0, 13.0, 4.8, 1.6, "Head (Detect)\nP3'", fc=c_head, ec=edge, text=text, bold=True)
    _arrow(ax, (6.6, 11.4), (6.4, 13.0))

    _box(ax, 9.6, 9.9, 4.8, 1.6, "Head (Detect)\nP4''", fc=c_head, ec=edge, text=text, bold=True)
    _arrow(ax, (12.3, 8.2), (12.0, 9.9))

    _box(ax, 15.2, 6.8, 4.8, 1.6, "Head (Detect)\nP5''", fc=c_head, ec=edge, text=text, bold=True)
    _arrow(ax, (18.1, 3.5), (17.6, 6.8))

    # Output node (boxes, scores)
    _arrow(ax, (8.8, 13.8), (12.0, 13.8))
    _arrow(ax, (12.0, 11.5), (12.0, 13.8))
    _arrow(ax, (17.6, 8.6), (12.2, 13.8))
    _box(ax, 12.2, 13.0, 6.4, 1.6, "Detections (bboxes + cls + conf)", fc=c_detect, ec=edge, text=text, bold=True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)

# ---- Usage ----
draw_yolov8_shape(dark=False, show=True)                            # show on screen
draw_yolov8_shape(dark=False, save_path="yolov8_arch.png", show=False)  # save file
