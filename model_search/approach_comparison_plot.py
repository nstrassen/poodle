import argparse
from typing import List, Dict

import matplotlib.pyplot as plt


# Baseline:
# We need to use 5000 items for training; below, we do not get the target accuracy.
# 1) Label 5000 items using LLM: 5000items / 13items/s                                                ; 385 s
# 2) Search Model: Just use BERT default model                                                        ;   0 s
# 3) Fine-tune model using 5000 items: 5 epochs * (74sec-18s) (fine-tune - inf of 4500 items)         ; 280 s
# SUMME                                                                                               ; 665 s
# Result 0.89 accuracy
#
# Baseline + Naive Model Search:
# We need to use 5000 items for training; below, we do not get the target accuracy.
# 1) Label 5000 items using LLM: 5000items / 13items/s                                                ; 385 s
# 2) Search Model: Fine-tune all 10 models                                                            ;2800 s
# 3) Just select best fine-tuned model                                                                ;   0 s
# SUMME                                                                                               ;3185 s
# Result 0.92 accuracy
#
# Model Search (Fine-tune on 500 items)
# 1) Label 1000 items using LLM: 500items / 13items/s                                                 ;  38 s
# 2) Search Model: 10 models * 6s/model                                                               ;  60 s
# 3) Fine-tune model using 500 items: 10 epochs * 7sec (fine-tune)                                    ;  70 s
# SUMME                                                                                               ; 168 s (3.96 Base, 18.96 Base + Naive Model Search)
# Result 0.91 accuracy
#
# Model Search (Fine-tune on 5000 items)
# 1) Label 5000 items using LLM: 500items / 13items/s                                                 ; 385 s
# 2) Search Model: 10 models * 6s/model                                                               ;  60 s
# 3) Fine-tune model using 5000 items: 5 epochs * (74sec-18s) (fine-tune - inf of 4500 items)         ; 280 s
# SUMME                                                                                               ; 725 s (4.39 Base + Naive Model Search)
# Result 0.92 accuracy
#

colors = ['#bae4bc', '#43a2ca', '#0868ac']


def sample_data():
    # Example times in seconds for a few approaches/models
    return [
        {"approach": "Baseline", "label_seconds": 385, "search_seconds": 0, "fine_tune_seconds": 280, "accuracy": 0.89},
        {"approach": "Naive Search", "label_seconds": 385, "search_seconds": 2800, "fine_tune_seconds": 0, "accuracy": 0.92},
        {"approach": "Search (500)", "label_seconds": 38, "search_seconds": 60, "fine_tune_seconds": 70, "accuracy": 0.91},
        {"approach": "Search (5000)", "label_seconds": 385, "search_seconds": 60, "fine_tune_seconds": 280, "accuracy": 0.92},
    ]


def format_seconds(s: float) -> str:
    if s is None:
        return "0s"
    s = float(s)
    if s >= 3600:
        return f"{s / 3600:.1f}h"
    if s >= 60:
        return f"{s / 60:.1f}m"
    if s >= 1:
        return f"{s:.0f}s"
    return f"{s * 1000:.0f}ms"



def plot_stack(rows: List[Dict], out_path: str = "timings.png"):
    if not rows:
        print("No data to plot")
        return

    approaches = [r["approach"] for r in rows]
    labels = [r["label_seconds"] for r in rows]
    searches = [r["search_seconds"] for r in rows]
    fines = [r["fine_tune_seconds"] for r in rows]
    accuracies = [r.get("accuracy", None) for r in rows]

    totals = [labels[i] + searches[i] + fines[i] for i in range(len(rows))]

    # convert seconds -> minutes for plotting
    labels_min = [s / 60.0 for s in labels]
    searches_min = [s / 60.0 for s in searches]
    fines_min = [s / 60.0 for s in fines]
    totals_min = [labels_min[i] + searches_min[i] + fines_min[i] for i in range(len(rows))]

    x = range(len(approaches))
    fig, ax = plt.subplots(figsize=(max(5, len(approaches) * 1.2), 3))

    p1 = ax.bar(x, labels_min, label="Label items", color=colors[0])
    p2 = ax.bar(x, searches_min, bottom=labels_min, label="Model search", color=colors[1])
    bottom_for_fines = [labels_min[i] + searches_min[i] for i in range(len(labels_min))]
    p3 = ax.bar(x, fines_min, bottom=bottom_for_fines, label="Fine-tuning", color=colors[2])

    ax.set_ylabel("Time (minutes)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(approaches, rotation=30, ha="right")

    # remove secondary axis; instead annotate accuracy above each stacked bar
    # use explicit legend handles ordered to match the stacked bars (top -> bottom)
    label_handle = p1[0]
    search_handle = p2[0]
    fine_handle = p3[0]
    ax.legend([fine_handle, search_handle, label_handle], ["Fine-tuning", "Model search", "Label items"], loc="upper right")

    # Add a small headroom above the highest bar so title/text doesn't overlap
    max_total = max(totals_min) if totals_min else 0.0
    headroom = max_total * 0.12 if max_total > 0 else 0.1
    ax.set_ylim(0, max_total + headroom)

    # annotate accuracy (as percentage) above each bar
    offset = max_total * 0.03 if max_total > 0 else 0.02
    for i, acc in enumerate(accuracies):
        if acc is None:
            continue
        y = totals_min[i] + offset
        # ax.text(i, y, f"{acc * 100:.1f}% acc.", ha="center", va="bottom", fontsize=9, color="black")

    # increase padding between title and top of plot
    ax.set_title("Timing breakdown per approach", pad=18)

    plt.tight_layout()
    plt.savefig(f"{out_path}.png")
    plt.savefig(f"{out_path}.pdf")
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot stacked timing bar chart")
    parser.add_argument("--out", default="./plots/timings", help="Output image path (PNG)")
    args = parser.parse_args()

    rows = sample_data()

    plot_stack(rows, args.out)


if __name__ == "__main__":
    main()
