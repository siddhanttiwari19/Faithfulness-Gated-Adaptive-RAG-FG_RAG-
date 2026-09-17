"""
FG-ARAG: Exploratory Data Analysis (EDA) on HotpotQA Distractor Dataset
========================================================================
Generates comprehensive visualizations and statistics for the MSE 1 viva.
Saves all plots to artifacts/plots/eda/.

Usage:
    python scripts/run_eda.py [--max_samples 500]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer

# ─── Configuration ───────────────────────────────────────────────────────────

PLOT_DIR = os.path.join("artifacts", "plots", "eda")
os.makedirs(PLOT_DIR, exist_ok=True)

# Style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.figsize": (10, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

PALETTE = sns.color_palette("Set2")
PALETTE_DARK = sns.color_palette("Dark2")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def save_plot(fig, name):
    path = os.path.join(PLOT_DIR, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [SAVED]: {path}")


def word_count(text):
    return len(text.split())


# ─── Main EDA ────────────────────────────────────────────────────────────────

def main(max_samples=None):
    print("=" * 70)
    print("  FG-ARAG: Exploratory Data Analysis on HotpotQA (Distractor)")
    print("=" * 70)

    # ── 1. Load Dataset ──────────────────────────────────────────────────
    print("\n[LOADING] Loading HotpotQA (distractor) validation split...")
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", cache_dir="data/raw", trust_remote_code=True)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"   Limited to {len(dataset)} samples for faster analysis.")
    else:
        print(f"   Full validation set: {len(dataset)} samples.")

    # ── 2. Build DataFrame ───────────────────────────────────────────────
    print("\n[BUILDING] Building analysis DataFrame...")

    records = []
    for item in dataset:
        titles = item["context"]["title"]
        sentences = item["context"]["sentences"]

        # Count supporting facts
        sup_titles = item["supporting_facts"]["title"]
        sup_sent_ids = item["supporting_facts"]["sent_id"]
        num_supporting_facts = len(sup_titles)
        unique_supporting_docs = len(set(sup_titles))

        # Context stats
        num_paragraphs = len(titles)
        all_sentences = []
        para_lengths = []
        for sents in sentences:
            all_sentences.extend(sents)
            para_lengths.append(len(sents))

        full_context = " ".join(all_sentences)

        records.append({
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "type": item["type"],
            "level": item["level"],
            "num_paragraphs": num_paragraphs,
            "num_total_sentences": len(all_sentences),
            "avg_sentences_per_para": np.mean(para_lengths) if para_lengths else 0,
            "question_word_count": word_count(item["question"]),
            "answer_word_count": word_count(item["answer"]),
            "answer_char_count": len(item["answer"]),
            "context_word_count": word_count(full_context),
            "context_char_count": len(full_context),
            "num_supporting_facts": num_supporting_facts,
            "num_supporting_docs": unique_supporting_docs,
            "full_context": full_context,
        })

    df = pd.DataFrame(records)
    print(f"   DataFrame shape: {df.shape}")

    # ── 3. Summary Statistics ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  [SUMMARY] SUMMARY STATISTICS")
    print("=" * 70)

    print(f"\n  Total Samples:          {len(df)}")
    print(f"  Question Types:         {dict(df['type'].value_counts())}")
    print(f"  Difficulty Levels:      {dict(df['level'].value_counts())}")
    print(f"\n  Answer Length (words):   mean={df['answer_word_count'].mean():.1f}, "
          f"median={df['answer_word_count'].median():.0f}, "
          f"max={df['answer_word_count'].max()}")
    print(f"  Question Length (words): mean={df['question_word_count'].mean():.1f}, "
          f"median={df['question_word_count'].median():.0f}, "
          f"max={df['question_word_count'].max()}")
    print(f"  Context Length (words):  mean={df['context_word_count'].mean():.1f}, "
          f"median={df['context_word_count'].median():.0f}, "
          f"max={df['context_word_count'].max()}")
    print(f"  Paragraphs per Q:       mean={df['num_paragraphs'].mean():.1f}, "
          f"mode={df['num_paragraphs'].mode().values[0]}")
    print(f"  Sentences per Q:        mean={df['num_total_sentences'].mean():.1f}")
    print(f"  Supporting Facts per Q: mean={df['num_supporting_facts'].mean():.1f}")
    print(f"  Supporting Docs per Q:  mean={df['num_supporting_docs'].mean():.1f}")

    # ── 4. PLOTS ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  [PLOTS] GENERATING PLOTS")
    print("=" * 70)

    # ── Plot 1: Question Type Distribution ────────────────────────────
    print("\n[1/10] Question Type Distribution...")
    fig, ax = plt.subplots(figsize=(8, 5))
    type_counts = df["type"].value_counts()
    bars = ax.bar(type_counts.index, type_counts.values, color=[PALETTE[0], PALETTE[1]], edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, type_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + len(df) * 0.01,
                f"{val}\n({val/len(df)*100:.1f}%)", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Distribution of Question Types", fontweight="bold")
    ax.set_xlabel("Question Type")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(type_counts.values) * 1.15)
    save_plot(fig, "01_question_type_distribution")

    # ── Plot 2: Difficulty Level Distribution ─────────────────────────
    print("[2/10] Difficulty Level Distribution...")
    fig, ax = plt.subplots(figsize=(8, 5))
    level_order = ["easy", "medium", "hard"]
    level_counts = df["level"].value_counts().reindex(level_order, fill_value=0)
    colors_level = [PALETTE[2], PALETTE[3], PALETTE[4]]
    bars = ax.bar(level_counts.index, level_counts.values, color=colors_level, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, level_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + len(df) * 0.01,
                f"{val}\n({val/len(df)*100:.1f}%)", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Distribution of Difficulty Levels", fontweight="bold")
    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(level_counts.values) * 1.15)
    save_plot(fig, "02_difficulty_level_distribution")

    # ── Plot 3: Question Type × Difficulty Heatmap ────────────────────
    print("[3/10] Question Type × Difficulty Heatmap...")
    fig, ax = plt.subplots(figsize=(8, 5))
    cross_tab = pd.crosstab(df["type"], df["level"])
    cross_tab = cross_tab.reindex(columns=level_order, fill_value=0)
    sns.heatmap(cross_tab, annot=True, fmt="d", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("Question Type × Difficulty Level", fontweight="bold")
    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Question Type")
    save_plot(fig, "03_type_difficulty_heatmap")

    # ── Plot 4: Answer Length Distribution ────────────────────────────
    print("[4/10] Answer Length Distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Word count
    axes[0].hist(df["answer_word_count"], bins=50, color=PALETTE[0], edgecolor="black", linewidth=0.5, alpha=0.85)
    axes[0].axvline(df["answer_word_count"].mean(), color="red", linestyle="--", label=f"Mean: {df['answer_word_count'].mean():.1f}")
    axes[0].axvline(df["answer_word_count"].median(), color="orange", linestyle="--", label=f"Median: {df['answer_word_count'].median():.0f}")
    axes[0].set_title("Answer Length (Word Count)", fontweight="bold")
    axes[0].set_xlabel("Number of Words")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # Character count
    axes[1].hist(df["answer_char_count"], bins=50, color=PALETTE[1], edgecolor="black", linewidth=0.5, alpha=0.85)
    axes[1].axvline(df["answer_char_count"].mean(), color="red", linestyle="--", label=f"Mean: {df['answer_char_count'].mean():.1f}")
    axes[1].axvline(df["answer_char_count"].median(), color="orange", linestyle="--", label=f"Median: {df['answer_char_count'].median():.0f}")
    axes[1].set_title("Answer Length (Character Count)", fontweight="bold")
    axes[1].set_xlabel("Number of Characters")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    fig.suptitle("Answer Length Distribution", fontweight="bold", fontsize=15, y=1.02)
    fig.tight_layout()
    save_plot(fig, "04_answer_length_distribution")

    # ── Plot 5: Question Length Distribution ──────────────────────────
    print("[5/10] Question Length Distribution...")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["question_word_count"], bins=30, color=PALETTE[2], edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.axvline(df["question_word_count"].mean(), color="red", linestyle="--",
               label=f"Mean: {df['question_word_count'].mean():.1f} words")
    ax.set_title("Question Length Distribution", fontweight="bold")
    ax.set_xlabel("Number of Words")
    ax.set_ylabel("Frequency")
    ax.legend()
    save_plot(fig, "05_question_length_distribution")

    # ── Plot 6: Context Size Analysis ─────────────────────────────────
    print("[6/10] Context Size Analysis...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(df["num_paragraphs"], bins=range(0, df["num_paragraphs"].max() + 2),
                 color=PALETTE[3], edgecolor="black", linewidth=0.5, alpha=0.85, align="left")
    axes[0].set_title("Paragraphs per Question", fontweight="bold")
    axes[0].set_xlabel("Number of Paragraphs")
    axes[0].set_ylabel("Frequency")
    axes[0].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    axes[1].hist(df["num_total_sentences"], bins=30, color=PALETTE[4], edgecolor="black", linewidth=0.5, alpha=0.85)
    axes[1].axvline(df["num_total_sentences"].mean(), color="red", linestyle="--",
                    label=f"Mean: {df['num_total_sentences'].mean():.1f}")
    axes[1].set_title("Total Sentences per Question", fontweight="bold")
    axes[1].set_xlabel("Number of Sentences")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    axes[2].hist(df["context_word_count"], bins=30, color=PALETTE[5], edgecolor="black", linewidth=0.5, alpha=0.85)
    axes[2].axvline(df["context_word_count"].mean(), color="red", linestyle="--",
                    label=f"Mean: {df['context_word_count'].mean():.0f}")
    axes[2].set_title("Context Length (Words)", fontweight="bold")
    axes[2].set_xlabel("Word Count")
    axes[2].set_ylabel("Frequency")
    axes[2].legend()

    fig.suptitle("Context Size Analysis", fontweight="bold", fontsize=15, y=1.02)
    fig.tight_layout()
    save_plot(fig, "06_context_size_analysis")

    # ── Plot 7: Supporting Facts Analysis ─────────────────────────────
    print("[7/10] Supporting Facts Analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sf_counts = df["num_supporting_facts"].value_counts().sort_index()
    axes[0].bar(sf_counts.index, sf_counts.values, color=PALETTE_DARK[0], edgecolor="black", linewidth=0.5)
    axes[0].set_title("Supporting Facts per Question", fontweight="bold")
    axes[0].set_xlabel("Number of Supporting Facts")
    axes[0].set_ylabel("Frequency")
    axes[0].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    sd_counts = df["num_supporting_docs"].value_counts().sort_index()
    axes[1].bar(sd_counts.index, sd_counts.values, color=PALETTE_DARK[1], edgecolor="black", linewidth=0.5)
    axes[1].set_title("Unique Supporting Documents per Question", fontweight="bold")
    axes[1].set_xlabel("Number of Supporting Documents")
    axes[1].set_ylabel("Frequency")
    axes[1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.suptitle("Supporting Facts Analysis (Gold Evidence)", fontweight="bold", fontsize=15, y=1.02)
    fig.tight_layout()
    save_plot(fig, "07_supporting_facts_analysis")

    # ── Plot 8: Answer Length by Type & Difficulty ────────────────────
    print("[8/10] Answer Length by Type & Difficulty...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(data=df, x="type", y="answer_word_count", palette="Set2", ax=axes[0])
    axes[0].set_title("Answer Length by Question Type", fontweight="bold")
    axes[0].set_xlabel("Question Type")
    axes[0].set_ylabel("Answer Word Count")

    sns.boxplot(data=df, x="level", y="answer_word_count", order=level_order, palette="Set3", ax=axes[1])
    axes[1].set_title("Answer Length by Difficulty", fontweight="bold")
    axes[1].set_xlabel("Difficulty Level")
    axes[1].set_ylabel("Answer Word Count")

    fig.suptitle("Answer Length Across Groups", fontweight="bold", fontsize=15, y=1.02)
    fig.tight_layout()
    save_plot(fig, "08_answer_length_by_group")

    # ── Plot 9: Token-Aware Chunking Analysis ─────────────────────────
    print("[9/10] Token-Aware Chunking Analysis...")
    print("   Loading BGE tokenizer for chunking simulation...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5", use_fast=True)

    CHUNK_SIZE = 256
    CHUNK_OVERLAP = 50
    STRIDE = CHUNK_SIZE - CHUNK_OVERLAP

    chunk_counts = []
    chunk_token_lengths = []
    context_token_lengths = []

    # Analyze a sample for chunking (full set can be slow)
    chunk_sample_size = min(len(df), 1000)
    print(f"   Simulating chunking on {chunk_sample_size} samples...")

    for i in range(chunk_sample_size):
        text = df.iloc[i]["full_context"]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        context_token_lengths.append(len(tokens))

        # Simulate chunking
        n_chunks = 0
        start = 0
        while start < len(tokens):
            end = min(start + CHUNK_SIZE, len(tokens))
            chunk_len = end - start
            chunk_token_lengths.append(chunk_len)
            n_chunks += 1
            start += STRIDE
        chunk_counts.append(n_chunks)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(context_token_lengths, bins=30, color=PALETTE[0], edgecolor="black", linewidth=0.5, alpha=0.85)
    axes[0].axvline(np.mean(context_token_lengths), color="red", linestyle="--",
                    label=f"Mean: {np.mean(context_token_lengths):.0f} tokens")
    axes[0].set_title("Context Length (Tokens)", fontweight="bold")
    axes[0].set_xlabel("Token Count")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    axes[1].hist(chunk_counts, bins=range(0, max(chunk_counts) + 2), color=PALETTE[1],
                 edgecolor="black", linewidth=0.5, alpha=0.85, align="left")
    axes[1].axvline(np.mean(chunk_counts), color="red", linestyle="--",
                    label=f"Mean: {np.mean(chunk_counts):.1f} chunks")
    axes[1].set_title(f"Chunks per Document\n(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})", fontweight="bold")
    axes[1].set_xlabel("Number of Chunks")
    axes[1].set_ylabel("Frequency")
    axes[1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    axes[1].legend()

    axes[2].hist(chunk_token_lengths, bins=30, color=PALETTE[2], edgecolor="black", linewidth=0.5, alpha=0.85)
    axes[2].axvline(CHUNK_SIZE, color="red", linestyle="--", label=f"Target: {CHUNK_SIZE}")
    axes[2].set_title("Chunk Token Length Distribution", fontweight="bold")
    axes[2].set_xlabel("Token Count per Chunk")
    axes[2].set_ylabel("Frequency")
    axes[2].legend()

    fig.suptitle("Token-Aware Chunking Analysis (BGE Tokenizer)", fontweight="bold", fontsize=15, y=1.02)
    fig.tight_layout()
    save_plot(fig, "09_chunking_analysis")

    # ── Plot 10: Top Question Starter Words & Answer Word Frequency ───
    print("[10/10] Word Frequency Analysis...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Question first words
    first_words = df["question"].str.split().str[0].str.lower().str.rstrip("?")
    fw_counts = first_words.value_counts().head(10)
    axes[0].barh(fw_counts.index[::-1], fw_counts.values[::-1], color=PALETTE_DARK[2], edgecolor="black", linewidth=0.5)
    axes[0].set_title("Top 10 Question Starter Words", fontweight="bold")
    axes[0].set_xlabel("Frequency")

    # Top answer words (filter stopwords loosely)
    stopwords = {"the", "a", "an", "of", "in", "to", "and", "is", "was", "for", "on", "at", "by", "with", "from", "or", "it", "as", "be", "that", "this", "are", "were"}
    all_answer_words = []
    for ans in df["answer"]:
        words = ans.lower().split()
        all_answer_words.extend([w.strip(".,!?;:'\"()") for w in words if w.lower().strip(".,!?;:'\"()") not in stopwords and len(w) > 1])
    aw_counts = Counter(all_answer_words).most_common(15)
    aw_labels = [w for w, _ in aw_counts]
    aw_values = [c for _, c in aw_counts]
    axes[1].barh(aw_labels[::-1], aw_values[::-1], color=PALETTE_DARK[3], edgecolor="black", linewidth=0.5)
    axes[1].set_title("Top 15 Answer Words (excl. stopwords)", fontweight="bold")
    axes[1].set_xlabel("Frequency")

    fig.suptitle("Word Frequency Analysis", fontweight="bold", fontsize=15, y=1.02)
    fig.tight_layout()
    save_plot(fig, "10_word_frequency_analysis")

    # ── Summary Table ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  [TABLE] FINAL SUMMARY TABLE")
    print("=" * 70)

    summary_data = {
        "Metric": [
            "Total Samples",
            "Question Types",
            "Difficulty Levels",
            "Mean Question Length (words)",
            "Mean Answer Length (words)",
            "Mean Context Length (words)",
            "Mean Context Length (tokens)",
            "Mean Paragraphs per Question",
            "Mean Sentences per Question",
            "Mean Supporting Facts per Question",
            "Mean Supporting Docs per Question",
            f"Mean Chunks per Document (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})",
            "Total Chunks Generated",
        ],
        "Value": [
            len(df),
            f"bridge ({(df['type']=='bridge').sum()}), comparison ({(df['type']=='comparison').sum()})",
            ", ".join([f"{lvl} ({(df['level']==lvl).sum()})" for lvl in level_order]),
            f"{df['question_word_count'].mean():.1f}",
            f"{df['answer_word_count'].mean():.1f}",
            f"{df['context_word_count'].mean():.0f}",
            f"{np.mean(context_token_lengths):.0f}",
            f"{df['num_paragraphs'].mean():.1f}",
            f"{df['num_total_sentences'].mean():.1f}",
            f"{df['num_supporting_facts'].mean():.1f}",
            f"{df['num_supporting_docs'].mean():.1f}",
            f"{np.mean(chunk_counts):.1f}",
            f"{sum(chunk_counts)}",
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Save summary as CSV
    summary_path = os.path.join(PLOT_DIR, "eda_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  [SUCCESS] Summary saved to: {summary_path}")

    print("\n" + "=" * 70)
    print(f"  [SUCCESS] ALL DONE! {10} plots saved to: {PLOT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FG-ARAG EDA Script")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples for faster analysis (default: use all)")
    args = parser.parse_args()
    main(max_samples=args.max_samples)
