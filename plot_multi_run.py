import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# === 設定 ===
RUNS_DIR = "runs"
RESULTS_FILE = "results.jsonl"

# === ユーティリティ ===
def get_output_filename(base_name, run_name=None):
    """
    出力ファイル名を生成する。run_nameが指定された場合は先頭に付加する。
    """
    if run_name:
        return f"{run_name}_{base_name}.png"
    return f"{base_name}.png"
def load_results_by_run(run_name):
    """
    指定run_nameの全サブディレクトリからresults.jsonlを読み込み、
    model_nameごとにDataFrameをまとめて返す。
    """
    pattern = os.path.join(RUNS_DIR, f"{run_name}_*/{RESULTS_FILE}")
    files = glob.glob(pattern)
    model2dfs = defaultdict(list)
    for f in files:
        rows = []
        with open(f, "r", encoding="utf-8") as fin:
            for line in fin:
                row = json.loads(line)
                model = row.get("model_name")
                if model is None:
                    continue
                rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            model = df["model_name"].iloc[0]
            model2dfs[model].append(df)
    # modelごとにconcat
    for model in model2dfs:
        model2dfs[model] = pd.concat(model2dfs[model], ignore_index=True)
    return model2dfs

# === 可視化 ===
def plot_scores(model2dfs, out_dir, run_name=None):
    os.makedirs(out_dir, exist_ok=True)
    # 全モデルのスコア範囲を揃える
    all_labels = ["is_correct", "bleu", "rougeL"]
    # ラベルの表示名を設定
    label_display = {"is_correct": "Accuracy", "bleu": "BLEU", "rougeL": "ROUGE-L"}
    # 0-1スケールで統一
    y_min, y_max = 0.0, 1.0
    models = list(model2dfs.keys())
    n = len(models)
    if n == 0:
        return
    import math
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
    if run_name:
        fig.suptitle(f"Run: {run_name}", fontsize=14)
    for idx, model in enumerate(models):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        df = model2dfs[model]
        values = []
        for k in all_labels:
            if k in df and df[k].notna().any():
                v = df[k].dropna().astype(float).mean()
                if k == "bleu" and v > 1.0:
                    v = v / 100.0
                values.append(v)
            else:
                values.append(0.0)
        display_labels = [label_display[k] for k in all_labels]
        ax.bar(display_labels, values)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"Scores: {model}")
        ax.set_ylabel("score (normalized)")
    # 空きサブプロットを非表示
    for idx in range(n, nrows*ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, get_output_filename("scores_subplots", run_name)))
    plt.close()

def plot_latency(model2dfs, out_dir, log_scale=True, run_name=None):
    os.makedirs(out_dir, exist_ok=True)
    # 全モデルのレイテンシ範囲を揃える
    # すべてのlatency_msを集めてmin/maxを決定
    all_lat = pd.concat([
        df["latency_ms"].dropna().astype(float)
        for df in model2dfs.values() if "latency_ms" in df and df["latency_ms"].notna().any()
    ], ignore_index=True)
    
    # 各モデルのtoken/sec計算
    def calculate_tokens_per_sec(df):
        if "total_tokens" not in df or "latency_ms" not in df:
            return None
        valid_rows = df[["total_tokens", "latency_ms"]].dropna()
        if valid_rows.empty:
            return None
        tokens_per_sec = (valid_rows["total_tokens"] / valid_rows["latency_ms"] * 1000)
        return tokens_per_sec.mean()
    if all_lat.empty:
        return
    x_min, x_max = all_lat.min(), all_lat.max()
    bins = 30
    # すべてのモデルを1つのグラフに重ねて描画
    # サブプロットでモデルごとに分割
    models = [m for m in model2dfs if "latency_ms" in model2dfs[m] and model2dfs[m]["latency_ms"].notna().any()]
    n = len(models)
    if n == 0:
        return
    import math
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
    if run_name:
        fig.suptitle(f"{run_name}", fontsize=20)
    for idx, model in enumerate(models):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        df = model2dfs[model]
        data = df["latency_ms"].dropna().astype(float)
        color = f"C{idx}"
        # データ数（N）、合計レイテンシ、token/secをタイトルの下、右寄せで表示
        total_latency = data.sum()
        tokens_per_sec = calculate_tokens_per_sec(df)
        info_text = f"N = {len(data)}\nTotal = {total_latency/1000:.1f}s"
        if tokens_per_sec is not None:
            info_text += f"\n{tokens_per_sec:.1f} tokens/sec"
        ax.text(0.98, 0.90, info_text, transform=ax.transAxes,
                va='top', ha='right', fontsize=9, alpha=0.8)

        # ヒストグラム（棒）を描画
        counts, bin_edges = np.histogram(data, bins=bins, range=(x_min, x_max))
        ax.hist(data, bins=bins, range=(x_min, x_max), color=color, alpha=0.7)
        ax.set_xlim(x_min, x_max)
        if log_scale:
            ax.set_xscale('log')
            ax.set_xlabel("ms (log scale)")
        else:
            ax.set_xlabel("ms")
        ax.set_title(f"Latency: {model}")
        ax.set_ylabel("count")

        # 累積分布関数 (CDF) を同一プロットに重ねる（右軸）
        ax2 = ax.twinx()
        total = counts.sum()
        if total > 0:
            cdf = np.cumsum(counts).astype(float) / float(total)
        else:
            cdf = np.zeros_like(counts, dtype=float)
        # ビンの中心を x 座標に使う
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        ax2.plot(bin_centers, cdf, color='k', linestyle='--', marker='o', markersize=3, alpha=0.9)
        ax2.set_ylabel('CDF')
        ax2.set_ylim(0.0, 1.0)
        # x 軸のスケールはメイン軸に合わせる
        if log_scale:
            ax2.set_xscale('log')
        # --- scores の accuracy を右軸（0-1）で表示（点線） ---
        acc = None
        if "is_correct" in df and df["is_correct"].notna().any():
            try:
                acc = float(df["is_correct"].dropna().astype(float).mean())
            except Exception:
                acc = None
        if acc is not None:
            # 水平方向に点線で表示（CDF軸に合わせて 0-1 の範囲）
            ax2.hlines(acc, xmin=x_min, xmax=x_max, colors=color, linestyles='--', linewidth=1.8, alpha=0.9)
            # 右上に注釈を追加
            try:
                # テキストを点線の上に表示する：xはプロット右端（axes fraction）、yはデータ座標 (acc)
                # 少しだけ上にずらすためにoffset pointsを使用
                ax2.annotate(f"Accuracy (mc): {acc:.3f}", xy=(0.98, acc), xycoords=('axes fraction', 'data'),
                             xytext=(0, 4), textcoords='offset points', va='bottom', ha='right',
                             color=color, fontsize=9)
            except Exception:
                pass
    # 空きサブプロットを非表示
    for idx in range(n, nrows*ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, get_output_filename("latency_hist_subplots", run_name)))
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="run_nameごとにmodel_name別でグラフ化")
    parser.add_argument("run_name", help="集計したいrun_name (例: sambanova)")
    parser.add_argument("--out_dir", default="multi_run_plots", help="出力ディレクトリ")
    parser.add_argument("--log_scale", action="store_true", help="latencyヒストグラムの横軸をlogスケールにする")
    parser.add_argument("--no-log_scale", dest="log_scale", action="store_false", help="latencyヒストグラムの横軸を線形スケールにする")
    parser.set_defaults(log_scale=True)
    args = parser.parse_args()

    model2dfs = load_results_by_run(args.run_name)
    if not model2dfs:
        print(f"No results found for run_name={args.run_name}")
        exit(1)
    plot_scores(model2dfs, args.out_dir, run_name=args.run_name)
    plot_latency(model2dfs, args.out_dir, log_scale=args.log_scale, run_name=args.run_name)
    print(f"Done. Plots saved in {args.out_dir}")
