# IgakuQA Eval

医学系問題のQA評価ツール。各種LLMの回答精度、レイテンシ、エラー分析を行い、結果を可視化します。

## 機能

- 複数選択式（Multiple Choice）および自由回答形式の問題に対応
- BLEUスコアとROUGE-Lによるテキスト評価
- OpenAI互換APIを使用したLLMの評価
- 複数のトークナイザ（MeCab/fugashi, janome, sudachipy）による日本語処理
- 包括的な可視化と分析レポート生成

### 可視化機能

- スコア分析
  - 正解率（Accuracy）
  - BLEUスコア
  - ROUGE-Lスコア
- レイテンシ分析
  - ヒストグラム
  - CDFプロット
  - モデルごとのサブプロット表示
- エラー分析
  - エラータイプの分布
  - エラーメッセージの頻度分析
- 年度別分析
  - スコアの時系列推移

## セットアップ

### 必要条件

- Python 3.8以上
- OpenAI SDK v1.0以上
- 必要なPythonパッケージ:
  ```bash
  pip install -r requirements.txt
  ```

### 環境変数の設定

`.env`ファイルを作成し、以下の環境変数を設定:

```ini
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://your-api-endpoint  # オプション
```

または、環境変数として直接設定:

```bash
export OPENAI_API_KEY='your-api-key'
export OPENAI_BASE_URL='https://your-api-endpoint'  # オプション
```

## 使い方

### 評価の実行

1. 設定ファイル（`config.yaml`）の準備:

```yaml
run_name: experiment_name
dataset:
  source: jsonl  # または 'hf' (HuggingFace)
  path: path/to/questions.jsonl
openai:
  model: gpt-4
  temperature: 0.0
  max_tokens: 1000  # オプション
prompt:
  mode: auto  # 'mc' または 'free' も指定可能
  system: ""  # システムプロンプト
```

2. 評価の実行:

```bash
python3 igakuqa_eval.py run_eval --config config.yaml
```

### 結果の可視化

複数のモデル/実行結果を比較:

```bash
python3 plot_multi_run.py experiment_name --out_dir plots
```

オプション:
- `--log_scale`: レイテンシのプロットを対数スケールで表示（デフォルト: True）
- `--no-log_scale`: 線形スケールを使用

## 出力ファイル

評価結果は `runs/<experiment_name>_<timestamp>/` に出力:

- `results.jsonl`: 各問題の評価結果（JSON Lines形式）
- `results.csv`: 評価結果（CSV形式）
- `summary.json`: 評価サマリー
- 可視化プロット:
  - `scores.png`: 全体スコア
  - `latency_hist.png`: レイテンシ分布
  - `scores_subplots.png`: モデルごとのスコア比較
  - `latency_hist_subplots.png`: モデルごとのレイテンシ分布
  - `error_types.png`: エラータイプの分布
  - `error_messages_top.png`: 主なエラーメッセージ
  - `*_by_year.png`: 年度別分析

## 日本語トークナイザ

以下のトークナイザを優先順で使用:
1. fugashi (MeCab)
2. janome
3. sudachipy

インストールされているものを自動的に選択します。

## ライセンス

MITライセンス

## 貢献

Issue、プルリクエストを歓迎します。