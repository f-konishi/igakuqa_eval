import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import typer
import yaml

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# .env support
def load_env_file():
    try:
        from dotenv import load_dotenv, find_dotenv, dotenv_values
        env_file = find_dotenv(usecwd=True)  # カレントディレクトリから.envを検索
        if not env_file:
            print("Warning: .env ファイルが見つかりません")
            return False
        
        # .envファイルの内容を読み取り
        env_values = dotenv_values(env_file)
        if not env_values:
            print(f"Warning: .envファイル {env_file} は空か、読み取れません")
            return False
            
        print(f"Debug: .envファイルの内容:")
        for key in env_values.keys():
            # APIキーは最初の数文字のみ表示
            value = env_values[key]
            if 'API_KEY' in key and value:
                value = value[:8] + '...'
            print(f"  {key}={value}")
        
        # 環境変数として設定
        load_dotenv(env_file, override=True)  # 既存の環境変数より.envを優先
        
        # 設定後の確認
        print(f"\nDebug: 環境変数の設定状態:")
        for key in env_values.keys():
            value = os.getenv(key)
            if 'API_KEY' in key and value:
                value = value[:8] + '...'
            print(f"  {key}={value}")
        
        print(f"\nInfo: 環境変数を {env_file} から読み込みました")
        return True
    except ImportError:
        print("Warning: python-dotenvがインストールされていません。pip install python-dotenvでインストールできます")
        return False
    except Exception as e:
        print(f"Warning: .envファイルの読み込み中にエラーが発生しました: {e}")
        return False

# 起動時に.envを読み込む
load_env_file()

# OpenAI SDK (v1+)
from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError

# Metrics
import sacrebleu
from rouge_score import rouge_scorer

# Option: Hugging Face datasets
try:
    from datasets import load_dataset
    HAS_HF = True
except Exception:
    HAS_HF = False

# ===============
# Utility
# ===============

def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def char_tokens(s: str) -> List[str]:
    # 日本語向けの簡易トークナイズ: 文字単位
    return list(str(s).strip())


def join_as_spaced(tokens: List[str]) -> str:
    return " ".join(tokens)


def try_mecab_tokens(s: str) -> Optional[List[str]]:
    """日本語用のトークナイザを順に試す。

    試行順:
      1. fugashi (MeCab)
      2. janome
      3. sudachipy
    いずれも利用できなければ None を返す。
    戻り値はトークン文字列のリスト。
    """
    if s is None:
        return None

    # 1) fugashi (MeCab)
    try:
        from fugashi import Tagger
        tagger = Tagger()
        return [m.surface for m in tagger(s)]
    except Exception:
        pass

    # 2) janome
    try:
        from janome.tokenizer import Tokenizer as JanomeTokenizer
        jt = JanomeTokenizer()
        return [t.surface for t in jt.tokenize(s)]
    except Exception:
        pass

    # 3) sudachipy
    try:
        from sudachipy import tokenizer as sudachi_tokenizer
        from sudachipy import dictionary as sudachi_dictionary
        tokenizer_obj = sudachi_dictionary.Dictionary().create()
        mode = sudachi_tokenizer.Tokenizer.SplitMode.C
        return [m.surface() for m in tokenizer_obj.tokenize(s, mode)]
    except Exception:
        pass

    return None


# ===============
# Dataset Loader
# ===============

def standardize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    入力のレコードを標準形式に正規化。
    標準形式:
      id, year, question, options(list|None), answer_text|None, answer_idx|None, text_only(bool)|None, has_image(bool)|None
    """
    sid = rec.get("sample_id") or rec.get("id") or rec.get("qid")
    year = rec.get("year")
    q = rec.get("question") or rec.get("prompt")
    options = rec.get("options")

    # 正解（テキスト or インデックス）
    a_text = rec.get("correct_answer") or rec.get("answer_text") or rec.get("reference")
    a_idx = rec.get("answer_idx")

    text_only = rec.get("text_only")
    has_image = rec.get("has_image")

    # ヒューリスティクス
    if has_image is None and text_only is not None:
        has_image = not bool(text_only)

    return {
        "id": sid if sid is not None else str(hash(json.dumps(rec, ensure_ascii=False)) % (10 ** 8)),
        "year": year,
        "question": q,
        "options": options,
        "answer_text": a_text,
        "answer_idx": a_idx,
        "text_only": text_only,
        "has_image": has_image,
    }


def load_igakuqa_from_hf(hf_name: str, split: str) -> List[Dict[str, Any]]:
    if not HAS_HF:
        raise RuntimeError("datasets がインストールされていません。requirements を確認してください。")
    ds = load_dataset(hf_name, split=split)
    recs = [standardize_record(r) for r in ds]
    return recs


def load_from_jsonl(path: str) -> List[Dict[str, Any]]:
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            recs.append(standardize_record(rec))
    return recs


# ===============
# Prompting
# ===============

def build_prompt(rec: Dict[str, Any], mode: str = "auto") -> Tuple[str, str]:
    """returns (user_prompt, parsing_mode) where parsing_mode in {"mc","free"}
    """
    q = rec["question"]
    opts = rec.get("options")

    if mode == "auto":
        mode = "mc" if isinstance(opts, (list, tuple)) and len(opts) > 0 else "free"

    if mode == "mc":
        lines = ["次の設問に答えてください。必ず最終行で 'ANSWER: <番号>' を出力してください。説明はあっても構いません。",
                 f"設問: {q}"]
        if opts:
            for i, o in enumerate(opts):
                lines.append(f"{i}: {o}")
        lines.append("最終行の形式: ANSWER: <番号>")
        return "\n".join(lines), "mc"
    else:
        return f"次の設問に簡潔に答えてください。\n設問: {q}\n最終行の形式: ANSWER: <短い最終回答>", "free"


def parse_answer(text: str, mode: str) -> Tuple[Optional[int], Optional[str]]:
    if text is None:
        return None, None
    last = text.strip().splitlines()[-1]
    if last.startswith("ANSWER:"):
        payload = last.split("ANSWER:", 1)[1].strip()
        if mode == "mc":
            try:
                return int(payload), None
            except Exception:
                # 数字抽出が難しい場合、数字だけ拾う
                import re
                m = re.search(r"(\d+)", payload)
                return (int(m.group(1)) if m else None), None
        else:
            return None, payload
    return None, None


# ===============
# OpenAI Caller
# ===============

def make_openai_client(cfg_openai: Dict[str, Any]) -> OpenAI:
    # configで指定された環境変数を優先的に使用
    api_key_env = cfg_openai.get("api_key_env", "OPENAI_API_KEY")
    base_url_env = cfg_openai.get("base_url_env", "OPENAI_BASE_URL")
    max_retries = cfg_openai.get("max_retries", 2)  # デフォルト: 2回
    timeout_s = cfg_openai.get("timeout_s", 60)  # デフォルト: 60秒
    
    # 環境変数から値を取得（.envファイルからの読み込みは既に完了している）
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"環境変数 {api_key_env} が見つかりません。\n"
            f"以下のいずれかの方法で設定してください：\n"
            f"1. .envファイルに追加:\n"
            f"   {api_key_env}=your-api-key\n"
            f"2. 環境変数として設定:\n"
            f"   export {api_key_env}='your-api-key'\n"
            f"現在の設定:\n"
            f"  設定ファイル: api_key_env = {api_key_env}\n"
            f"  環境変数の値: {api_key}\n"
            f"  .envファイル: {os.path.join(os.getcwd(), '.env')}\n"
        )
    
    base_url = os.getenv(base_url_env)
    if base_url_env and not base_url:
        print(f"Warning: 環境変数 {base_url_env} が見つかりません。.envファイルに追加するか、環境変数として設定してください。")
        print(f"デフォルトのベースURLを使用します。")
    client = OpenAI(
        api_key=api_key,
        base_url=base_url or None,
        max_retries=max_retries,  # リトライ回数を設定
        timeout=timeout_s,  # タイムアウトをクライアント作成時に設定
    )
    return client


class InferenceError(Exception):
    pass


@retry(reraise=True,
       stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=1, max=8),
       retry=retry_if_exception_type((APIError, RateLimitError, APITimeoutError, InferenceError)))
def query_once(client: OpenAI, model: str, system_prompt: str, user_prompt: str,
               temperature: float = 0.0, seed: Optional[int] = None, timeout_s: int = 60,
               max_tokens: Optional[int] = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        # build params so we only include max_tokens when provided
        params = dict(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            seed=seed,
        )
        if max_tokens is not None:
            params["max_tokens"] = int(max_tokens)

        # Chat Completions API呼び出し
        resp = client.chat.completions.create(**params)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        
        # 応答の取り出し
        content = None
        if resp and resp.choices and len(resp.choices) > 0:
            content = resp.choices[0].message.content
        
        # 使用量情報の取り出し
        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        
        return {
            "content": content,
            "elapsed_ms": elapsed_ms,
            "usage": usage,
            "error": None,
        }
    except (APIError, RateLimitError, APITimeoutError) as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        raise e
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        # tenacity に拾わせる独自例外
        raise InferenceError(str(e))


# ===============
# Scoring
# ===============

def score_bleu(refs: List[str], hyps: List[str]) -> float:
    # 文字単位（or MeCab）で空白区切りにする
    tok_refs = []
    tok_hyps = []
    for r, h in zip(refs, hyps):
        rt = try_mecab_tokens(r) or char_tokens(r)
        ht = try_mecab_tokens(h) or char_tokens(h)
        tok_refs.append(join_as_spaced(rt))
        tok_hyps.append(join_as_spaced(ht))
    bleu = sacrebleu.corpus_bleu(tok_hyps, [tok_refs], tokenize="none")
    return float(bleu.score/100)


def score_rouge_l(refs: List[str], hyps: List[str]) -> float:
    # rouge-score は独自トークナイザ。ここでは空白区切り文字列を渡す
    tok_refs = []
    tok_hyps = []
    for r, h in zip(refs, hyps):
        rt = try_mecab_tokens(r) or char_tokens(r)
        ht = try_mecab_tokens(h) or char_tokens(h)
        tok_refs.append(join_as_spaced(rt))
        tok_hyps.append(join_as_spaced(ht))

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(tok_refs, tok_hyps)]
    return float(np.mean(scores))


# ===============
# Runner
# ===============

def run(cfg_path: str = "config.yaml") -> None:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_name = cfg.get("run_name", now_ts())
    out_dir = os.path.join(cfg.get("io", {}).get("out_dir", "runs"), f"{run_name}_{now_ts()}")
    ensure_dir(out_dir)

    ds_cfg = cfg.get("dataset", {})
    if ds_cfg.get("source") == "hf":
        recs = load_igakuqa_from_hf(ds_cfg.get("hf_name"), ds_cfg.get("split", "train"))
    elif ds_cfg.get("source") == "jsonl":
        recs = load_from_jsonl(ds_cfg.get("path"))
    else:
        raise RuntimeError("dataset.source は 'hf' か 'jsonl' を指定してください")

    if ds_cfg.get("text_only", False):
        recs = [r for r in recs if r.get("has_image") in (False, None)]

    limit = ds_cfg.get("limit")
    if isinstance(limit, int) and limit > 0:
        recs = recs[:limit]

    client = make_openai_client(cfg.get("openai", {}))
    model = cfg.get("openai", {}).get("model")
    temperature = cfg.get("openai", {}).get("temperature", 0.0)
    seed = cfg.get("openai", {}).get("seed")
    timeout_s = cfg.get("openai", {}).get("timeout_s", 60)
    max_tokens = cfg.get("openai", {}).get("max_tokens")

    prompt_mode = cfg.get("prompt", {}).get("mode", "auto")
    system_prompt = cfg.get("prompt", {}).get("system", "")

    save_jsonl = cfg.get("io", {}).get("save_jsonl", True)
    save_csv = cfg.get("io", {}).get("save_csv", True)

    rows = []
    jsonl_path = os.path.join(out_dir, "results.jsonl")

    for rec in tqdm(recs, desc="Evaluating"):
        if cfg.get("evaluation", {}).get("skip_image_questions", True) and rec.get("has_image"):
            continue
        user_prompt, pmode = build_prompt(rec, prompt_mode)

        error_type = None
        error_msg = None
        content = None
        usage = {}
        elapsed_ms = None

        try:
            res = query_once(client, model, system_prompt, user_prompt, temperature, seed, timeout_s, max_tokens)
            content = res["content"]
            usage = res.get("usage", {})
            elapsed_ms = res.get("elapsed_ms")
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)[:500]

        pred_idx, pred_text = parse_answer(content, pmode)

        # If MC mode produced an index but no text, try to map it back to the options
        if pmode == "mc" and pred_text is None and pred_idx is not None:
            try:
                opts = rec.get("options")
                if isinstance(opts, (list, tuple)):
                    idx = int(pred_idx)
                    if 0 <= idx < len(opts):
                        pred_text = opts[idx]
            except Exception:
                # ignore mapping errors and leave pred_text as-is
                pass

        # 正解比較
        gold_idx = rec.get("answer_idx")
        gold_text = rec.get("answer_text")

        is_correct = None
        bleu = None
        rougeL = None

        if pmode == "mc" and gold_idx is not None and pred_idx is not None:
            is_correct = int(pred_idx == int(gold_idx))
        # Compute BLEU / ROUGE when gold_text and pred_text are available.
        # Respect config flags if present.
        eval_cfg = cfg.get("evaluation", {}) if cfg else {}
        compute_bleu_flag = eval_cfg.get("compute_bleu", True)
        compute_rouge_flag = eval_cfg.get("compute_rougeL", True)
        if gold_text and pred_text:
            if compute_bleu_flag:
                try:
                    bleu = score_bleu([gold_text], [pred_text])
                except Exception:
                    pass
            if compute_rouge_flag:
                try:
                    rougeL = score_rouge_l([gold_text], [pred_text])
                except Exception:
                    pass

        row = {
            "run_name": run_name,
            "model_name": model,
            "id": rec.get("id"),
            "year": rec.get("year"),
            "mode": pmode,
            "question": rec.get("question"),
            "options": rec.get("options"),
            "gold_idx": gold_idx,
            "gold_text": gold_text,
            "pred_idx": pred_idx,
            "pred_text": pred_text,
            "is_correct": is_correct,
            "bleu": bleu,
            "rougeL": rougeL,
            "latency_ms": elapsed_ms,
            "error_type": error_type,
            "error_msg": error_msg,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
        rows.append(row)

        if save_jsonl:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    df = pd.DataFrame(rows)
    if save_csv:
        df.to_csv(os.path.join(out_dir, "results.csv"), index=False)

    # 集計 & 可視化
    summarize_and_plot(df, out_dir, cfg, run_name=run_name, model_name=model)


def summarize_and_plot(df: pd.DataFrame, out_dir: str, cfg: Dict[str, Any], run_name: Optional[str] = None, model_name: Optional[str] = None) -> None:
    if df.empty:
        print("No results to summarize.")
        return

    # 集計
    overall = {}
    # attach run and model identifiers
    overall["run_name"] = run_name
    overall["model_name"] = model_name if model_name is not None else (cfg.get("openai", {}).get("model") if cfg else None)
    if "is_correct" in df:
        acc = df["is_correct"].dropna().astype(float)
        overall["accuracy(mc)"] = float(acc.mean()) if not acc.empty else None
    if "bleu" in df:
        b = df["bleu"].dropna().astype(float)
        overall["BLEU"] = float(b.mean()) if not b.empty else None
    if "rougeL" in df:
        r = df["rougeL"].dropna().astype(float)
        overall["ROUGE-L"] = float(r.mean()) if not r.empty else None

    lat = df["latency_ms"].dropna().astype(float)
    overall["latency_ms_p50"] = float(lat.quantile(0.5)) if not lat.empty else None
    overall["latency_ms_p95"] = float(lat.quantile(0.95)) if not lat.empty else None

    err_rate = float((df["error_type"].notna()).mean()) if not df.empty else None
    overall["error_rate"] = err_rate

    # エラー分析
    error_analysis = {}
    if not df.empty and "error_type" in df and "error_msg" in df:
        # エラータイプの集計
        error_types = df["error_type"].value_counts().to_dict()
        error_analysis["error_types"] = error_types

        # エラーメッセージの頻度分析
        error_msgs = df["error_msg"].value_counts().to_dict()
        error_analysis["error_messages"] = error_msgs

    overall["error_analysis"] = error_analysis
    # attach model/config info (non-sensitive) to summary
    try:
        openai_cfg = cfg.get("openai", {}) if cfg else {}
    except Exception:
        openai_cfg = {}
    model_info = {
        "model_name": openai_cfg.get("model"),
        "max_tokens": openai_cfg.get("max_tokens"),
        "temperature": openai_cfg.get("temperature"),
        "seed": openai_cfg.get("seed"),
        "timeout_s": openai_cfg.get("timeout_s"),
        "api_key_env": openai_cfg.get("api_key_env"),
        "base_url_env": openai_cfg.get("base_url_env"),
    }
    overall["model_info"] = model_info

    # Normalization: map metrics to 0..1 so they are directly comparable
    # - accuracy is already 0..1
    # - BLEU from sacrebleu is 0..100 -> divided by 100
    # - ROUGE-L is 0..1
    norm_map = {}
    if overall.get("accuracy(mc)") is not None:
        norm_map["accuracy(mc)"] = float(overall["accuracy(mc)"])
    if overall.get("BLEU") is not None:
        try:
            norm_map["BLEU"] = float(overall["BLEU"]) 
        except Exception:
            norm_map["BLEU"] = None
    if overall.get("ROUGE-L") is not None:
        norm_map["ROUGE-L"] = float(overall["ROUGE-L"]) if overall.get("ROUGE-L") is not None else None

    # aggregated normalized score (mean of available normalized metrics)
    norm_vals = [v for v in norm_map.values() if v is not None]
  
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        
        json.dump(overall, f, ensure_ascii=False, indent=2)

    # 可視化（PNG 保存）
    # 1) スコア棒グラフ
    try:
        labels = []
        values = []
        for k in ["accuracy(mc)", "BLEU", "ROUGE-L"]:
            if overall.get(k) is not None:
                labels.append(k)
                values.append(overall[k])
        if values:
            plt.figure()
            plt.bar(labels, values)
            model_name = cfg.get("openai", {}).get("model") if cfg else None
            title = f"Overall scores ({model_name})" if model_name else "Overall scores"
            plt.title(title)
            plt.ylabel("score")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "scores.png"))
            plt.close()
    except Exception as e:
        print(f"Scores plotting skipped due to: {e}")

    # 2) レイテンシ分布
    try:
        lat = df["latency_ms"].dropna().astype(float)  # ensure defined here
        if not lat.empty:
            plt.figure()
            plt.hist(lat, bins=30)
            model_name = cfg.get("openai", {}).get("model") if cfg else None
            title = f"Latency (ms) — {model_name}" if model_name else "Latency (ms)"
            plt.title(title)
            plt.xlabel("ms")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "latency_hist.png"))
            plt.close()
    except Exception as e:
        print(f"Latency plotting skipped due to: {e}")

    # 3) 年度別の精度/スコア
    try:
        if "year" in df.columns and df["year"].notna().any():
            by_year = df.copy()
            by_year["year"] = by_year["year"].astype(str)

            def _agg_mean(col):
                s = by_year[col].dropna().astype(float)
                return s.mean() if not s.empty else np.nan

            piv = by_year.groupby("year").agg(
                acc=("is_correct", lambda s: s.dropna().astype(float).mean() if s.dropna().size>0 else np.nan),
                bleu=("bleu", _agg_mean),
                rougeL=("rougeL", _agg_mean),
            )
            piv.to_csv(os.path.join(out_dir, "by_year.csv"))

            cols = [c for c in ["acc", "bleu", "rougeL"] if c in piv.columns]
            for c in cols:
                # drop NaNs before plotting to avoid pandas/matplotlib mask errors
                series = piv[c].dropna()
                if series.empty:
                    continue
                plt.figure()
                # ensure numeric values when possible
                try:
                    series = series.astype(float)
                except Exception:
                    pass
                series.plot(kind="bar")
                plt.title(f"{c} by year")
                plt.ylabel(c)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"{c}_by_year.png"))
                plt.close()
    except Exception as e:
        print(f"By-year plotting skipped due to: {e}")

    # 4) エラータイプの円グラフ
    try:
        if "error_type" in df and df["error_type"].dropna().size > 0:
            plt.figure(figsize=(10, 8))
            error_counts = df["error_type"].dropna().value_counts()
            if not error_counts.empty:
                plt.pie(error_counts.values, labels=error_counts.index, autopct='%1.1f%%')
                model_name = cfg.get("openai", {}).get("model") if cfg else None
                title = f"Error Types Distribution ({model_name})" if model_name else "Error Types Distribution"
                plt.title(title)
                plt.axis('equal')
                plt.savefig(os.path.join(out_dir, "error_types.png"))
                plt.close()
    except Exception as e:
        print(f"Error types plotting skipped due to: {e}")

    # 5) エラーメッセージの頻度分析（上位をバーで表示）
    try:
        if "error_msg" in df and df["error_msg"].dropna().size > 0:
            error_msgs = df["error_msg"].dropna().value_counts()
            if not error_msgs.empty:
                top_n = min(20, len(error_msgs))
                plt.figure(figsize=(10, 6))
                error_msgs.iloc[:top_n].plot(kind='bar')
                plt.title('Top error messages')
                plt.ylabel('count')
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "error_messages_top.png"))
                plt.close()
    except Exception as e:
        print(f"Error messages plotting skipped due to: {e}")


# ===============
# Typer CLI
# ===============
app = typer.Typer(add_completion=False)


@app.command()
def run_eval(config: str = typer.Option("config.yaml", help="設定ファイルパス")):
    """IgakuQA を用いた評価を実行"""
    run(config)


if __name__ == "__main__":
    app()
