
#!/usr/bin/env python3
"""custom_metrics.py
Collect metrics from AGA persona‑level JSON logs (like 1.json).

Usage:
    python custom_metrics.py <log_glob> [--policy POLICY.txt]

Each JSON file must look like:
{
  "meta": {"curr_time": "..."},
  "persona": {
    "Isabella Rodriguez": {
       "description": "...",
       "pronunciatio": "🧠🚫->☕",
       ...
    },
    ...
  }
}

Metrics computed (per entire dataset):
  avg_tokens_per_turn  : approximate LLM tokens for each persona description
                         (simple space split, or tiktoken if installed)
  unique_action_count  : unique 'pronunciatio' strings
  novel_action_rate    : % of unique actions not present in POLICY list
"""

import argparse, glob, json, statistics, re, pathlib, importlib.util, sys, typing as T
import tiktoken

# --- optional: tiktoken for real token counts ---
def get_token_counter():
    try:
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return lambda s: len(enc.encode(s))
    except Exception:
        return lambda s: len(s.split())   # fallback: word count

token_len = get_token_counter()

def load_policy(path:str)->set[str]:
    if not path: return set()
    with open(path, 'r', encoding='utf-8') as f:
        return {ln.strip() for ln in f if ln.strip()}

def iter_logs(glob_pattern:str):
    for fp in glob.glob(glob_pattern):
        with open(fp,'r',encoding='utf-8') as f:
            yield json.load(f)

def analyse(logs:T.Iterable[dict], policy:set[str]):
    token_list=[]
    pron_set=set()
    for obj in logs:
        for p,info in obj.get("persona",{}).items():
            desc = info.get("description","")
            if desc:
                token_list.append(token_len(desc))
            pron = info.get("pronunciatio")
            if pron:
                pron_set.add(pron)
    total_turns=len(token_list)
    avg_tokens = statistics.mean(token_list) if total_turns else 0
    uniq_cnt = len(pron_set)
    novel_rate=0
    if uniq_cnt and policy:
        novel = pron_set - policy
        novel_rate = len(novel)/uniq_cnt*100
    return {
        "avg_tokens_per_turn": round(avg_tokens,2),
        "unique_action_count": uniq_cnt,
        "novel_action_rate_%": round(novel_rate,2)
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("glob", help="File glob, e.g. './logs/*.json'")
    ap.add_argument("--policy", help="Policy TXT file with one pronunciatio per line")
    args=ap.parse_args()
    policy=load_policy(args.policy) if args.policy else set()
    logs=list(iter_logs(args.glob))
    res=analyse(logs, policy)
    print("\n===== Metrics =====")
    for k,v in res.items(): print(f"{k:22}: {v}")
    if not args.policy:
        print("\n(novel_action_rate requires --policy)")
if __name__=='__main__':
    main()
