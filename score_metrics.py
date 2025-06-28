
#!/usr/bin/env python3
"""
score_metrics.py  — Quick‑n‑dirty log analyser for AGA+SCORE experiments
---------------------------------------------------------------
Input  : JSONL log produced by AGA. One JSON object per turn, e.g.
         {
           "turn": 17,
           "tokens_prompt": 192,
           "tokens_completion": 54,
           "action": "move_to(cafe)",
           "policy_name": "LifestylePolicy",
           "policy_hit": true,
           "contradict": false
         }

Optional: A text file with one action per line (the 'policy' catalogue)
          to compute “novel action” rate.

Output : Prints the six key metrics:
           - avg_tokens_per_turn
           - naturalness (placeholder, needs GPT eval)
           - contras_rate
           - policy_hit_rate
           - unique_action_count
           - novel_action_rate
"""

import argparse, json, sys, pathlib, statistics

def load_log(path):
    with open(path, 'r', encoding='utf‑8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def load_policy(path):
    if not path:
        return set()
    with open(path, 'r', encoding='utf‑8') as f:
        return {ln.strip() for ln in f if ln.strip()}

def analyse(log_iter, policy_set):
    tokens_per_turn = []
    contradict_flags = []
    policy_hits = []
    actions = []

    for row in log_iter:
        tp = row.get('tokens_prompt', 0) + row.get('tokens_completion', 0)
        tokens_per_turn.append(tp)
        contradict_flags.append(bool(row.get('contradict', False)))
        policy_hits.append(bool(row.get('policy_hit', False)))
        actions.append(row.get('action', ''))

    total_turns = len(tokens_per_turn)
    avg_tokens = statistics.mean(tokens_per_turn) if total_turns else 0
    contra_rate = sum(contradict_flags)/total_turns if total_turns else 0
    hit_rate = sum(policy_hits)/total_turns if total_turns else 0

    unique_actions = set(a for a in actions if a)
    unique_count = len(unique_actions)

    novel_actions = unique_actions - policy_set if policy_set else set()
    novel_rate = (len(novel_actions)/unique_count) if unique_count else 0

    return {
        "avg_tokens_per_turn": round(avg_tokens, 2),
        "contradiction_rate": round(contra_rate*100, 2),
        "policy_hit_rate": round(hit_rate*100, 2),
        "unique_action_count": unique_count,
        "novel_action_rate": round(novel_rate*100, 2),
        "naturalness": "TODO (requires GPT batch eval)"
    }

def main():
    ap = argparse.ArgumentParser(description="Compute SCORE metrics from AGA logs")
    ap.add_argument("log", help="Path to JSONL log file")
    ap.add_argument("--policy", help="TXT file with one canonical action per line")
    args = ap.parse_args()

    log_iter = list(load_log(args.log))
    policy_set = load_policy(args.policy) if args.policy else set()
    result = analyse(log_iter, policy_set)

    print("===== SCORE Metrics =====")
    for k,v in result.items():
        print(f"{k:24}: {v}")

if __name__ == "__main__":
    main()
