import json
import pandas as pd
import os

if not os.path.exists('outputs/logs/rl_history.jsonl'):
    print("Log file not found!")
else:
    history = []
    with open('outputs/logs/rl_history.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                history.append(json.loads(line))

    df = pd.DataFrame(history)

    # Calculate some aggregate metrics
    recent_df = df.tail(50)

    print(f"Total Cycles: {len(df)}")
    print(f"Recent Stats (Last 50 cycles):")
    # Using simple mean for rates to avoid division by zero issues
    avg_sany_rate = (recent_df['sany_pass'] / recent_df['specs_generated']).replace([float('inf'), -float('inf')], 0).fillna(0).mean()
    avg_tlc_rate = (recent_df['tlc_pass'] / recent_df['specs_generated']).replace([float('inf'), -float('inf')], 0).fillna(0).mean()

    print(f"  Avg specs_generated: {recent_df['specs_generated'].mean():.2f}")
    print(f"  Avg sany_pass: {recent_df['sany_pass'].mean():.2f} ({avg_sany_rate*100:.1f}%)")
    print(f"  Avg tlc_pass: {recent_df['tlc_pass'].mean():.2f} ({avg_tlc_rate*100:.1f}%)")
    print(f"  Avg gold_count: {recent_df['gold_count'].mean():.2f}")
    print(f"  Avg new_train_examples: {recent_df['new_train_examples'].mean():.2f}")
    print(f"  Avg new_dpo_pairs: {recent_df['new_dpo_pairs'].mean():.2f}")
    if 'benchmark_sany_rate' in recent_df.columns:
        print(f"  Avg benchmark_sany_rate: {recent_df['benchmark_sany_rate'].mean()*100:.1f}%")
        print(f"  Avg benchmark_tlc_rate: {recent_df['benchmark_tlc_rate'].mean()*100:.1f}%")

    retrained_indices = df[df['retrained'] == True].index
    print("\nRetraining events (last 5):")
    for idx in retrained_indices[-5:]:
        row = df.loc[idx]
        print(f"  Cycle {row['cycle_id']}: {row['timestamp']} (Retrained: {row['retrained']}, Deployed: {row['deployed']})")

    errors = df[df['error'] != ""]
    if not errors.empty:
        print(f"\nRecent Errors: {len(errors)}")
        print(errors[['cycle_id', 'error']].tail())
    else:
        print("\nNo errors recorded in history.")
