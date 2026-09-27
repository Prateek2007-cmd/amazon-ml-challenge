import pandas as pd

ledger_path = "experiments/final_challenger/artifacts/error_ledger.csv"
df = pd.read_csv(ledger_path)

fp = df[df["error_type"] == "FP_FALSE_POSITIVE_MERGE"]
fn = df[df["error_type"] == "FN_MATCHER_REJECTED"]

print(f"\n--- 5 FALSE POSITIVES (FP_FALSE_POSITIVE_MERGE) ---")
for i, row in fp.sort_values("model_probability", ascending=False).head(5).iterrows():
    print(f"Prob={row['model_probability']:.3f} | S1={row['s1_id']} | True={row['true_match_id']} | Pred={row['predicted_ids']}")
    print(f"  Provenance: {row['retrieval_provenance']} | NameSim: {row['name_similarity']:.1f}% | AddrSim: {row['address_similarity']:.1f}% | SemCos: {row['semantic_cosine']:.3f} | Country: {row['country']}")

print(f"\n--- 5 MATCHER FALSE NEGATIVES (FN_MATCHER_REJECTED) ---")
for i, row in fn.sort_values("model_probability", ascending=False).head(5).iterrows():
    print(f"Prob={row['model_probability']:.3f} | S1={row['s1_id']} | True={row['true_match_id']} | Pred={row['predicted_ids']}")
    print(f"  Provenance: {row['retrieval_provenance']} | NameSim: {row['name_similarity']:.1f}% | AddrSim: {row['address_similarity']:.1f}% | SemCos: {row['semantic_cosine']:.3f} | Country: {row['country']}")

print(f"\n--- 5 HARD MATCHER FALSE NEGATIVES (Prob < 0.20) ---")
for i, row in fn[fn['model_probability'] < 0.20].head(5).iterrows():
    print(f"Prob={row['model_probability']:.3f} | S1={row['s1_id']} | True={row['true_match_id']} | Pred={row['predicted_ids']}")
    print(f"  Provenance: {row['retrieval_provenance']} | NameSim: {row['name_similarity']:.1f}% | AddrSim: {row['address_similarity']:.1f}% | SemCos: {row['semantic_cosine']:.3f} | Country: {row['country']}")
