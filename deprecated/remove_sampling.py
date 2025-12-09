import pandas as pd

df = pd.read_csv("summary_dygformer.csv")

df = df[~df["sampling"].isin(["historical", "inductive"])]

df = df.drop(columns=["sampling"])

df.to_csv("summary_dygformer_filtered.csv", index=False)

print("Done! Generated summary_dygformer_filtered.csv")
