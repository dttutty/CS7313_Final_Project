import wandb
import pandas as pd

ENTITY = "replace_this_with_your_wandb_entity"
PROJECT = "DyGFormer-LinkPrediction"
RUN_SHORT_ID = "replace_this_with_run_id"  

path = f"{ENTITY}/{PROJECT}/{RUN_SHORT_ID}"

wandb.login()  

api = wandb.Api()
run = api.run(path)

df = run.history(samples=200)

print("Columns in history:")
for c in df.columns:
    print(c)
