import pandas as pd
import ollama
import time
from split import test_df
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# a function for choosing between models:
def choose_model():
    answer = input(f"""choose the model you want to run inferece with:
1. qwen.
2. phi.
enter the number you want to choose(1/2): """)
    if answer == "1":
        model = "myqwen"
        backup_path = os.path.join(BASE_DIR, "testset_results", "backup_qwen.csv")
        final_path = os.path.join(BASE_DIR, "testset_results", "final_qwen.csv")
    elif answer == "2":
        model = "myphi"
        backup_path = os.path.join(BASE_DIR, "testset_results", "backup_phi.csv")
        final_path = os.path.join(BASE_DIR, "testset_results", "final_phi.csv")
    return model, backup_path, final_path


options = {
    "num_thread": 12,
    # "temperature": 0,
    # "num_ctx": 256
}
MODEL, BACKUP_PATH, FINAL_PATH = choose_model()
# load backup if exists
try:
    backup = pd.read_csv(BACKUP_PATH)
    predictions = backup["prediction"].tolist()
    start_idx = len(predictions)
    print(f"Resuming from {start_idx}")
except:
    predictions = []
    start_idx = 0
    print("Starting fresh")

start_time = time.time()
for i in range(start_idx, len(test_df)):

    text = str(test_df.iloc[i]["text"])[:300]
    prompt = f"""Classify sentiment as positive or negative
Return ONLY one word
Text: {text}
Sentiment:"""

    try:
        response = ollama.generate(
            model=MODEL,
            prompt=prompt,
            options=options
        )

        pred = response["response"].strip().lower()
        
        if any(x in pred for x in ["positive", "pos"]):
            pred = "positive"

        elif any(x in pred for x in ["negative", "neg"]):
            pred = "negative"

        else:
            pred = "unknown"

        predictions.append(pred)

    except Exception as e:
        print(f"error {i}: {e}")
        predictions.append("error")
        time.sleep(1)

    # checkpoint every 100
    if i % 100 == 0:
        pd.DataFrame({
            "prediction": predictions
        }).to_csv(BACKUP_PATH, index=False)
        
    elapsed = time.time() - start_time
    speed = (i + 1) / elapsed
    print(
        f"\rprocessed: {i+1} out of {len(test_df)} | speed: {speed:.2f} records/sec",
        end="",
        flush=True
    )


# final save
pd.DataFrame({
    "prediction": predictions
}).to_csv(FINAL_PATH, index=False)

os.remove(BACKUP_PATH)

