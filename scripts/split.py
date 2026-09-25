import os
import pandas as pd
from sklearn.model_selection import train_test_split
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(BASE_DIR, "data", "dataprocessed_reviews.csv")
df = pd.read_csv(path)
df = pd.DataFrame(df)
dev_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# print(f"Length of development set: {len(dev_df)}")
# print(f"Length of test set: {len(test_df)}")