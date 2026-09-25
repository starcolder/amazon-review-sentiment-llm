# imports
import os
import pandas as pd
from split import test_df
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
# loading data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

def load_results():
    qwen_path = os.path.join(BASE_DIR, "testset_results/final_qwen.csv")
    phi_path = os.path.join(BASE_DIR, "testset_results/final_phi.csv")

    qwen_df = pd.read_csv(qwen_path)
    phi_df = pd.read_csv(phi_path)

    return qwen_df, phi_df

# normalizing values and syncing indexes
qwen_results, phi_results = load_results()
qwen_df = pd.DataFrame(qwen_results)
phi_df = pd.DataFrame(phi_results)
test_df = test_df["label"]

def normalize(model_df, test): 
    test = pd.DataFrame(test)
    test = test.reset_index(drop=True)
    mask = (model_df["prediction"] != "unknown") & (model_df["prediction"] != "error")
    model_df = model_df[mask].reset_index(drop=True)
    test = test[mask].reset_index(drop=True)
    model_df["prediction"] = model_df["prediction"].map({
        "positive": 1,
        "negative": 0
    })
    return model_df, test

# evaluating model
def get_metrics(model):

    if model == "qwen":
        model_df = qwen_df.copy()
    elif model == "phi":
        model_df = phi_df.copy()
    else:
        raise ValueError("model must be qwen or phi")

    y_pred, y_true = normalize(model_df, test_df.copy())

    y_true = y_true["label"]
    y_pred = y_pred["prediction"]

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }


def report(model):
    metrics = get_metrics(model)
    accuracy = metrics["accuracy"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]
    classificationReport = classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
    confusionMatrix = confusion_matrix(
        y_true,
        y_pred
    )
    print(f"--------------------------evaluation report for {model}---------------------------")
    print(f"evaluation report for {model}")
    print(f"accuracy: {accuracy}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1: {f1}")
    print("--------------------------model classification report--------------------------")
    print(classificationReport)
    print("-------------------------------confusion matrix--------------------------------")
    print(confusionMatrix)


# charts 
def compare_models_chart():
    qwen_metrics = get_metrics("qwen")
    phi_metrics = get_metrics("phi")
    metrics = ["accuracy", "precision", "recall", "f1"]
    qwen_values = [qwen_metrics[m] for m in metrics]
    phi_values = [phi_metrics[m] for m in metrics]
    x = np.arange(len(metrics))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, qwen_values, width, label="Qwen")
    plt.bar(x + width/2, phi_values, width, label="Phi")
    plt.xticks(x, [m.capitalize() for m in metrics])
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("Qwen vs Phi Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(CHARTS_DIR, f"model_comparison.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

def single_model_chart(model):

    metrics = get_metrics(model)

    names = ["accuracy", "precision", "recall", "f1"]

    values = [
        float(metrics["accuracy"]),
        float(metrics["precision"]),
        float(metrics["recall"]),
        float(metrics["f1"])
    ]

    plt.figure(figsize=(6, 4))

    bars = plt.bar(names, values)

    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(f"{model.upper()} Evaluation Metrics")

    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            h + 0.01,
            f"{h:.3f}",
            ha="center"
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(CHARTS_DIR, f"{model}chart.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    
def confusion_matrix_chart(model):
    metrics = get_metrics(model)
    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]
    plt.figure(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred
    )
    plt.title(f"{model.upper()} Confusion Matrix")
    filename = os.path.join(
        CHARTS_DIR,
        f"{model}_confusion_matrix.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()