import eval
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def print_menu():
    print("\n==============================")
    print("     SENTIMENT EVALUATOR")
    print("==============================")
    print("1. Evaluate Qwen")
    print("2. Evaluate Phi")
    print("3. Compare Models (Charts)")
    print("4. Confusion Matrix (Qwen)")
    print("5. Confusion Matrix (Phi)")
    print("6. Single Model Charts")
    print("7. Exit")
    print("==============================\n")


def ensure_charts_dir():
    os.makedirs("charts", exist_ok=True)


def run_evaluation(model):
    eval.report(model)


def run_comparison():
    eval.compare_models_chart()
    print("✔ Comparison chart saved in charts/")


def run_confusion(model):
    eval.confusion_matrix_chart(model)
    print(f"✔ Confusion matrix saved for {model}")


def run_single_chart(model):
    eval.single_model_chart(model)
    print(f"✔ Metrics chart saved for {model}")


def main():
    ensure_charts_dir()

    while True:
        print_menu()
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            run_evaluation("qwen")

        elif choice == "2":
            run_evaluation("phi")

        elif choice == "3":
            run_comparison()

        elif choice == "4":
            run_confusion("qwen")

        elif choice == "5":
            run_confusion("phi")

        elif choice == "6":
            model = input("Which model? (qwen/phi): ").strip().lower()
            if model in ["qwen", "phi"]:
                run_single_chart(model)
            else:
                print("❌ Invalid model name")

        elif choice == "7":
            print("Exiting program...")
            break

        else:
            print("❌ Invalid choice, try again")


if __name__ == "__main__":
    main()