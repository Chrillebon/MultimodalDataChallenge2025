import pandas as pd
import re
import seaborn as sns

def load_class_results(txt_path):
    """
    Parse the class-wise results from your validation log into a Pandas DataFrame.

    Parameters
    ----------
    txt_path : str
        Path to the text file containing the validation report.

    Returns
    -------
    df : pd.DataFrame
        Columns: [Class, Count, Correct, Accuracy]
    """
    with open(txt_path, "r") as f:
        lines = f.readlines()

    # Find the section between "Class-wise Results:" and the summary
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Class-wise Results"):
            start_idx = i
        if line.strip().startswith("Summary Statistics"):
            end_idx = i
            break

    if start_idx is None or end_idx is None:
        raise ValueError("Could not find Class-wise Results section in file.")

    # Pattern to capture: Class, Count, Correct, Accuracy
    pattern = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d\.]+)")

    records = []
    for line in lines[start_idx:end_idx]:
        m = pattern.match(line)
        if m:
            cls, count, correct, acc = m.groups()
            records.append({
                "Class": int(cls),
                "Count": int(count),
                "Correct": int(correct),
                "Accuracy": float(acc)
            })

    df = pd.DataFrame(records)
    return df

# Example usage:
if __name__ == "__main__":
    df = load_class_results("../lookup/10sample_validation_classwise.txt")
    print(df.head())
    # Now you can sort, filter, etc.
    worst = df.sort_values("Accuracy", ascending=True)
    
    sns.scatterplot(
        data = worst,
        x="Count",
        y = "Accuracy"
    )

    print("\nWorst 10 classes:\n", worst)