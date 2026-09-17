from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "E_Commerce_Dataset.csv"

def load_data():
     df = pd.read_csv(DATA_PATH)
     return df 

def main():
    df = load_data()
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    print(df["Churn"].value_counts())

if __name__ == "__main__":
    main()