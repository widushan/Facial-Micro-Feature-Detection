
import pandas as pd
import os
from run_pipeline import perform_pca_selection, train_model

if __name__ == "__main__":
    if os.path.exists("all_features.csv"):
        print("Loading from all_features.csv...")
        df = pd.read_csv("all_features.csv")
        df_selected, selected_features = perform_pca_selection(df)
        train_model(df_selected, selected_features)
    else:
        print("all_features.csv not found, please run full pipeline.")
