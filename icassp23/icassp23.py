def load_dataframe(folds):
    "Load DataFrame corresponding to the entire dataset (100k drum sounds)."
    fold_dfs = {}
    for fold in folds:
        csv_name = fold + "_param_log_v2.csv"
        csv_path = os.path.join("data", csv_name)
        fold_df = pd.read_csv(csv_path)
        fold_dfs[fold] = fold_df

    full_df = pd.concat(fold_dfs.values())
    full_df = full_df.sort_values(by="ID", ignore_index=False)
    assert len(set(full_df["ID"])) == len(full_df)
    return full_df
