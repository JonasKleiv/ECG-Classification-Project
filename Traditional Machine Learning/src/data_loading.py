import numpy as np
import os
import pandas as pd
import ast
import wfdb
from tqdm import tqdm
import concurrent.futures
import gc
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# Helper functions for loading ECG records
def load_single_record(file_path):
    signal, meta = wfdb.rdsamp(file_path)
    return signal

# Load ECG records from cache or raw files
def load_ecg_records(cache_x_path, cache_y_path, PATH, SAMPLING_RATE=100, FORCE_RELOAD=False):
    if os.path.exists(cache_x_path) and os.path.exists(cache_y_path) and not FORCE_RELOAD:
        Y = pd.read_pickle(cache_y_path)
        print("Cached data found. Number of records:", len(Y))
        return Y
    else:
        if not FORCE_RELOAD:
            print("Cache not found. Loading from raw records...")
        else:
            print("Force reloading raw records...")

        # Load metadata for ECG records
        Y = pd.read_csv(PATH + "ptbxl_database.csv", index_col="ecg_id")
        Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)

        # Determine file paths based on the sampling rate
        if SAMPLING_RATE == 100:
            file_paths = [PATH + f for f in Y.filename_lr]
        else:
            file_paths = [PATH + f for f in Y.filename_hr]

        # Multithreaded loading of ECG records
        X_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(tqdm(executor.map(load_single_record, file_paths), total=len(file_paths)))

        # Collect results from multithreaded loading and convert to numpy array
        X_list.extend(results)
        X = np.array(X_list)

        # Save loaded data to cache for further use
        np.save(cache_x_path, X)            # Numpy array
        pd.to_pickle(Y, cache_y_path)       # Pandas pickle

        print("Loaded from raw records. Number of records:", X.shape[0])

        del X_list
        del X
        gc.collect()
        return Y


def prepare_ptbxl_labels(Y_df, X_features_df, scp_statements_path):
    """
    Map raw PTB-XL SCP codes to the 5 diagnostic superclasses (NORM, MI, STTC, CD, HYP),
    filter out records without a diagnostic superclass ('UNKNOWN'), and return aligned
    single-label and multi-label targets along with patient IDs and official PTB-XL strat_folds.
    """
    Y = Y_df.copy()

    # If scp_codes are still strings, parse them into dicts
    if isinstance(Y["scp_codes"].iloc[0], str):
        Y["scp_codes"] = Y["scp_codes"].apply(ast.literal_eval)
    
    # Load official SCP statement lookup table and keep only diagnostic statements
    agg_df = pd.read_csv(scp_statements_path, index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    # Map each patient's SCP codes to unique diagnostic superclasses
    def aggregate_diagnostic(y_dic):
        superclasses = []
        for key in y_dic.keys():
            if key in agg_df.index:
                superclass = agg_df.loc[key, "diagnostic_class"]
                if superclass not in superclasses:
                    superclasses.append(superclass)
        return superclasses
    
    Y["diagnostic_superclass"] = Y["scp_codes"].apply(aggregate_diagnostic)

    Y["single_label"] = Y["diagnostic_superclass"].apply(
        lambda labels: labels[0] if len(labels) > 0 else "UNKNOWN"
    )

    # Keep only the records that have a valid diagnostic superclass
    valid_mask = Y["single_label"] != "UNKNOWN"
    Y_valid = Y.loc[valid_mask].reset_index(drop=True)
    X_valid = X_features_df.loc[valid_mask.values].reset_index(drop=True)

    patient_ids = Y_valid["patient_id"].to_numpy()
    strat_folds = Y_valid["strat_fold"].to_numpy()

    # 1. Single-label integer encoding (0 to 4)
    label_encoder = LabelEncoder()
    y_single_encoded = label_encoder.fit_transform(Y_valid["single_label"])
    
    # 2. Multi-label binary matrix (N x 5)
    mlb = MultiLabelBinarizer()
    y_multi_binarized = mlb.fit_transform(Y_valid["diagnostic_superclass"])
    
    return X_valid, Y_valid, y_single_encoded, y_multi_binarized, patient_ids, strat_folds, label_encoder, mlb
