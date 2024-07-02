# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import os

from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss, recall_score, precision_score, f1_score
from flwr.common.logger import log
from logging import INFO

datapath = os.getenv("FL_DATAPATH")

### Load datasets
def _unsw_load(unsw_path):
    log(INFO,f"Loading data from {unsw_path}")
    header = 0
    encoding ='utf-8'
    unsw = pd.read_csv(unsw_path+"/generatedflows/unsw.csv", header=header, low_memory=False, sep=',', encoding=encoding)
    return unsw

def _cicids_load(cic_path):
    log(INFO,f"Loading data from {cic_path} ...")
    header = 0
    encoding = 'utf-8'
    cicids = pd.read_csv(cic_path+"/cicids.csv", header=header, low_memory=False, sep=',', encoding=encoding)
    # Before any other tasks, the columns will be renamed in order to delete the initial blank spaces. 
    cicids.columns = [feat.lstrip(' ') for feat in cicids.columns]
    cicids.columns
    return cicids

### Clean datasets
def _unsw_cleaning(unsw):
    log(INFO,"Cleaning UNSW-NB15 dataset ...")
## Drop unnecessary features
    unsw_rmv = ["attack_cat", "ct_flw_http_mthd","is_ftp_login", "ct_ftp_cmd" ]
    unsw = unsw.drop(columns= unsw_rmv)
    return unsw
def _cicids_cleaning(cicids):
    log(INFO,"Cleaning CIC-IDS2017 dataset ...")
    ## Drop unnecesary features
    cicids_rmv = ["Fwd Header Length.1","Fwd Header Length2", "Flow ID"]
    if "Fwd Header Length2" in cicids.columns:
        cicids_rmv = ["Fwd Header Length.1","Fwd Header Length2", "Flow ID"]
    else:
        cicids_rmv = ["Fwd Header Length.1", "Flow ID"]
    # Select columns for the fusion from each df
    cicids = cicids.drop(columns= cicids_rmv)
    # Drop rows where all features are NaN
    cicids.dropna(how='all', axis=0 , inplace=True)
    cicids["Flow Bytes/s"] = cicids["Flow Bytes/s"].fillna(0)
    return cicids

## Preprocessing
def _unsw_preprocessing(unsw_X):
    log(INFO, f"Preprocessing UNSW-NB15 dataset...")
    ##(sload + dload)/8 -> cicids["Flow bytes/s"]
    unsw_X["Sload"] = unsw_X["Sload"].astype(float) + unsw_X["Dload"].astype(float)
    unsw_X.rename(columns = {"Sload":"flowbytesps"}, inplace=True)
    unsw_X.flowbytesps.map(lambda x: x/8)
    unsw_X.drop(columns="Dload", inplace = True)
    log(INFO,"\tTime encoding ...")    

    ## Convert epoch time to datetime
    unsw_X["Stime"] = pd.to_datetime(unsw_X["Stime"], unit='s')
    # Extract time component and convert to numeric representation
    unsw_X["Stime"] = unsw_X["Stime"].dt.hour * 3600 + unsw_X["Stime"].dt.minute * 60 + unsw_X["Stime"].dt.second

    ## Categorical features for feature encoding
    categorical_features = ["sport", "dsport","proto","service","state","srcip","dstip"]
    # Convert numerical features to strings
    unsw_X[categorical_features] = unsw_X[categorical_features].astype(str)
    # Numerical features for feature scaling
    numerical_features = [ feat for feat in unsw_X.columns if feat not in categorical_features]

    ## Scalers and encoders   
     # Feature encoding using Ordinal Encoder
    ord_enc = OrdinalEncoder()
     # Scaling using z-score for numerical features
    zsc = StandardScaler()
    log(INFO,"\tFeature scaling and encoding ...")
    ## Transform
    preprocessor = ColumnTransformer(
            [("standard_scaling", zsc, numerical_features),
            ("feature_encoding", ord_enc, categorical_features)],
            remainder="passthrough"
    )
    unsw_X = preprocessor.fit_transform(unsw_X)
    unsw_X= pd.DataFrame(unsw_X, columns= numerical_features + categorical_features)
    return unsw_X

def _unsw_label(unsw_y):
    log(INFO,"UNSW-NB15 Label encoding ...")
    le = LabelEncoder()
    unsw_y = le.fit_transform(unsw_y)
    unsw_y = pd.DataFrame(unsw_y, columns=["Label"])
    return unsw_y

def _cicids_preprocessing(cicids_X):
    log(INFO,"CIC-IDS2017 preprocessing...")
## Time preprocessing
    # Timestamp to Epoch
    log(INFO,"\tTime encoding ...")  
    timestamps = pd.to_datetime(cicids_X["Timestamp"],format="mixed", dayfirst=True)
    cicids_X["Timestamp"] = (timestamps - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    # Convert epoch time to datetime
    cicids_X["Timestamp"] = pd.to_datetime(cicids_X["Timestamp"], unit='s')
    # Extract time component and convert to numeric representation
    cicids_X["Timestamp"] = cicids_X["Timestamp"].dt.hour * 3600 + cicids_X["Timestamp"].dt.minute * 60 + cicids_X["Timestamp"].dt.second
    
## Categorical features for feature encoding
    categorical_features = ["Source IP","Source Port", "Destination IP", "Destination Port","Protocol"]
## Numerical features for feature scaling
    numerical_features = [ feat for feat in cicids_X.columns if feat not in categorical_features]
    
## Scaling
    log(INFO,("\tInfinity values treatment..."))
    # Change infinite values in "Flow Bytes/s" for the maximum value +1 
    max_finite_flowrate = cicids_X["Flow Bytes/s"][np.isfinite(cicids_X["Flow Bytes/s"])].max()
    infinite_flowrate = max_finite_flowrate + 1
    cicids_X["Flow Bytes/s"] = cicids_X["Flow Bytes/s"].replace(np.inf, infinite_flowrate)
    # Change infinite values in "Total Length of Bwd Packets" for the maximum value +1 
    max_finite_bwdpkt = cicids_X["Total Length of Bwd Packets"][np.isfinite(cicids_X["Total Length of Bwd Packets"])].max()
    infinite_bwdpkt = max_finite_bwdpkt + 1
    cicids_X["Total Length of Bwd Packets"] = cicids_X["Total Length of Bwd Packets"].replace(np.inf, infinite_bwdpkt)
    # Probar a quitar antes los infinitos de flowpackets
    max_finite_flowpkts = cicids_X["Flow Packets/s"][np.isfinite(cicids_X["Flow Packets/s"])].max()
    infinite_flowpkts = max_finite_flowpkts + 1
    cicids_X["Flow Packets/s"] = cicids_X["Flow Packets/s"].replace(np.inf, infinite_flowpkts)
    # Z-score for numerical features
    zsc = StandardScaler()
## Transform
    log(INFO,"\tFeature scaling and encoding ...")  
    preprocessor = ColumnTransformer(
        [("standard_scaling", zsc, numerical_features)],
        remainder="passthrough"
    )
    cicids_X = preprocessor.fit_transform(cicids_X)
    cicids_X = pd.DataFrame(cicids_X, columns=numerical_features + categorical_features)
    return cicids_X

def _cicids_label(cicids_y):
    log(INFO,"CIC-IDS2017 Label encoding ...")
    le = LabelEncoder()
    cicids_y = le.fit_transform(cicids_y)
    cicids_y= pd.DataFrame(cicids_y, columns=["Label"])
    cicids_y.Label= (cicids_y.Label > 0).astype(int)
    return cicids_y

# Reindexing columns to join by column axis 
def _reindex_columns(df_X, df_y, new_cols):
    df_X.reset_index(drop=True, inplace=True)
    df_y.reset_index(drop=True, inplace=True)
    df = pd.concat([df_X, df_y], axis=1)
    df = df.reindex(new_cols, axis=1)
    return df

def final_preprocesing(data, columns, final_cols):
    data.reset_index(drop=True, inplace=True)
    log(INFO,"Scaling fused data...")
    zsc = StandardScaler()
    preprocessor = ColumnTransformer(
        [("standard_scaling", zsc, columns)],
        remainder="passthrough"
    )
    data = preprocessor.fit_transform(data)
    data = pd.DataFrame(data, columns=final_cols)
    return data

# Data fusion: feature fusion
def _data_fusion(unsw_X, unsw_y, cicids_X, cicids_y):
     # Column order
    unsw_cols = ["sport", "dsport", "proto", "dur", "Spkts", "Dpkts", "Stime", 
                "smeansz", "dmeansz", "flowbytesps", "Sintpkt", "swin","dwin", "Label"]
    cicids_cols = ["Source Port", "Destination Port", "Protocol", "Flow Duration", "Total Length of Fwd Packets", 
                "Total Length of Bwd Packets", "Timestamp", "Fwd Packet Length Mean","Flow Bytes/s", 
                "Fwd IAT Total", "Bwd IAT Total", "Init_Win_bytes_forward", "Init_Win_bytes_backward","Label"]
    # Reindexing dataframes
    unsw_reindex = _reindex_columns(unsw_X, unsw_y, unsw_cols)
    cicids_reindex = _reindex_columns(cicids_X, cicids_y, cicids_cols)
    log(INFO,"Fusing data ...")
    # New features vector
    fused_data_cols = ["s_port","d_port","protocol","duration","s_pktslength", "d_pktslength","timestamp", "s_pktsmean", "d_pktsmean", 
                    "bytesps", "s_IAT", "s_win", "d_win", "label"]
    features = ["s_port","d_port","protocol","duration","s_pktslength", "d_pktslength","timestamp", "s_pktsmean", "d_pktsmean", 
                    "bytesps", "s_IAT", "s_win", "d_win"]
    unsw_reindex.columns = fused_data_cols
    cicids_reindex.columns = fused_data_cols
    # Fuse 
    fused_data = pd.concat([unsw_reindex, cicids_reindex])
    fused_data = fused_data.iloc[:1000000]
    fused_data_X = fused_data.drop("label", axis=1)
    fused_data_y = fused_data["label"]
    fused_data = final_preprocesing(fused_data, features, fused_data_cols)
    n_features = len(fused_data_X.columns)
    return fused_data_X,fused_data_y, n_features

def load_data(test_split):
    random_state=42
    unsw_path=datapath+"raw/unsw-nb15"   
    cic_path =datapath+ "raw/cic-ids2017"
    unsw = _unsw_load(unsw_path)
    cicids = _cicids_load(cic_path)
    unsw = _unsw_cleaning(unsw=unsw)
    cicids = _cicids_cleaning(cicids)
    # Split in X, y
    unsw_X = unsw.drop("Label", axis=1)
    unsw_y = unsw["Label"]
    cicids_X = cicids.drop("Label", axis=1)
    cicids_y = cicids["Label"]
    # UNSW
    unsw_X = _unsw_preprocessing(unsw_X)
    unsw_y = _unsw_label(unsw_y)
    # CICIDS 
    cicids_X = _cicids_preprocessing(cicids_X)
    cicids_y = _cicids_label(cicids_y)
    fused_data_X, fused_data_y,n_features = _data_fusion(unsw_X,unsw_y,cicids_X,cicids_y)
    log(INFO,"Saving data ...")
    if test_split: 
        X_train, X_test, y_train, y_test = train_test_split(fused_data_X,fused_data_y,test_size=0.2,random_state=42)
        n_features = len(X_train.columns)
        return X_train,X_test,y_train,y_test,n_features
    else:
        return fused_data_X, fused_data_y,n_features

def eval_metrics(predictions, y_test):
    acc = accuracy_score(y_test, predictions)
    rec = recall_score(y_test, predictions,zero_division = 1)
    prec = precision_score(y_test,predictions, zero_division =0)
    f1s = f1_score(y_test, predictions,zero_division = 0)
    loss = log_loss(y_test, predictions)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    
    return acc, rec, prec, f1s, tn, fp, fn, tp, loss

def plot_global_metric( history: None, metric_type: None, metric: None ) -> None:
    model = os.getenv("MODEL")
    metric_dict = (history.metrics_centralized if metric_type == "centralized" else history.metrics_distributed)
    rounds, values = zip(*metric_dict[metric])
    plt.figure(figsize=(10, 6))
    plt.plot(np.asarray(rounds), np.asarray(values), linewidth=2, label=f'{metric} ({metric_type})')
    plt.legend(fontsize=12)
    plt.title(f'{metric.capitalize()} tras rondas de entrenamiendo de ({model.capitalize()})', fontsize=16)
    plt.xlabel('Ronda de entrenamiento', fontsize=14)
    plt.ylabel(metric.capitalize(), fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    #plt.style.use('seaborn-darkgrid')
    plt.savefig(f"results/{model}/results_{metric}_{metric_type}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss(history: None, metric_type: None,) -> None:
    log(INFO, history)
    model = os.getenv("MODEL")
    metric_dict = history.losses_centralized if metric_type == "centralized" else history.losses_distributed
    rounds, values = zip(*metric_dict)
    plt.figure(figsize=(10, 6))
    plt.plot(np.asarray(rounds), np.asarray(values), linewidth=2, label='Loss')
    plt.legend(fontsize=12)
    plt.title(f'Pérdidas tras rondas de entrenamiento de ({model.capitalize()})', fontsize=16)
    plt.xlabel('Ronda de entrenamiento', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.ylim(0, max(values) * 1.1)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    #plt.style.use('seaborn-darkgrid')
    plt.savefig(f"results/{model}/results_loss_{metric_type}.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_final_results(history: None, metric_type: None) -> None:
    model = os.getenv("MODEL")
    metric_dict = history.metrics_centralized if metric_type == "centralized" else history.metrics_distributed
    metrics = ["accuracy", "recall", "precision", "f1score", "fpr"]
    results_dict = {metric: [] for metric in metrics}
    rounds = []
    for metric in metrics:
        rounds, values = zip(*metric_dict[metric])
        results_dict[metric] = values
    results_dict["round"] = rounds
    results = pd.DataFrame(results_dict)
    columns_order = ["round"] + metrics
    results = results[columns_order]
    results.to_csv(f"results/{model}/results.csv",index=False)
    