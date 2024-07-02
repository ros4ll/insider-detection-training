# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from sklearn.model_selection import train_test_split
# Variables
unsw_path="data/raw/unsw-nb15"
cic_path = "data/raw/cic-ids2017"

def unsw_select_samples(unsw):
    class_0 = unsw[unsw['Label'] == 0]
    class_1 = unsw[unsw['Label'] == 1]
    # Determinar el tamaño mínimo entre las dos clases
    min_size = min(len(class_0), len(class_1))

    # Submuestrear las clases para que tengan el mismo tamaño
    class_0_balanced = class_0.sample(n=min_size, random_state=42)
    class_1_balanced = class_1.sample(n=min_size, random_state=42)

    unsw_balanced = pd.concat([class_0_balanced, class_1_balanced])
    return unsw_balanced
def cicids_select_samples(cicids):
    # Crear una nueva columna para la clasificación binaria
    cicids['binary_label'] = cicids['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

    # Separar el conjunto de datos en clases
    class_benign = cicids[cicids['binary_label'] == 0]
    class_other = cicids[cicids['binary_label'] == 1]

    # Determinar el tamaño mínimo entre las dos clases
    min_size = min(len(class_benign), len(class_other))

    # Submuestrear las clases para que tengan el mismo tamaño
    class_benign_balanced = class_benign.sample(n=min_size, random_state=42)
    class_other_balanced = class_other.sample(n=min_size, random_state=42)
    cicids_balanced = pd.concat([class_benign_balanced, class_other_balanced])
    cicids_balanced = cicids_balanced.drop('binary_label', axis=1)
    return cicids_balanced
    
def main():
    #unsw
    header = None
    encoding ='utf-8'
    files_list = [os.path.join(f'{unsw_path}/generatedflows', file) for file in os.listdir(f'{unsw_path}/generatedflows') if file.endswith(".csv")]
    data = []
    for file in files_list:
        df_temp = pd.read_csv(file, header=header, low_memory=False, sep=',', encoding=encoding)
        data.append(df_temp)
    unsw = pd.concat(data)
    #UNSW feature names loading
    features = []
    with open (f'{unsw_path}/UNSW-NB15_features.csv') as features_csv:
        filereader = csv.reader(features_csv)
        for row in filereader: 
            features.append(row[1])
    features.pop(0)
    unsw.columns = features
    #cicids
    header = 0
    encoding = 'latin1'
    files_list = [os.path.join(cic_path, file) for file in os.listdir(cic_path) if file.endswith(".csv")]
    data = []
    for file in files_list:
        df_temp = pd.read_csv(file, header=header, low_memory=False, sep=',', encoding=encoding)
        data.append(df_temp)
    cicids = pd.concat(data)
    cicids.columns = [feat.lstrip(' ') for feat in cicids.columns]
    # unsw = unsw_select_samples(unsw)
    # cicids = cicids_select_samples(cicids)
    half_unsw = int(round(len(unsw)/2))
    unsw = unsw.sample(n= half_unsw,random_state=42)
    half_cicids = int(round(len(cicids)/2))
    cicids = cicids.sample(n= half_cicids,random_state=42)
    # Dividir el DataFrame en 80% (site-1,site-2) y 20% (test)
    cicids_train, cicids_test = train_test_split(cicids, test_size=0.2, random_state=42)

    # Dividir el DataFrame de train_val_df en 50% (site1) y 50% (site2) para que sea 40% del total cada uno
    cicids_site1, cicids_site2 = train_test_split(cicids_train, test_size=0.5, random_state=42)
    cicids_site1.to_csv("data/site-1/raw/cic-ids2017/cicids.csv", index=False, encoding='utf-8')
    cicids_site2.to_csv("data/site-2/raw/cic-ids2017/cicids.csv", index=False, encoding='utf-8')
    cicids_test.to_csv("data/server/raw/cic-ids2017/cicids.csv", index=False, encoding='utf-8')
    # Dividir el DataFrame en 80% (site-1,site-2) y 20% (test)
    unsw_train, unsw_test = train_test_split(unsw, test_size=0.2, random_state=42)

    # Dividir el DataFrame de train_val_df en 50% (site1) y 50% (site2) para que sea 40% del total cada uno
    unsw_site1, unsw_site2 = train_test_split(unsw_train, test_size=0.5, random_state=42)
    unsw_site1.to_csv("data/site-1/raw/unsw-nb15/generatedflows/unsw.csv", index=False, encoding='utf-8')
    unsw_site2.to_csv("data/site-2/raw/unsw-nb15/generatedflows/unsw.csv", index=False, encoding='utf-8')
    unsw_test.to_csv("data/server/raw/unsw-nb15/generatedflows/unsw.csv", index=False, encoding='utf-8')

if __name__ == "__main__":
    main()