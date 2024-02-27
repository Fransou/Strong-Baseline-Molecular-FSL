import pandas as pd
import numpy as np

df_domain = pd.read_csv("datasets/targets/target_info.csv")


def process_EC(EC):
    if isinstance(EC, str):
        try:
            return int(EC)
        except Exception as e:
            EC = EC[1:-1].replace("'", "")
            return int(EC.split(",")[0][:])
    if np.isnan(EC):
        return 0
    return EC


df_domain["EC"] = df_domain["EC_super_class"].apply(process_EC)
