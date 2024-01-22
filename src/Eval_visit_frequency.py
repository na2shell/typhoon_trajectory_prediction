import pandas as pd
import numpy as np 

df_original = pd.read_csv("/src/dev_train_encoded_final.csv")
# df_generated = pd.read_csv("/src/k-same-net_generated_traj.csv")
# df_generated = pd.read_csv("/src/dev_test_encoded_final.csv")
df_generated = pd.read_csv("/src/k-same-net_generated_traj_k=2.csv")


category_dict = {}
category_dict["day"] = [i for i in range(7)]
category_dict["hour"] = [i for i in range(24)]
category_dict["category"] = [i for i in range(10)]

def eval_coeff(feature="day"):
    df_origin = df_original.groupby([feature]).size()
    df_gen = df_generated.groupby([feature]).size()

    for i in category_dict[feature]:
        if not df_gen.index.isin([i]).any():
            df_gen = df_gen.append(pd.Series([0], index = [i]))
        
        if not df_origin.index.isin([i]).any():
            df_origin.append(pd.Series([0], index = [i]))
            
    
    df_gen = df_gen.sort_index()
    df_origin = df_origin.sort_index()
    print(df_gen, df_origin)
    df_gen = df_gen.values
    df_origin = df_origin.values

    
    corrcoef = np.corrcoef(df_origin, df_gen)[0][1]
    

    return corrcoef

coeff = eval_coeff(feature="day")
print("day", coeff)
print("="*40)

coeff = eval_coeff(feature="hour")
print("hour", coeff)
print("="*40)

coeff = eval_coeff(feature="category")
print("category", coeff)
print("="*40)