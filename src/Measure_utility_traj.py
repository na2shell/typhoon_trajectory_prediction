import pandas as pd
import torch
from utils_for_seq2seq import MUITAS
import tqdm

df = pd.read_csv("/data/dev_train_encoded_final.csv")
df_gen = pd.read_csv("/src/k-same-net_generated_traj_k=3.csv")

tids = df["tid"].unique()


def anytime_zero(x, y):
    if abs(x - y) < 0.01:
        return 0
    else:
        return 1


def match_or_not(x, y):
    if x == y:
        return 0
    else:
        return 1


dist_functions = [anytime_zero for _ in range(2)] + [match_or_not for _ in range(3)]
thresholds = [0.5 for _ in range(5)]
features = [i for i in range(5)]
weights = torch.tensor([1 for _ in range(5)])

Utility_function_class = MUITAS(dist_functions, thresholds, features, weights)
similarities = []

for tid in tqdm.tqdm(tids):
    original = torch.tensor(
        df[df["tid"] == tid][["lat", "lon", "day", "hour", "category"]].values
    )
    gen = torch.tensor(
        df_gen[df_gen["tid"] == tid][["lat", "lon", "day", "hour", "category"]].values
    )

    original = original.view(1, original.size(0), original.size(1))
    gen = gen.view(1, gen.size(0), gen.size(1))

    sim = Utility_function_class.similarity(t1=original, t2=gen)
    similarities.append(sim.item())

print(sum(similarities) / len(similarities))
