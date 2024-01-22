import torch
import torch.nn as nn
from traj_Dataset import MyDataset
from utils import time_collate_fn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from utils import build_encoder, build_int_encoder
import pandas as pd
from TUL_model import TUL_attack_model


def eval_model(model, test_dataloader, DEVICE):
    model.eval()
    preds = []
    actuals = []
    for data, traj_len, traj_class_indices, label, mask in test_dataloader:
        pred_y = model(data.to(DEVICE), traj_len, mask.to(DEVICE))
        pred_y_class = torch.argmax(pred_y, dim=1)

        preds.append(pred_y_class.tolist())
        actuals.append(label.tolist())

    preds = sum(preds, [])
    actuals = sum(actuals, [])

    print("accuracy score:", accuracy_score(actuals, preds))


if __name__ == "__main__":
    BATCH_SIZE = 128
    lr = 1e-4
    epoch_num = 1000
    is_train = False

    target_dict = {}
    target_dict["day"] = [i for i in range(7)]
    target_dict["hour"] = [i for i in range(24)]
    target_dict["category"] = [i for i in range(10)]

    encoder_dict = {}

    for col in ["day", "category", "hour"]:
        target = target_dict[col]
        encoder_dict[col] = build_encoder(target)

    data_path = "/data/dev_train_encoded_final.csv"
    df = pd.read_csv(data_path)
    int_label_encoder = build_int_encoder(df["label"].unique())
    print(int_label_encoder.classes_)

    train_data_set = MyDataset(
        df=df,
        encoder_dict=encoder_dict,
        int_label_encoder=int_label_encoder,
        is_applied_geohash=True,
    )

    data_path = "/data/k-same-net_generated_traj_k=2.csv"
    df = pd.read_csv(data_path)
    df["label"] = int_label_encoder.inverse_transform(df["uid"])
    gen_data_set = MyDataset(
        df=df,
        encoder_dict=encoder_dict,
        int_label_encoder=int_label_encoder,
        is_applied_geohash=True,
    )

    data_path = "/data/dev_test_encoded_final.csv"
    df = pd.read_csv(data_path)
    test_data_set = MyDataset(
        df=df,
        encoder_dict=encoder_dict,
        int_label_encoder=int_label_encoder,
        is_applied_geohash=True,
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_data_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=time_collate_fn,
        drop_last=True,
        num_workers=2,
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_data_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=time_collate_fn,
        drop_last=True,
        num_workers=2,
    )

    gen_dataloader = torch.utils.data.DataLoader(
        dataset=gen_data_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=time_collate_fn,
        drop_last=True,
        num_workers=2,
    )

    model = TUL_attack_model(input_dim=43, hidden_dim=128, output_dim=193)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    DEVICE = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print("device", DEVICE)
    model.to(DEVICE)

    if is_train:
        for epoch in range(epoch_num):
            model.train()
            losses = []

            for data, traj_len, traj_class_indices, label, mask in train_dataloader:
                # print(data.size())
                pred_y = model(data.to(DEVICE), traj_len, mask.to(DEVICE))
                loss = criterion(pred_y, label.to(DEVICE))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.detach())

            print("epoch {}: loss {}".format(epoch, sum(losses) / len(losses)))

            if epoch % 50 == 0:
                torch.save(
                    model.state_dict(),
                    "/src/TUL_model_weight/tul_model_weight_epoch_{}.pth".format(epoch),
                )

    model.load_state_dict(
        torch.load("/src/TUL_model_weight/tul_model_weight_epoch_950.pth")
    )
    model.eval()
    print("normal test:")
    eval_model(model=model, test_dataloader=test_dataloader, DEVICE=DEVICE)

    print("generated data test:")
    eval_model(model=model, test_dataloader=gen_dataloader, DEVICE=DEVICE)
