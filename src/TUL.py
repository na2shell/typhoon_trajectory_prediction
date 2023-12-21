import torch
import torch.nn as nn
from traj_Dataset import MyDataset
from torch.nn.utils.rnn import pad_packed_sequence
from utils import time_collate_fn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from utils import convert_onehot, convert_label_to_inger, build_encoder, build_int_encoder
import pandas as pd


class TUL_attack_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TUL_attack_model, self).__init__()
        self.precision = 8
        each_hidden_dim = 128

        self.postion_last_index = 5*self.precision
        self.day_last_index = self.postion_last_index + 7
        self.category_last_index = self.day_last_index + 10

        self.latlon_embbedding = nn.Linear(5*self.precision, each_hidden_dim)
        self.day_embbedding = nn.Linear(7, each_hidden_dim)
        self.hour_embbedding = nn.Linear(24, each_hidden_dim)
        self.category_embbedding = nn.Linear(10, each_hidden_dim)

        self.dense = nn.Linear(each_hidden_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=each_hidden_dim*4, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=3)

        self.lstm = nn.LSTM(each_hidden_dim*4, hidden_dim, batch_first=False)

        self.output_layer = nn.Linear(each_hidden_dim*4, output_dim)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x, traj_len, mask):

        latlon_e = self.latlon_embbedding(x[:, :, :self.postion_last_index])
        # print(latlon_e.size())
        day_e = self.day_embbedding(
            x[:, :, self.postion_last_index:self.postion_last_index+7])
        category_e = self.category_embbedding(
            x[:, :, self.day_last_index:self.day_last_index+10])
        hour_e = self.hour_embbedding(x[:, :, self.category_last_index:])

        embedding_all = torch.cat([latlon_e, day_e, category_e, hour_e], dim=2)

        # print(embedding_all.size())
        # embedding_all = torch.mean(all, dim=0)
        # embedding_all = self.dense(embedding_all_tmp)

        # packed_data = torch.nn.utils.rnn.pack_padded_sequence(
        #     embedding_all, batch_first=True, lengths=traj_len, enforce_sorted=False)
        # x, (hn, cn) = self.lstm(packed_data)
        # seq_unpacked, lens_unpacked = pad_packed_sequence(x, batch_first=True)

        # x = self.output_layer(hn.squeeze())

        B, L, C = embedding_all.size(
            0), embedding_all.size(1), embedding_all.size(2)

        x = self.transformer_encoder(embedding_all, src_key_padding_mask=mask)
        # max_val, indices= torch.max(x, dim=1)
        # print("x size", x[1, :, :])
        x = torch.mean(x, dim=1)
        # print("after max pooling", x.size())
        x = self.output_layer(x)
        # print("trans former", x.size())
        x = self.soft_max(x)
        return x


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

    target_dict = {}
    target_dict["day"] = ([i for i in range(7)])
    target_dict["hour"] = ([i for i in range(24)])
    target_dict["category"] = ([i for i in range(10)])

    encoder_dict = {}

    for col in ["day", "category", "hour"]:
        target = target_dict[col]
        encoder_dict[col] = build_encoder(target)

    data_path = "./dev_train_encoded_final.csv"
    df = pd.read_csv(data_path)
    int_label_encoder = build_int_encoder(df["label"].unique())
    print(int_label_encoder.classes_)

    train_data_set = MyDataset(
        df=df, encoder_dict=encoder_dict, int_label_encoder=int_label_encoder)

    data_path = "./dev_test_encoded_final.csv"
    df = pd.read_csv(data_path)
    test_data_set = MyDataset(
        df=df, encoder_dict=encoder_dict, int_label_encoder=int_label_encoder)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_data_set,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   collate_fn=time_collate_fn,
                                                   drop_last=True,
                                                   num_workers=2)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_data_set,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=True,
                                                  collate_fn=time_collate_fn,
                                                  drop_last=True,
                                                  num_workers=2)

    model = TUL_attack_model(input_dim=43, hidden_dim=128, output_dim=193)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print("device", DEVICE)
    model.to(DEVICE)

    
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

        print("epoch {}: loss {}".format(epoch, sum(losses)/len(losses)))

        if epoch % 50 == 0:
            eval_model(model=model, test_dataloader=test_dataloader, DEVICE=DEVICE)
            torch.save(model.state_dict(
            ), "./TUL_model_weight/tul_model_weight_epoch_{}.pth".format(epoch))
