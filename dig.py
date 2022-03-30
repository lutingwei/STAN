# %%

import numpy as np
import torch
from torch import optim
from metric import get_mrr, get_recall
import datetime
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import pickle

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

test64 = pickle.load(open('datasets/diginetica/test.txt', 'rb'))
train64 = pickle.load(open('datasets/diginetica/train.txt', 'rb'))

train64_x = train64[1]
train64_y = train64[2]

test64_x = test64[1]
test64_y = test64[2]
train_pos = list()
test_pos = list()

item_set = set()

for items in train64[1]:
    pos = list()
    for id_ in range(len(items)):
        item_set.add(items[id_])
        pos.append(id_ + 1)
    train_pos.append(pos)

for item in train64[2]:
    item_set.add(item)

for items in test64[1]:
    pos = []
    for id_ in range(len(items)):
        item_set.add(items[id_])
        pos.append(id_ + 1)
    test_pos.append(pos)

for item in test64[2]:
    item_set.add(item)
item_list = sorted(list(item_set))  # 40840个item
item_dict = dict()  # 再次更改item_id
for i in range(1, len(item_set) + 1):
    item = item_list[i - 1]
    item_dict[item] = i

# %%

train64_x = list()
train64_y = list()

test64_x = list()
test64_y = list()

for items in train64[1]:
    new_list = []
    for item in items:
        new_item = item_dict[item]
        new_list.append(new_item)
    train64_x.append(new_list)
for item in train64[2]:
    new_item = item_dict[item]
    train64_y.append(new_item)
for items in test64[1]:
    new_list = []
    for item in items:
        new_item = item_dict[item]
        new_list.append(new_item)
    test64_x.append(new_list)
for item in test64[2]:
    new_item = item_dict[item]
    test64_y.append(new_item)

# %%

max_length = 0  # 69
for sample in train64_x:
    max_length = len(sample) if len(sample) > max_length else max_length
for sample in test64_x:
    max_length = len(sample) if len(sample) > max_length else max_length

train_seqs = np.zeros((len(train64_x), max_length))
train_poses = np.zeros((len(train64_x), max_length))
test_seqs = np.zeros((len(test64_x), max_length))
test_poses = np.zeros((len(test64_x), max_length))

# 前面是0，后面放数据
for i in range(len(train64_x)):
    seq = train64_x[i]
    pos = train_pos[i]
    length = len(seq)
    train_seqs[i][-length:] = seq
    train_poses[i][-length:] = pos

for i in range(len(test64_x)):
    seq = test64_x[i]
    pos = test_pos[i]
    length = len(seq)
    test_seqs[i][-length:] = seq
    test_poses[i][-length:] = pos

target_seqs = np.array(train64_y)
target_test_seqs = np.array(test64_y)

# %%

item_set = set()
for items in train64_x:
    for item in items:
        item_set.add(item)
for item in train64_y:
    item_set.add(item)
for items in test64_x:
    for item in items:
        item_set.add(item)
for item in test64_y:
    item_set.add(item)

# %%

train_x = torch.Tensor(train_seqs)  # torch.Size([526135, 69])
train_pos = torch.Tensor(train_poses)
train_y = torch.Tensor(target_seqs)
test_x = torch.Tensor(test_seqs)  # torch.Size([44279, 69])
test_pos = torch.Tensor(test_poses)
test_y = torch.Tensor(target_test_seqs)


class DualAttention(nn.Module):
    def __init__(self, item_dim, pos_dim, n_items, n_pos, w, atten_way='dot', decoder_way='trilinear', dropout=0,
                 activate='relu'):
        super(DualAttention, self).__init__()
        self.item_dim = item_dim  # 100
        self.pos_dim = pos_dim  # 100
        dim = item_dim + pos_dim
        self.dim = dim  # 200
        self.n_pos = n_pos
        self.n_items = n_items
        self.embedding = nn.Embedding(n_items, item_dim, padding_idx=0, max_norm=1.5)
        self.pos_embedding = nn.Embedding(n_pos + 1, pos_dim, padding_idx=0, max_norm=1.5)

        self.atten_way = atten_way
        self.decoder_way = decoder_way

        self.vanila_w0 = nn.Parameter(torch.Tensor(1, dim))  # vanila attention中的W_0
        self.vanila_w1 = nn.Parameter(torch.Tensor(dim, dim))  # vanila attention中的W_1
        self.vanila_w2 = nn.Parameter(torch.Tensor(dim, dim))  # vanila attention中的W_2
        self.vanila_bias = nn.Parameter(torch.Tensor(dim))  # vanila attention中的b_a

        self.dropout = nn.Dropout(dropout)

        self.w1 = nn.Linear(dim, dim)  # FFN的W1
        self.w2 = nn.Linear(dim, dim)  # FFN的W2

        self.wA = nn.Parameter(torch.Tensor(1, n_pos))  # FFN结果变为[1, 200]

        self.w_global1 = nn.Linear(dim, dim)  # 计算s_global的w1
        self.w_global2 = nn.Linear(dim, dim)  # 计算s_global的w2
        self.global_q = nn.Parameter(torch.Tensor(dim, 1))  # 计算s_global的q

        self.w_f = nn.Linear(3 * dim, item_dim)  # prediction layer中的W

        self.LN = nn.LayerNorm(dim)  # 对FFN结果层归一化
        self.is_dropout = True

        self.w = w  # 20

        if activate == 'relu':
            self.activate = F.relu
        elif activate == 'selu':
            self.activate = F.selu

        self.initial_()

    def initial_(self):
        init.normal_(self.vanila_w0, 0, 1/self.w)
        init.normal_(self.vanila_w1, 0, 1/self.w)
        init.normal_(self.vanila_w2, 0, 1/self.w)
        init.constant_(self.vanila_bias, 0)

        init.normal_(self.wA, 0, 1/self.w)
        init.normal_(self.global_q, 0, 1/self.w)

        init.constant_(self.embedding.weight[0], 0)
        init.constant_(self.pos_embedding.weight[0], 0)

    def forward(self, x, pos):
        self.is_dropout = True
        x_embeddings = self.embedding(x)  # B, seq, dim
        pos_embeddings = self.pos_embedding(pos)  # B, seq, dim
        mask = (x != 0).float()  # B, seq  有item为1，无item为0
        x_ = torch.cat((x_embeddings, pos_embeddings), 2)  # B, seq, 2*dim
        S = torch.Tensor().cuda()  # 用于存储target item遍历每个item的session representation
        for i in range(self.n_pos):
            s_i = self.vanila_attention(x_[:, i, :].unsqueeze(1), x_, x_, mask)  # B, 1, dim
            S = torch.cat((S, s_i), 1)
        s_target = self.MLP(S)  # [512, 1, 200]
        s_local = x_[:, -1, :].unsqueeze(1)  # [512, 1, 200]
        s_global = self.global_s(s_local, x_)  # [512, 1, 200]
        result = self.decoder(s_target, s_local, s_global)
        return result

    def vanila_attention(self, target, k, v, mask=None):
        alpha = torch.matmul(
            torch.relu(k.matmul(self.vanila_w1) + target.matmul(self.vanila_w2) + self.vanila_bias),
            self.vanila_w0.t())  # (B, seq, 1)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            alpha = alpha.masked_fill(mask == 0, -np.inf)
        alpha = torch.softmax(alpha, 1)
        c = torch.matmul(alpha.transpose(1, 2), v)  # (B, 1, dim)
        return c

    def MLP(self, S):
        if self.is_dropout:  # FFN操作
            S = self.dropout(self.w2(self.activate(self.w1(S)))) + S
        else:
            S = self.w2(self.activate(self.w1(S))) + S
        S = self.LN(S)
        S = torch.matmul(self.wA, S)  # [512, 1, 200]
        return S

    def global_s(self, s_local, x):
        alpha = torch.matmul(torch.sigmoid(self.w_global1(x) + self.w_global2(s_local)),
                             self.global_q)
        sg = torch.matmul(alpha.transpose(1, 2), x)
        return sg

    def decoder(self, st, sl, sg):
        if self.is_dropout:
            c = self.dropout(torch.selu(self.w_f(torch.cat((st, sl, sg), 2))))  # [512, 1, 100]
        else:
            c = torch.selu(self.w_f(torch.cat((st, sl, sg), 2)))  # [512, 1, 100]
        c = c.squeeze()  # [512, 100]
        l_c = (c / torch.norm(c, dim=-1).unsqueeze(1))  # [512, 100]
        l_emb = self.embedding.weight[1:] / torch.norm(self.embedding.weight[1:], dim=-1).unsqueeze(
            1)
        z = self.w * torch.matmul(l_c, l_emb.t())  # self.w=20
        return z

    def predict(self, x, pos, k=20):
        self.is_dropout = False
        x_embeddings = self.embedding(x)  # B, seq, dim
        pos_embeddings = self.pos_embedding(pos)  # B, seq, dim
        mask = (x != 0).float()  # B, seq  有item为1，无item为0
        x_ = torch.cat((x_embeddings, pos_embeddings), 2)  # B, seq, 2*dim
        S = torch.Tensor().cuda()  # 用于存储target item遍历每个item的session representation
        for i in range(self.n_pos):
            s_i = self.vanila_attention(x_[:, i, :].unsqueeze(1), x_, x_, mask)  # B, 1, dim [512, 1, 200]
            S = torch.cat((S, s_i), 1)
        s_target = self.MLP(S)  # [512, 1, 200]
        s_local = x_[:, -1, :].unsqueeze(1)  # [512, 1, 200]
        s_global = self.global_s(s_local, x_)  # [512, 1, 200]
        result = self.decoder(s_target, s_local, s_global)
        rank = torch.argsort(result, dim=1, descending=True)
        return rank[:, 0:k]


# %%

w_list = [20]
record = list()
batch_size = 512
# batch_size = 256
for w in w_list:
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_sets = TensorDataset(train_x.long(), train_pos.long(), train_y.long())
    train_dataload = DataLoader(train_sets, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss().cuda()
    test_x, test_pos, test_y = test_x.long(), test_pos.long(), test_y.long()
    all_test_sets = TensorDataset(test_x, test_pos, test_y)
    test_dataload = DataLoader(all_test_sets, batch_size=batch_size, shuffle=False)
    model = DualAttention(100, 100, 40841, 69, w, dropout=0.5, activate='relu').cuda()  # w=20
    opti = optim.Adam(model.parameters(), lr=0.001, weight_decay=0, amsgrad=True)
    best_result = 0
    total_time = 0
    best_result_5 = 0
    best_result_ = []
    for epoch in range(50):
        start_time = datetime.datetime.now()
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        losses = 0
        for step, (x_train, pos_train, y_train) in enumerate(train_dataload):
            opti.zero_grad()
            q = model(x_train.cuda(), pos_train.cuda())  # [512, 40841]
            loss = criterion(q, y_train.cuda() - 1)
            loss.backward()
            opti.step()
            losses += loss.item()
            if (step + 1) % 100 == 0:
                print("[%02d/%d] [%03d/%d] mean_loss : %0.2f" % (
                    epoch + 1, 50, step, len(train_sets) / batch_size, losses / step + 1))
        end_time = datetime.datetime.now()
        with torch.no_grad():
            y_pre_all = torch.LongTensor().cuda()
            y_pre_all_10 = torch.LongTensor()
            y_pre_all_5 = torch.LongTensor()
            for x_test, pos_test, y_test in test_dataload:
                with torch.no_grad():
                    y_pre = model.predict(x_test.cuda(), pos_test.cuda(), 20)
                    y_pre_all = torch.cat((y_pre_all, y_pre), 0)
                    y_pre_all_10 = torch.cat((y_pre_all_10, y_pre.cpu()[:, :10]), 0)
                    y_pre_all_5 = torch.cat((y_pre_all_5, y_pre.cpu()[:, :5]), 0)
            recall = get_recall(y_pre_all, test_y.cuda().unsqueeze(1) - 1)
            recall_10 = get_recall(y_pre_all_10, test_y.unsqueeze(1) - 1)
            recall_5 = get_recall(y_pre_all_5, test_y.unsqueeze(1) - 1)
            mrr = get_mrr(y_pre_all, test_y.cuda().unsqueeze(1) - 1)
            mrr_10 = get_mrr(y_pre_all_10, test_y.unsqueeze(1) - 1)
            mrr_5 = get_mrr(y_pre_all_5, test_y.unsqueeze(1) - 1)

            print(
                "Recall@20: " + "%.4f" % recall + " Recall@10: " + "%.4f" % recall_10 + "  Recall@5:" + "%.4f" % recall_5)
            print(
                "MRR@20:" + "%.4f" % mrr.tolist() + " MRR@10:" + "%.4f" % mrr_10.tolist() + " MRR@5:" + "%.4f" % mrr_5.tolist())
            if best_result < recall:
                best_result = recall
                best_result_ = [recall_5, recall_10, recall, mrr_5, mrr_10, mrr]
                # torch.save(model.state_dict(), 'BestModel/best_dn_w_%s.pth' % str(w))
            print("best result: " + str(best_result))
            print("best result_: " + str(best_result_))
            print("==================================")
    record.append(best_result_)
print(record)
