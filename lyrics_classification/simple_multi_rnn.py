"""
감성분석을 할 수 있는 charRNN을 만드는 것.
analyser("나는 자연어처리가 좋아") -> 확률
"""
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import mean_squared_error
from torch.nn import functional as F
import pandas as pd
import MeCab
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

df = pd.read_csv('./발라드.csv')
df2 = pd.read_csv('./latent_vector_label.csv')
text = df['lyrics'].values
label =df2['label'].values
wakati = MeCab.Tagger("-Owakati")
DATA = [(wakati.parse(data), lb) for (data, lb) in zip(text, label)]

# DATA: List[Tuple[str, int]] = [
#     # 긍정적인 문장 - 1
#     ("나는 자연어처리", 1),
#     ("도움이 되었으면", 1),
#     # 병국님
#     ("오늘도 수고했어", 1),
#     # 영성님
#     ("너는 할 수 있어", 1),
#     # 정무님
#     ("오늘 내 주식이 올랐다", 1),
#     # 우철님
#     ("오늘 날씨가 좋다", 1),
#     # 유빈님
#     ("난 너를 좋아해", 1),
#     # 다운님
#     ("지금 정말 잘하고 있어", 1),
#     # 민종님
#     ("지금처럼만 하면 잘될거야", 1),
#     ("사랑해", 1),
#     ("저희 허락없이 아프지 마세요", 1),
#     ("오늘 점심 맛있다", 1),
#     ("오늘 너무 예쁘다", 1),
#     # 다운님
#     ("곧 주말이야", 1),
#     # 재용님
#     ("오늘 주식이 올랐어", 1),
#     # 병운님
#     ("우리에게 빛나는 미래가 있어", 1),
#     # 재용님
#     ("너는 참 잘생겼어", 1),
#     # 윤서님
#     ("콩나물 무침은 맛있어", 1),
#     # 정원님
#     ("강사님 보고 싶어요", 1),
#     # 정원님
#     ("오늘 참 멋있었어", 1),
#     # 예은님
#     ("맛있는게 먹고싶다", 1),
#     # 민성님
#     ("로또 당첨됐어", 1),
#     # 민성님
#     ("이 음식은 맛이 없을수가 없어", 1),
#     # 경서님
#     ("오늘도 좋은 하루보내요", 1),
#     # 성민님
#     ("내일 시험 안 본대", 1),
#     # --- 부정적인 문장 - 레이블 = 0
#     ("난 너를 싫어해", 0),
#     # 병국님
#     ("넌 잘하는게 뭐냐?", 0),
#     # 선희님
#     ("너 때문에 다 망쳤어", 0),
#     # 정무님
#     ("오늘 피곤하다", 0),
#     # 유빈님
#     ("난 삼성을 싫어해", 0),
#     ("진짜 가지가지 한다", 0),
#     ("꺼져", 0),
#     ("그렇게 살아서 뭘하겠니", 0),
#     # 재용님 - 주식이 파란불이다?
#     ("오늘 주식이 파란불이야", 0),
#     # 지현님
#     ("나 오늘 예민해", 0),
#     ("주식이 떨어졌다", 0),
#     ("콩나물 다시는 안먹어", 0),
#     ("코인 시즌 끝났다", 0),
#     ("배고파 죽을 것 같아", 0),
#     ("한강 몇도냐", 0),
#     ("집가고 싶다", 0),
#     ("나 보기가 역겨워", 0),  # 긍정적인 확률이 0
#     # 진환님
#     ("잘도 그러겠다", 0),
# ]

class ReviewsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def train_model(model, train_dl, val_dl, epochs=10, lr=0.001):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    crit = nn.CrossEntropyLoss()
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for x, y in train_dl:
            x = x.long()
            y = y.long()
            y_pred = model(x)

            optimizer.zero_grad()

            loss=crit(y_pred, y.argmax(dim=1))
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        val_loss, val_acc, val_rmse = validation_metrics(model, val_dl)
        if i % 5 == 0:
            print("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (sum_loss/total, val_loss, val_acc, val_rmse))

def validation_metrics (model, valid_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    for x, y in valid_dl:
        x = x.long()
        y = y.long()
        y_hat = model(x)
        y = y.argmax(dim=1)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        # print(pred, y)
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
    return sum_loss/total, correct/total, sum_rmse/total


class LSTM_fixed_len(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 8)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])


def main():
    # 데이터셋을 구축 #
    sents = [
        sent
        for sent, _ in DATA
    ]

    temp = [0]*8
    from copy import deepcopy
    labels = [
        label
        for _, label in DATA
    ]
    result = []
    for label in labels:
        temp_ = deepcopy(temp)
        temp_[label] = 1
        result.append(deepcopy(temp_))

    def get_class_distribution(y):
        count_dict = {
            "rating_0": 0,
            "rating_1": 0,
            "rating_2": 0,
            "rating_3": 0,
            "rating_4": 0,
            "rating_5": 0,
            "rating_6": 0,
            "rating_7": 0,
        }
        for i in y:
            if i == 0:
                count_dict['rating_0'] += 1
            elif i == 1:
                count_dict['rating_1'] += 1
            elif i == 2:
                count_dict['rating_2'] += 1
            elif i == 3:
                count_dict['rating_3'] += 1
            elif i == 4:
                count_dict['rating_4'] += 1
            elif i == 5:
                count_dict['rating_5'] += 1
            elif i == 6:
                count_dict['rating_6'] += 1
            elif i == 7:
                count_dict['rating_7'] += 1
            else:
                print("Check classes.")
        return count_dict

    class_count = [i for i in get_class_distribution(labels).values()]
    # print('class_count',class_count)
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
    # print('class_weights', class_weights)

    weighted_sampler = WeightedRandomSampler(
        weights=class_weights,
        num_samples=len(class_weights),
        replacement=True
    )


    # print(sents)
    # print('result', result)

    # 이제 뭐해요? #
    # 경서님 - 토큰화 && 정수인코딩. #
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(texts=sents)  # 정수인코딩 학습 #
    seqs = tokenizer.texts_to_sequences(texts=sents)

    #for seq in seqs:
    #    print('seq', seq)
    #    break

    seqs = pad_sequences(sequences=seqs, padding="post", value=0)
    #for seq in seqs:
    #    print(seq)
    #
    vocab_size = len(tokenizer.word_index.keys())
    vocab_size += 1  # 왜 이걸 해줘야할까?

    train_ds = ReviewsDataset(seqs[:800], result[:800])
    valid_ds = ReviewsDataset(seqs[800:], result[800:])

    batch_size = 100

    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=weighted_sampler)
    val_dl = DataLoader(valid_ds, batch_size=batch_size)


    # vocab_size, embedding_dim, hidden_dim
    model_fixed = LSTM_fixed_len(vocab_size, 50, 20)

    # model = train_model(model_fixed, epochs=30, lr=0.01)
    train_model(model_fixed, train_dl, val_dl, epochs=100, lr=0.001)

    #
    #model1 = nn.Embedding(vocab_size, 50, padding_idx=0)
    #
    #
    #model2 = nn.LSTM(input_size=50, hidden_size=15, num_layers=1, batch_first=True)


    #x = torch.tensor(seqs)

    #out1 = model1(x)
    #print()
    #print('out1 size()', out1.size())
    #print()

    #out2 = model2(out1)
    #out2 = torch.tensor(out2)
    #print('out2', out2)
    #print('out2 size()', out2.size())


if __name__ == '__main__':
    main()