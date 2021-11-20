"""
감성분석을 할 수 있는 charRNN을 만드는 것.
analyser("나는 자연어처리가 좋아") -> 확률
"""
from typing import List, Tuple
import torch
import torch.nn as nn
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from torch import optim
from torch.nn import functional as F
import pandas as pd
import MeCab
import tqdm

class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x


df = pd.read_csv('./발라드.csv')
df2 = pd.read_csv('./latent_vector_label.csv')
text = df['lyrics'].values
label =df2['label'].values
wakati = MeCab.Tagger("-Owakati")
DATA = [(wakati.parse(data), lb) for (data, lb) in zip(text, label)]

# 데이터셋을 구축 #
X = [
    sent
    for sent, _ in DATA
]
y = [
    label
    for _, label in DATA
]

# 이제 뭐해요? #
# 경서님 - 토큰화 && 정수인코딩. #
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(texts=X)  # 정수인코딩 학습 #
seqs = tokenizer.texts_to_sequences(texts=X)

for seq in seqs:
    print('seq', seq)
    break

EPOCHS = 300
BATCH_SIZE = 16
LEARNING_RATE = 0.0007

NUM_CLASSES = 8


# 그럼 이제 뭐해요? #
# 정수인코딩의 나열의 길이가 다 다르다 #
# 문제: 하나의 행렬 연산으로 다 계산할 수가 없다. #
# 패딩을 해서, 나열의 길이를 통일한다. #
seqs = pad_sequences(sequences=seqs, padding="post", value=0)
NUM_FEATURES = len(seqs[0])
# for seq in seqs:
#     print(seq)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = MulticlassClassification(num_feature = NUM_FEATURES, num_class=NUM_CLASSES)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(model)


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc) * 100

    return acc

accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}

print("Begin training.")
print('seq size', len(seqs))
for e in range(1, EPOCHS + 1):

    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    print('seq length', len(seqs))
    print('seq[0] length', len(seqs[0]))
    for X_train_batch, y_train_batch in zip(seqs, y):
        X_train_batch = torch.FloatTensor([X_train_batch])
        y_train_batch = torch.FloatTensor([y_train_batch])
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()

        print(X_train_batch.size())
        y_train_pred = model(X_train_batch)

        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = multi_acc(y_train_pred, y_train_batch)

        train_loss.backward()
        optimizer.step()

        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()

    # VALIDATION
    with torch.no_grad():

        val_epoch_loss = 0
        val_epoch_acc = 0

        model.eval()
        for X_val_batch, y_val_batch in zip(seq,y) :
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

            y_val_pred = model(X_val_batch)

            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)

            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
    loss_stats['train'].append(train_epoch_loss / len(X))
    loss_stats['val'].append(val_epoch_loss / len(X))
    accuracy_stats['train'].append(train_epoch_acc / len(X))
    accuracy_stats['val'].append(val_epoch_acc / len(X))

    print(
        f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(X):.5f} | Val Loss: {val_epoch_loss / len(X):.5f} | Train Acc: {train_epoch_acc / len(X):.3f}| Val Acc: {val_epoch_acc / len(X):.3f}')
