
"""
lstm.
long short term memory.
참고: https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091
"""
import torch
from torch.nn import functional as F
from typing import List, Tuple
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import pandas as pd
import MeCab

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
df = pd.read_csv('./발라드.csv')
df2 = pd.read_csv('./latent_vector_label.csv')
text = df['lyrics'].values
label =df2['label'].values
wakati = MeCab.Tagger("-Owakati")
DATA = [(wakati.parse(data), lb) for (data, lb) in zip(text, label)]

class SimpleLSTM(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, embed_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        # 학습해야하는 가중치가 무엇 무엇이 있었나?
        # 입력.. 출력을 어떻게 해야하나...?
        # 우선, 학습해야하는 가중치는 이 둘이다.
        self.E = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.W = torch.nn.Linear(in_features=hidden_size + embed_size, out_features=hidden_size*4)
        self.W_hy = torch.nn.Linear(in_features=hidden_size, out_features=1)
        torch.nn.init.xavier_uniform_(self.E.weight)
        torch.nn.init.xavier_uniform_(self.W.weight)
        torch.nn.init.xavier_uniform_(self.W_hy.weight)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # C_0 = torch.zeros(size=(X.shape[0], self.hidden_size))
        # H_0 = torch.zeros(size=(X.shape[0], self.hidden_size))  # (N, H)
        # X = self.E(X)  # (N, L) -> (N, L, E)
        # X_1 = X[:, 0]  # (N, L, E) -> (N, E)
        # Concat = torch.cat([H_0, X_1], dim=1)   # (N, H) cat (N, E) -> (N, H + E).
        # Gates = self.W(Concat)  # (N, H + E) * ???  (H + E, H*4) -> (N, H + H + H + H).
        # I_1 = torch.sigmoid(Gates[:, :self.hidden_size])  # (N, H*4) -> (N, H)
        # F_1 = torch.sigmoid(Gates[:, self.hidden_size:self.hidden_size*2])  # (N, H*4) -> (N, H)
        # O_1 = torch.sigmoid(Gates[:, self.hidden_size*2:self.hidden_size*3])  # (N, H*4) -> (N, H)
        # G_1 = torch.tanh(Gates[:, :-self.hidden_size])  # (N, H*4) -> (N, H)
        # # C_t = f * C_t-1 +  i * g
        # C_1 = F_1 * C_0 + I_1 * G_1
        # H_last = O_1 * torch.tanh(C_1)
        #
        C_t = torch.zeros(size=(X.shape[0], self.hidden_size))  # (N, H)
        H_t = torch.zeros(size=(X.shape[0], self.hidden_size))  # (N, H)
        for time in range(X.shape[1]):
            X_t = X[:, time]  # (N, L) -> (N, 1)
            X_t = self.E(X_t)  # (N, 1) -> (N, E)
            Concat = torch.cat([H_t, X_t], dim=1)  # (N, H) cat (N, E) -> (N, H + E).
            Gates = self.W(Concat)  # (N, H + E) * ???  (H + E, H*4) -> (N, H + H + H + H).
            I_t = torch.sigmoid(Gates[:, :self.hidden_size])  # (N, H*4) -> (N, H)
            F_t = torch.sigmoid(Gates[:, self.hidden_size:self.hidden_size*2])  # (N, H*4) -> (N, H)
            O_t = torch.sigmoid(Gates[:, self.hidden_size*2:self.hidden_size*3])  # (N, H*4) -> (N, H)
            G_t = torch.tanh(Gates[:, self.hidden_size*3:self.hidden_size*4])  # (N, H*4) -> (N, H)
            # C_t = f * C_t-1 +  i * g
            C_t = torch.mul(F_t, C_t) + torch.mul(I_t, G_t)  # component-wise mul.
            H_t = torch.mul(O_t, torch.tanh(C_t))  # component-wise mul
        # 마지막 hidden state를 출력.
        H_last = H_t
        return H_last

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        H_t = self.forward(X)  # (N, L) -> (N, H)
        y_pred = self.W_hy(H_t)  # (N, H) -> (N, 1)
        y_pred = torch.sigmoid(y_pred)  # (N, H) -> (N, 1)
        return y_pred

    def training_step(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_pred = self.predict(X)  # (N, L) -> (N, 1)
        y_pred = torch.reshape(y_pred, y.shape)  # y와 차원의 크기를 동기화
        loss = F.binary_cross_entropy(y_pred, y)  # 이진분류 로스
        loss = loss.sum()  # 배치 속 모든 데이터 샘플에 대한 로스를 하나로
        return loss


# builders #
def build_X(sents: List[str], tokenizer: Tokenizer, max_length: int) -> torch.Tensor:
    seqs = tokenizer.texts_to_sequences(texts=sents)
    seqs = pad_sequences(sequences=seqs, padding="post", maxlen=max_length, value=0)
    return torch.LongTensor(seqs)


def build_y(labels: List[int]) -> torch.Tensor:
    return torch.FloatTensor(labels)


EPOCHS = 20
LR = 0.001
HIDDEN_SIZE = 32
EMBED_SIZE = 12
MAX_LENGTH = 30


class Analyser:

    def __init__(self, lstm: SimpleLSTM, tokenizer: Tokenizer, max_length: int):
        self.lstm = lstm
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, text: str) -> float:
        X = build_X(sents=[text], tokenizer=self.tokenizer, max_length=self.max_length)
        y_pred = self.lstm.predict(X)
        return y_pred.item()


def main():
    sents = [sent for sent, label in DATA]
    labels = [label for sent, label in DATA]
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(texts=sents)
    X = build_X(sents, tokenizer, MAX_LENGTH)
    y = build_y(labels)

    vocab_size = len(tokenizer.word_index.keys())
    vocab_size += 1
    lstm = SimpleLSTM(vocab_size=vocab_size,hidden_size=HIDDEN_SIZE, embed_size=EMBED_SIZE)
    optimizer = torch.optim.Adam(params=lstm.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        loss = lstm.training_step(X, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()  #  다음 에폭에서 기울기가 축적이 되지 않도록 리셋
        print(epoch, "-->", loss.item())

    analyser = Analyser(lstm, tokenizer, MAX_LENGTH)
    print("##### TEST #####")
    sent = "나는 자연어처리가 좋아"
    print(sent, "->", analyser(sent))

    print("##### training acc ####")
    correct = 0
    total = len(DATA)
    for sent, y in DATA:
        y_pred = analyser(sent)
        print(sent, y, y_pred)
        if (y_pred >= 0.5) and y:
            correct += 1
        elif (y_pred < 0.5) and not y:
            correct += 1

    print("training acc:", correct / total)


if __name__ == '__main__':
    main()