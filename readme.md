<h2>노래 주제 분류</h2>
1. CBow 를 통해 Embedding Vector(Doc2Vec)를 만든다.
2. Auto Encoder 를 사용하여 각 가사의 Embedding Vector 의 차원을 축소한다.
3. 차원 축소된 벡터를 k-means를 이용하여 군집화(라벨링)한다.
4. 도출된 라벨링 벡터를 이용하여 다시 각 가사에 대한 LSTM을 수행한다.

<pre>
    Model : LSTM
    Input : 라벨링된 이별노래 가사 Dataset
    Output : 이별노래 가사 Label
    Test Acc : 약 83%
</pre>