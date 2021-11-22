<h2>노래 주제 분류</h2>
<div>
    <ol>
        <li>CBow 를 통해 Embedding Vector(Doc2Vec)를 만든다.</li>
        <li>Auto Encoder 를 사용하여 각 가사의 Embedding Vector 의 차원을 축소한다.</li>
        <li>차원 축소된 벡터를 k-means를 이용하여 군집화(라벨링)한다.</li>
        <li>도출된 라벨링 벡터를 이용하여 다시 각 가사에 대한 LSTM을 수행한다.</li>
    </ol>
</div>
<div>
    <pre>
        Model : LSTM
        Input : 라벨링된 이별노래 가사 Dataset
        Output : 이별노래 가사 Label
        Val Acc : 좋지 않음
    </pre>
</div>
<div>
Insight
<ol>
    <li>
        왜 차원 축소를 하는가?
        <ul>
            <li>
                효율성 : pc에서 하는 경우라면 좋겠지만, 슈퍼컴이나 큰 컴퓨터에서 이를 추진하면 좋다.
            </li>
            <li>
            정확도에서는 차원축소는 손해가 된다.
            </li>
            <li>
            다차원을 가지고 군집하는 것이 좋다.
            </li>
            <li>
            버트 임베딩 차원의 옵션을 반, 반의반을 사용하는 것을 해보자.
            </li>
            <li>
            코버트를 사용하는 것이 3% 정도 나아질 것 가다.
            </li>
        </ul>
    </li>
    <li>
        방향 설정
<ul>
<li>
우선 잘되는 클러스터링을 찾아서 Baseline을 정하자
</li>
<li>
주제가 갈라지는 것이 나오면, 이를 베이스라인으로 잡고, 이를 수정하는 방법으로 진행하자.
</li>
<li>
클러스터 불균형 분제를 효율적으로 해결하기 : 해결책을 하나씩 접근하자.
</li>
<li>
TO DO
    <ul>
        <li>예를 들어 이별 등의 단어 체크</li>
        <li>토픽 모델링</li>
        <li>장르 등의 메타 데이터를 포함</li>
        <li>컬렉션, 토픽 </li>
        <li>원인 분석 후 클러스터링을 반복하기</li>
        <li>토픽 모델링: LSD 등</li>
        <li>DMR 진화된 모델</li>
    </ul>

</li>
</ul>
    </li>
    
</ol>
</div>
<pre>
    텍스트 클러스터링에서 중요한 것은 사람이 군집의 잘됨을 판단해서 검증해야 함. 
</pre>