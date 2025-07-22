---
date: 2025-07-04
tags:
  - ai
  - computer_vision
  - deep_learning
---
# 1. 주목
## 1.1. 고전 주목 알고리즘
### 특징 선택
특징 선택<sub>feature selection</sub>은 쓸모가 많은 특징은 나기고 나머지는 제거하는 작업이다. 
특징 선택은 분별력이 높은 특징에 주목한다고 해석할 수 있다. 주목 여부는 0(선택 안 함)과 1(선택함)로 표현하며 모든 샘플이 같은 선택을 공유한다. 
### 돌출 맵
주목을 위한 고전 방법에서도 사람이 특징을 설계한다. Itti는 컬러 대비, 명암 대비, 방향 대비 맵을 결합해 돌출 맵<sub>saliency map</sub>을 구성하는 방법을 제안한다.
돌출 맵은 주목해야 할 정도를 실수로 표현하고 입력 영상의 내용에 따라 주목할 곳을 결정하기 때문에 특징 선택에 비해 크게 개선되었지만 여전히 한계를 안고 있다. 주목할 특징을 사람이 설계하고 모든 데이터셋에 공통의 수작업 특징을 적용한다는 점이 한계의 근본 원인이다.
## 1.2. 딥러닝의 주목
### 컴퓨터 비전에서 주목의 발전 과정
2014년에 발표된 RAM<sub>Recurrent Attention Model</sub>은 딥러닝에 최초로 주목을 적용한 논문으로 평가된다. RAM은 순환 신경망을 이용해 입력 영상에서 주목할 곳을 순차적으로 알아내는 방법을 제안한다. 2015년에 발표된 STN<sub>Spatial Transformer Network</sub>은 특징 맵에 이동과 크기, 회전 변환을 적용해 주목할 곳을 정한다. 
![[Pasted image 20250704163356.png]]
2017년에 발표된 SENet<sub>Squeeze-and-Excite Network</sub>은 특징 맵의 어떤 채널에 주목할지 알아낸다. 주목이 없는 신경망 $m \times n \times k$ 특징 맵에 컨볼루션을 적용해 $\mathbf{X}$로 표기된 $m' \times n' \times k'$ 특징 맵을 만들어 다음 층에 전달한다. SENet은 Squeeze와 Excite, Scale 연산을 추가로 적용해 $\mathbf{X}$를 $\mathbf{X}'$로 변환하여 다음 층에 전달한다.
![[Pasted image 20250704164113.png]]
### 자기 주목
자기 주목<sub>self-attention</sub>에서는 영상을 구성하는 요소 상호 간에 주목을 결정한다. 
![[Pasted image 20250704165104.png]]
Non-local 신경망은 자기 주목을 컴퓨터 비전에 처음 도입한 모델이다. 출력 특징 맵 $\mathbf{y}$는 입력 특징 맵 $\mathbf{x}$와 같은 크기다. 
$$
y_i=\frac{1}{C(\mathbf{x})}\sum_{모든\ j}a(\mathbf{x}_i, \mathbf{x}_j)g(\mathbf{x}_j)
$$
# 2. 순환 신경망과 주목
## 2.1. 순환 신경망 기초
### 구조와 동작
순환 신경망<sub>RNN; Recurrent Neural Network</sub>의 새로운 점은 은닉 노드끼리 에지를 가진다는 것 뿐이다. 
![[Pasted image 20250704170630.png]]
은닉층에 에지를 추가한 순환 신경망이 문장을 처리하는 방법은 다음과 같다.
$$
\mathbf{o}=(\mathbf{x}^1,\mathbf{x}^2,\mathbf{x}^3, \cdots, \mathbf{x}^T)
$$
아래 식은 순환 신경망의 순간 $i$의 동작을 정의한다.
$$
\mathbf{h}^i=\tau_1(\mathbf{U}^3\mathbf{h}^{i-1}+\mathbf{U}^1\mathbf{x}^i)
$$
$$
\mathbf{o}^i=\tau_2(\mathbf{U}^2\mathbf{h}^i)
$$
### 장거리 의존과 LSTM
순환 신경망은 시간을 처리할 수 있는 능력을 갖추었지만 길이기 긴 샘플에서 한계를 보인다. 종종 앞쪽 단어와 멀리 뒤에 있는 단어가 밀접하게 상호작용해야 하는데 이를 장거리 의존<sub>long-range dependency</sub>이라 부른다. 긴 문장에서 장거리 의존을 제대로 처리하지 못하는 문제가 발생한다.
LSTM<sub>Long Short-Term Memory</sub>은 순환 신경망을 개조해 장거리 의존을 처리하는 능력을 강화한다. 신경망 곳곳에 입력과 출력을 열거나 막을 수 있는 게이트를 두어 선별적으로 기억하는 기능을 확보한다. 
![[Pasted image 20250707112229.png]]
### 서츠케버의 seq2seq 모델
seq2seq의 혁신성은 가변 길이의 문장을 가변 길이의 문장으로 변환할 수 있다는 사실에 있다. seq2seq 모델은 인코더와 디코더로 구성되어 있다. 학습할 때는 디코더의 입력 부분과 출력 부분에 참값을 배치한다. 다시 말해 정답에 해당하는 출력을 알려주는 교사 강요<sub>teacher forcing</sub> 방식을 사용한다. 하지만 추론 단계에서는 정답을 모르기 때문에 자기 회귀<sub>auto-regressive</sub> 방식으로 동작한다. 아래 식은 자기 회귀 방식의 동작을 정의한다.
$$
\mathbf{g}^i=\tau_1(\mathbf{U}^3 \mathbf{g}^{i-1}+\mathbf{U}^1 \mathbf{y}^{i-1})
$$
$$
\mathbf{o}^i=\tau_2 (\mathbf{U}^2 \mathbf{g}^i)
$$
이 모델은 언어 번역에 큰 공헌을 했지만 가장 큰 문제는 인코더의 마지막 은닉 상태만 디코더로 전달한다는 점이다. 따라서 인코더는 마지막 은닉 상태에 모든 정보를 압축해야 하는 부담이 있고 디코더는 풍부한 정보가 인코더의 모든 순간이 있음에도 불구하고 마지막 상태만 보고 문장을 생성해야 하는 부담이 있다.
## 2.2. query-key-value로 계산하는 주목
주목을 계산하는 방법은 여러 가지인데, 최근에는 query가 key와 유사한 정도를 측정하고 유사성 정보를 가중치로 사용해 value를 가중하여 합하는 방법을 주로 사용한다. query, key, value를 행렬 계산으로 표현하면 아래와 같다. 
$$
\mathbf{c}=\text{softmax}(\mathbf{qK}^T)\mathbf{V}
$$
## 2.3. 주목을 반영한 seq2seq 모델
### 바다나우 주목
바다나우는 위 식의 주목을 활용해 디코더가 인코더의 모든 상태 $\mathbf{h}^1, \mathbf{h}^2, \cdots, \mathbf{h}^5$에 접근할 수 있게 허용함으로써 성능을 향상한다. 
# 3. 트랜스포머
## 3.1. 기본 구조
![[Pasted image 20250707134731.png]]
트랜스포머는 인코더와 디코더로 구성된다. 인코더에서 Inputs라고 표시된 곳을 보면 순차적으로 입력하는 구조가 사라졌다. 트랜스포머는 모든 단어를 한꺼번에 입력한다. 단어는 단어 임베딩을 통해 $d_{model}$ 차원의 임베딩 벡터로 표현되는데 논문에서는 $d_{model}=512$를 사용했다. 단어의 위치 정보를 보완하려고 위치 인코딩<sub>positional encoding</sub>을 통해 임베딩 벡터에 위치 정보를 더한다. 이처럼 모든 단어를 한꺼번에 입력하는 방식은 자기 주목을 가능하게 한다. 
인코더에 주황색으로 표시한 MHA<sub>Multi-Head Attention</sub> 층과 하늘색으로 표시한 FF<sub>Feed Forward</sub>층이 있다. MHA층은 자기 주목을 수행한다. $h$개 헤드가 독립적으로 자기 주목을 수행하고 결과를 결합한다. 헤드를 여러 개 쓰는 이유는 성능을 높이려는 데 있다. MHA층의 출력은 FF층의 입력이 된다. FF층은 완전연결층이다. MHA층과 FF층 모두 Add&Norm을 적용하는데, Add는 지름길 연결, Norm은 층 정규화<sub>layer normalization</sub> 연산이다. 
오른쪽에 있는 디코더를 살펴보면 인코더와 마찬가지로 단어 임베딩과 위치 인코딩을 통해 변환된 단어들이 한꺼번에 입력된다. 디코더에는 MHA층과 FF층이 있다. 디코더도 층의 출력에 Add&Norm을 적용한다. 출력층은 softmax를 통해 확률 벡터를 출력한다. 
인코더에 배치된 MHA는 입력 문장을 구성하는 단어 사이의 주목을 처리한다. 같은 문장을 구성하는 단어끼리 주목한다는 뜻에서 자기 주목<sub>self-attention</sub>이라고 한다. 디코더에 있는 MHA는 출력 문장을 구성하는 단어끼리 주목을 처리하므로 역시 자기 주목이다. 
## 3.2. 인코더의 동작
### 단어 임베딩과 위치 인코딩
인코더는 입력 문장을 구성하는 단어를 단어 임베딩 과정을 거쳐 $d_{model}$ 차원의 벡터로 변환한다. 모든 단어를 한꺼번에 입력하기 위해 문장을 $T \times d_{model}$ 크기의 $\mathbf{S}$ 행렬로 표현한다. $T$는 단어의 개수다.
행렬 $\mathbf{S}$의 모든 행이 동시에 처리되기 때문에 그대로 입력하면 신경망은 순서 정보를 받지 못하는 셈이 된다. 이런 정보 손실을 보완할 목적으로 위치 정보를 표현한 행렬 $\mathbf{P}$를 $\mathbf{S}$ 행렬에 더해 $\mathbf{X}$ 행렬을 만들고 $\mathbf{X}$를 신경망에 입력한다. 이 과정을 위치 인코딩<sub>positional encoding</sub>이라고 한다.
$$
\mathbf{X}=\mathbf{S}+\mathbf{P}
$$
![[Pasted image 20250707144814.png]]
### 자기 주목
자기 주목은 위 식으로 만든 $\mathbf{X}$ 행렬을 가지고 인코더에 있는 MHA가 수행한다. 자기 주목은 입력 문장이 자기 자신에 주목하는 과정이므로 query와 key, value 모두 입력 문장을 표현한 $\mathbf{X}$ 행렬이다. 이때 query의 확장이 필요하다. query가 벡터인 경우를 위한 식을 query가 $T$개의 행을 가진 행렬 $\mathbf{Q}$로 확장하면 아래 식이 된다. 
$$
\mathbf{C}= \text{softmax}\mathbf{QK}^T \mathbf{V}
$$
자기 주목을 구현하는 가장 단순한 방법은 $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$를 모두 $\mathbf{X}$로 설정하고 위 식을 적용하는 것이다. 트랜스포머는 이런 단순한 방법을 쓰지 않는다. 대신 query와 key, value가 각자 가중치 행렬 $\mathbf{W}^Q$와 $\mathbf{W}^k$, $\mathbf{W}^V$를 가지고 $\mathbf{X}$를 변환해 사용한다. 아래 식은 트랜스포머가 자기 주목을 수행하는 과정이다.
$$
C = \text{softmax} \left(\frac{\mathbf{QK}^T}{\sqrt{d_{key}}}\right) \mathbf{V}
$$
$$
\mathbf{Q}=\mathbf{XW}^Q ,\ \mathbf{K}=\mathbf{XW}^K,\ \mathbf{V}=\mathbf{XW}^V
$$
### Multi-head 주목
인코더에 있는 MHA에는 여러 개의 헤드가 있다. 각 헤드에는 고유한 변환 행렬 $\mathbf{W}^Q$, $\mathbf{W}^K$, $\mathbf{W}^V$를 가지고 위 식의 자기 주목을 독립적으로 수행해 상호 보완한다. 트랜스포머는 이렇게 여러 개의 헤드로 자기 주목을 수행한 결과를 결합해 성능 향상을 꾀하는 전략을 쓴다.
헤드마다 고유한 변환 행렬이 있으므로 $i$번째 헤드의 행렬을 $\mathbf{W}_i^Q$, $\mathbf{W}_i^K$, $\mathbf{W}_i^V$와 같이 표기한다. 아래 식은 MHA의 동작을 정의한다.
$$
\mathbf{C}=\text{Concatenate}(\mathbf{C}_1,\mathbf{C}_2, \cdots, \mathbf{C}_h) \mathbf{W}^O
$$
$$
\mathbf{C}_i=\text{softmax} \left(\frac{\mathbf{Q}_i \mathbf{K}_i^T}{\sqrt{d_{key}}} \right)\mathbf{V}_i
$$
$$
\mathbf{Q}_i=\mathbf{XW}_i^Q,\ \mathbf{K}_i=\mathbf{XW}_i^K,\ \mathbf{V}_i=\mathbf{XW}_i^V,\ i=1,2, \cdots, h
$$
MHA층을 통해 구한 $\mathbf{C}$ 행렬은 Add&Norm층을 통과한다. 이 층은 아래 식으로 지름길 연결(Add)과 층 정규화(Norm)를 수행한다. 지름길 연결은 MHA층의 출력 특징 맵 $\mathbf{C}$와 층에 입력되었던 특징 맵 $\mathbf{X}$를 더해주는 연산이다. 아래 그림은 층 정규화<sub>layer normalization</sub>와 배치 정규화<sub>batch normalization</sub>를 설명한다. 트랜스포머는 층 정규화를 적용하는데, 층 정규화는 미니 배치를 구성하는 샘플별로 특징 맵의 평균이 0, 표준편차가 1이 되도록 정규화한다. 
![[Pasted image 20250708113333.png]]
### 위치별 FF층
위치별 FF층<sub>position-wise FeedForward layer</sub>은 MHA층의 식이 출력된 $\mathbf{X}'$를 입력으로 받아 $\mathbf{X}''$를 출력한다. 이처럼 트랜스포머는 신경망을 흐르는 텐서의 모양을 유지해 행렬 연산이 원활히 이루어지도록 설계되어 있다. 아래 식은 FF층의 연산을 정의한다. 이 층은 다층 퍼셉트론과 같다.
$$
\mathbf{X}''=\text{FFN}(\mathbf{X}')=\text{ReLU}(\mathbf{X}'\mathbf{W}_1+\mathbf{b}_1)\mathbf{W}_2+\mathbf{b}_2
$$
### 학습이 알아내야 할 가중치
인코더에 있는 가중치를 살펴봄으로써 인코더의 학습에 대해 생각해보자. 위치 인코딩에서 $\mathbf{P}$를 학습으로 알아낼 수도 있는데 원래 트랜스포머는 위치 행렬 $\mathbf{P}$를 구하기 때문에 가중치가 없다. 
이제 MHA층의 가중치 개수를 따져보자. 트랜스포머 논문이 실험에 사용한 하이퍼 매개변수는 $d_{model}=512$고 헤드를 8개 사용해 $h=8$이다. 따라서 $d_{key}=d_{value}=64$다. 
FF층은 가중치 집합으로 $\mathbf{W}_1$과 $\mathbf{W}_2$를 가진다. 각각 $d_{model} \times d_{ff}$와 $d_{ff} \times d_{model}$개의 가중치를 가지는데 실험에서 $d_{model}=512$, $d_{ff}=2048$을 사용하므로 FF층은 $512 \times (2048+1)+2048 \times (512+1)=2,099,712$개의 가중치를 가진다.
## 3.3. 디코더의 동작
### Masked MHA층
Masked MHA층은 행렬 일부를 마스크로 가린다는 점을 빼면 모든 과정이 인코더의 MHA층과 같다. 
언어 번역이란 입력 문장 `저게 저절로 붉어질 리 없다`가 들어오면 출력 문장 `That can't turn red by itself`를 알아내는 일이다. 추론(예측) 단계에서는 입력 문장은 알고 있지만 출력 문장을 모른 채 동작해야 한다. 따라서 예측하는 일을 재귀적으로 수행하는 자기 회귀 방식을 적용해야 한다.
학습 단계는 추론 단계와 상황이 다르다. 추론 단계는 출력 문장을 모른 채 동작해야 하지만 학습 단계는 입력 문장과 출력 문장이 모두 주어진다. 학습 단계도 추론 단계처럼 자기 회귀 방식을 쓸 수 있는데 실제로는 교사 강요<sub>teacher forcing</sub> 방식을 주로 사용한다. 학습 단계에서 교사 강요를 쓰면 학습은 교사 강요, 추론은 자기 회귀를 사용하는 차이로 인해 문제가 생길 수 있다. 하지만 학습 단계에서 자기 회귀를 사용할 때 학습이 잘 안 되는 문제가 두 방식의 차이로 생기는 문제보다 심각하기 때문에 실제로는 교사 강요를 사용한다. 
### 인코더와 연결된 MHA층
디코더의 MHA층은 인코더와 디코더가 상호작용하는 곳이다. 인코더의 MHA층은 입력 문장 내의 단어끼리 주목을 처리하는 자기 주목이고 디코더의 Masked MHA층은 출력 문장 내의 단어끼리 주목을 처리하는 자기 주목이다. 반면에 디코더의 MHA층은 입력된 문장의 단어가 인코더로 입력된 문장의 단어에 주목할 정보를 처리한다.
MHA 동작 식은 자기 주목을 처리하므로 같은 문장에서 구한 행렬 $\mathbf{X}$만 있다. 따라서 query와 key, value를 계산하는 데 모두 $\mathbf{X}$를 사용했다. 그런데 디코더의 MHA층에는 인코더에서 온 $\mathbf{X}$와 디코더에서 온 $\mathbf{X}$가 있다. 디코더의 MHA층에서 query는 $\mathbf{X}_{dec}$를 사용하고 key와 value는 $\mathbf{X}_{enc}$를 사용한다. 이렇게 설정하면 언어 번역에서 출력 문장의 정보를 가진 query가 입력 문장의 정보를 가진 key와 value에 주목하는 정도를 계산할 수 있다. 
$$
\mathbf{C}=\text{Concatenate}(\mathbf{C}_1, \mathbf{C}_2, \cdots, \mathbf{C}_h)\mathbf{W}^O
$$
$$
\mathbf{C}_i=\text{softmax} \left( \frac{\mathbf{Q}_i \mathbf{K}_i^T}{\sqrt{d_{key}}} \right) \mathbf{V}_i
$$
$$
\mathbf{Q}_i=\mathbf{X}_{dec} \mathbf{W}_i^Q,\ \mathbf{K}_i=\mathbf{X}_{enc}\mathbf{W}_i^K,\ \mathbf{V}_i=\mathbf{X}_{enc} \mathbf{W}_i^V,\ i=1,2, \cdots, h
$$
### 학습이 알아내야 할 가중치
디코더에 있는 MHA층 2개의 가중치 개수는 인코더의 MHA층과 같고 디코더의 FF층 가중치 개수는 인코더의 FF층과 같다.
# 4. 비전 트랜스포머
트랜스포머가 발표된 이후 컨볼루션 신경망에 트랜스포머를 접목시키려는 여러 시도가 재빠르게 이루어졌다. 초기에는 컨볼루션 신경망을 백본으로 두고 트랜스포머로 추출한 자기 주목 정보를 보조로 사용하는 방식을 사용했다. 그중 아래 그림은 주목 증강 컨볼루션 신경망<sub>attention augmented convolutional network</sub>의 구조를 보여준다. 기존 컨볼루션 신경망은 컨볼루션 연산을 통해 특징 맵을 만들어 다음 층으로 전달하는데, 자기 주목으로 증강한 신경망에서는 컨볼루션으로 만든 특징 맵과 트랜스포머의 MHA로 만든 특징 맵을 결합해 다음 층에 넘긴다. 
![[Pasted image 20250708162805.png]]
2020년에는 컨볼루션 신경망에 트랜스포머를 어정쩡하게 결합하는 형태를 벗어나려는 시도가 다양하게 이루어졌다. 새로운 방식에서는 트랜스포머가 백본이고 컨볼루션 신경망이 보조하거나 또는 컨볼루션을 아예 들어내고 주목으로만 구성된 트랜스포머 방식을 사용한다. 트랜스포머가 주도하는 이런 모델은 분류, 검출, 분할, 추적, 자세 추정 등의 컴퓨터 비전 문제에서 큰 성공을 거두었다. 최근에는 한 모델이 분류, 검출, 분할, 추적 문제를 모두 해결하는 방식으로 진화했는데, 스윈 트랜스포머<sub>Swin transformer</sub>가 대표적이다. 
컴퓨터 비전에 트랜스포머를 적용하는 연구의 폭과 너비가 빠르게 확대되자 연구 결과를 잘 정리한 서베이 논문이 발표되었다. 컴퓨터 비전에 적용된 트랜스포머 모델을 통칭해 비전 트랜스포머<sub>vision transformer</sub>라 부른다.
## 4.1. 분류를 위한 트랜스포머
### ViT
ViT는 트랜스포머가 컴퓨터 비전의 기본 문제인 분류 문제를 얼마나 잘 푸는지 확인할 목적으로 트랜스포머 구조를 최대한 그대로 따라 실험한다. 원래 트랜스포머는 언어 번역을 위해 개발되었기 때문에 문장을 입력 받아 문장을 출력하는 구조, 즉 인코더는 입력 문장을 처리하고 디코더는 출력 문장을 처리하는 구조를 가진다. 문장은 1차원 구조의 데이터인데 영상은 화소가 2차원 구조로 배열된 데이터이기 때문에 트랜스포머의 구조를 적절히 변경해야 한다. 
영상 분류에서는 영상을 입력 받아 부류 확률 벡터를 출력하면 된다. 따라서 인코더와 디코더를 가진 트랜스포머에서 인코더만 있으면 된다. 인코더는 디코더로 텐서를 전달하는 대신 자신이 추출한 특징 맵을 부류 확률 벡터로 변환해 출력층을 통해 출력하면 된다. 
![[Pasted image 20250708172555.png]]
지금까지 설명한 비전 트랜스포머는 부류 확률 벡터를 출력하면 되기 때문에 컴퓨터 비전이 사용하는 트랜스포머 중에서 가장 단순한 편이다. 검출 문제의 경우 출력이 가변 개수의 박스이기 때문에 인코더와 디코더를 모두 사용해야 하며 적절한 출력 형태를 고안하고 손실 함수를 설계해야 한다. 
### ViT 실습
``` python
import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

n_class = 10
img_siz = (32, 32, 3)

patch_siz = 4
p2 = (img_siz[0]//patch_siz)**2
d_model = 64
h=8
N=6

class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, p2, d_model):
        super(PatchEncoder, self).__init__()
        self.p2=p2
        self.projection = keras.layers.Dense(units=d_model)
        self.position_embedding = keras.layers.Embedding(input_dim=p2, output_dim=d_model)
    
    def call(self, patch):
        positions = tf.range(start=0, limit=self.p2, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
def create_vit_classifier():
    input=keras.layers.Input(shape=img_siz)
    nor = keras.layers.Normalization()(input)

    patches= Patches(patch_siz)(nor)
    x = PatchEncoder(p2, d_model)(patches)

    for _ in range(N):
        x1 = keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x2 = keras.layers.MultiHeadAttention(num_heads=h, key_dim=d_model//h, dropout=0.1)(x1, x1)
        x3 = keras.layers.Add()([x2, x])
        x4 = keras.layers.LayerNormalization(epsilon=1e-6)(x3)
        x5 = keras.layers.Dense(units=d_model*2, activation=tf.nn.gelu)(x4)
        x6 = keras.layers.Dropout(0.1)(x5)
        x7 = keras.layers.Dense(d_model, activation=tf.nn.gelu)(x6)
        x8 = keras.layers.Dropout(0.1)(x7)
        x = keras.layers.Add()([x8, x3])
    
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(2048, activation=tf.nn.gelu)(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(1024, activation=tf.nn.gelu)(x)
    x = keras.layers.Dropout(0.5)(x)
    output = keras.layers.Dense(n_class, activation='softmax')(x)

    model = keras.Model(inputs=input, outputs=output)
    return model

model = create_vit_classifier()
model.layers[1].adapt(x_train)

model.compile(optimizer=keras.optimizers.Adam(), 
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test), verbose=1)

res = model.evaluate(x_test, y_test, verbose=0)
print('정확률 =', res[1]*100)

import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'], label='train accuracy')
plt.plot(hist.history['val_accuracy'], label='test accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['train accuracy', 'test accuracy'])
plt.grid()
plt.show()

plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label='test loss')
```
```
정확률 = 72.63000011444092
```
![[Pasted image 20250708183128.png]]
## 4.2. 검출과 분할을 위한 트랜스포머
### DETR
faster RCNN과 같은 컨볼루션 신경망은 박스 집합을 바로 예측하기 어렵기 때문에 영역 후보를 생성한 다음 영역을 분류하는 우회 방법을 쓴다. 다시 말해 원래 문제를 대리로 바뿌고 대리 문제를 푸는 셈이다. Carion은 트랜스포머를 이용해 바로 집합 예측을 하는 DETR<sub>DEtection TRansformer</sub>이라는 새로운 모델을 제안한다. DETR은 후보 영역을 생성하는 단계가 없고 비최대 억제라는 후처리 단계가 없기 때문에 통째 학습이 가능하다.
DETR은 3개의 모듈로 구성된다. 1번째 모듈은 컨볼루션 신경망으로 특징을 추출한다. $m \times n \times 3$ 컬러 영상이 들어오면 ResNet을 거쳐 $h \times w \times C$ 특징 맵으로 변환한다. 이 텐서에 $1 \times 1$ 컨볼루션을 적용해 채널의 개수를 $C$에서 $d_{model}$로 줄여 $h \times w \times d_{model}$ 텐서로 변환한다. $h \times w$에 있는 화소를 Flatten으로 이어 붙여 행에 배치하면 $hw \times d_{model}$ 행렬을 얻는다. 이렇게 하면 트랜스포머에 입력할 $T \times d_{model}$ 행렬을 얻는다. 자기 주목을 계산할 때 행이 query로 참여하기 때문에 축소된 화소끼리 주목 정보를 추출하는 셈이다. 축소된 화소를 패치로 간주하기 때문에 DETR은 자기 주목을 반영한다고 볼 수 있다. 
두 번째 모듈은 인코더와 디코더로 구성된 트랜스포머다. 첫 번째 모듈에 입력된 $T \times d_{model}$ 행렬에 위치 인코딩을 적용해 첫번째 인코더 블록에 입력한다. 첫번째 인코더 블록, 두번째 인코더 블록, ... , M번째 인코더 블록을 통해 변환된 행렬은 디코더의 MHA에 입력되어 key와 value 역할을 한다. 원래 트랜스포머에서는 디코더가 자기 회귀 방식으로 작동한다. 문장은 단어가 순서를 미루지만 물체 박스는 순서가 없어 집합으로 표현해야만 한다. 따라서 디코더에 자기 회귀를 적용할 수 없다. 디코더의 최초 입력인 object queries는 박스에 대한 아무런 정보가 없기 때문에 위치 인코딩을 가지고 출발한다. DETR은 학습을 통해 알아낸 $\mathbf{P}$를 사용한다.
3번째 모듈에는 디코더 블록이 출력한 $K \times d_{model}$ 행렬의 각 행을 완전연결층(FF)에 통과시켜 박스 정보로 변환한다. 이렇게 모델이 예측한 박스 집합을 $\hat{y}$로 표기한다. 정답에 해당하는 참값 박스 집합을 $y$로 표기한다. $y$에 $\varnothing$ 박스를 추가해 $\hat{y}$처럼 $K$개 박스를 가지게 한다. $\hat{y}$과 $y$에 헝가리안 알고리즘을 적용해 최적 매칭 쌍을 구한다. 매칭된 참값 박스 $i$와 예측 박스 $j$ 쌍을 이용해 아래 식으로 손실 함수를 정의한다.
$$
J(y,\hat{y})=\sum_{(i,j)\in 매칭\ 쌍}(-\log \hat{p}_j(c_i)+\lambda_1 \text{IoU}(b_i, \hat{b}_j)+\lambda_2 ||b_i-\hat{b}_j||_1)
$$
![[Pasted image 20250709100947.png]]
### 분할을 위한 비전 트랜스포머
DETR은 물체를 검출할 목적으로 개발되었는데, 트랜스포머의 특성으로 인해 분할을 할 수 있도록 쉽게 확장할 수 있다. 위의 3번째 모듈은 출력 행렬을 해석해 박스 정보로 변환하는 검출 용도의 헤드다. 검출 헤드를 떼내고 분할 헤드를 붙이면 분할 용도의 DETR로 변신한다. 
SETR<sub>SEgmentation TRansformer</sub>은 $m \times n \times 3$의 컬러 영상을 ViT에서처럼 패치로 나누고 임베딩 과정을 거쳐 $\left(\frac{m}{16} \frac{n}{16}\right)\times d_{model}$ 크기의 행렬로 변환해 트랜스포머 인코더에 입력한다. 여러 개의 인코더 블록을 거쳐 나온 $\left(\frac{m}{16} \frac{n}{16}\right)\times d_{model}$ 크기의 행렬을 특징 맵으로 받아 디코더에 입력한다. 요약하면 SETR에서 인코더는 트랜스포머를 사용하고 디코더는 컨볼루션 신경망을 사용하는 구조다. 
## 4.3. 백본 트랜스포머
컨볼루션 신경망으로 전이 학습<sub>transfer learning</sub>을 할 수 있다는 점은 딥러닝의 큰 장점이다. 지역 정보를 추출하는 컨볼루션 신경망에 비해 자기 주목에 의존하는 트랜스포머가 전이 학습에 훨씬 유리하다는 이론적인 근거와 실험적 입증이 많이 발표되었다. 
스윈 트랜스포머<sub>Swin transformer</sub>는 분류, 검출, 분할, 추적 등에 두루 사용할 수 있는 백본 비전 트랜스포머를 목표로 설계되었다. 
백본이 되려면 영상의 특성을 충분히 반영한 구조를 설계하는 일이 핵심이다. 원래 트랜스포머는 문장을 처리할 목적으로 문장과 영상은 근본 특성이 크게 다르다. 문장을 구성하는 단어는 스케일 변화가 없는데 영상을 구성하는 물체는 아주 다양한 크기로 나타나 스케일 변화가 심하다. 또한 문장은 띄어쓰기를 통해 단어가 분할되어 있는데 영상을 구성하는 물체는 물체끼리 또는 물체와 배경이 심하게 섞여있다. 
### 계층적 특징 맵과 이동 윈도우
스윈 트랜스포머는 계층적 특징 맵과 이동 윈도우<sub>shifted window</sub>라는 두 가지 핵심 아이디어를 사용해 영상의 영상의 특성을 트랜스포머에 반영한다. 모든 층이 같은 크기의 패치를 사용하는 ViT와 달리 스윈 트랜스포머는 작은 패치부터 큰 패치로 구분해 처리한다. 
![[Pasted image 20250709113203.png]]
스윈 트랜스포머의 또 다른 핵심 아이디어는 이동 윈도우<sub>shifted window</sub>이다. 스윈이란 이름은 Shifted WINdow에서 유래한다. layer 1번째 층에서는 이전처럼 윈도우를 나누는데, 그 다음 layer 1+1번째 층에서는 윈도우를 절반 크기만큼 수평과 수직 방향으로 이동해 나눈다. 이동 윈도우를 사용하면 윈도우끼리 연결성이 강화되어 성능 향상이 나타난다.
![[Pasted image 20250709113617.png]]
### 스윈 트랜스포머의 구조
왼쪽에서 오른쪽 방향으로 흐르는 단계 1~4는 위 그림에서 아래에서 위로 진행하며 패치는 커지고 윈도우 개수는 작아지는 방향에 해당한다. 
$m \times n \times 3$ 영상이 입력되면 맨 아래층처럼 $4 \times 4$ 크기의 패치로 분할한다. $\frac{m}{4} \times \frac{n}{4}$개의 패치를 얻고 패치 하나는 $4 \times 4 \times 3=48$개 값을 가지므로 $\frac{m}{4} \times \frac{n}{4} \times 48$ 텐서가 된다.
1단계는 이 텐서에 선형 임베딩을 적용한다. 선형 임베딩은 $\frac{m}{4} \times \frac{n}{4} \times 48$에 $1\times 1$ 컨볼루션을 $C$번 적용해 $\frac{m}{4} \times \frac{n}{4} \times C$ 텐서로 변환한다. 이웃한 $M \times N$개 패치를 묶어 윈도우를 구성하면 $\frac{mn}{16M^2}$개의 윈도우를 얻는다. 각 윈도우는 $M^2=49$개 패치를 행에 배치한 $49 \times C$ 행렬로 표현되어 스위 트랜스포머 블록을 통과한다. 이때 각 윈도우는 독립적으로 처리된다. 
2단계는 1단계로부터 $\frac{m}{4} \times \frac{n}{4} \times C$ 텐서를 받아 패치 합치기를 적용한다. 패치 합치기는 맨 아래층에서 그 위층으로 진행하는 과정에 해당하는데, 단순히 이웃한 4개 패치를 하나의 패치로 합친다. 결과적으로 패치 크기는 $4 \times 4$에서 $8 \times 8$이 되고 윈도우는 $\frac{mn}{16M^2}$개에서 $\frac{mn}{64M^2}$개로 줄어든다. 같은 과정을 3단계와 4단계에 대해 적용하면 최종적으로 $\frac{m}{32} \times \frac{n}{32} \times 8C$ 텐서를 얻는다. 이 특징 맵에 분류를 위한 헤드를 붙이면 분류기가 되고 검출을 위한 헤드를 붙이면 분할기가 된다.
![[Pasted image 20250709114045.png]]
## 4.4. 자율지도 학습
BERT<sub>Bidirectional Encoder Representations from Transformers</sub>와 GPT-3<sub>Generative Pre-Trained transformer-3</sub>의 사전 학습 모델은 다양한 자연어 처리 응용에 성공적으로 활용되어 자연어 처리에서 없어서는 안될 백본 모델로 자리잡는다. 이런 성공을 떠받치는 토대는 트랜스포머의 확장성<sub>scalability</sub>이다. 확장성은 모델의 크기와 데이터셋의 크기가 커져도 원만하게 학습이 이루어지고 성능이 향상되는 특성을 뜻한다. 
### 자연어 처리를 위한 MLM
이런 거대한 모델을 학습하려면 거대한 데이터셋이 있어야 하는데, 사람이 레이블할 수 있는 양에는 한계가 있어 지도 학습<sub>supervised learning</sub>은 불가능하다. 대안은 레이블링이 안 된 데이터로 자율지도 학습<sub>self-supervised learning</sub>을 하는 것이다. 자연어 처리에서는 문장을 구성하는 단어 일부를 랜덤하게 선택해 가린 다음 모델이 알아내게 학습하는 마스크 언어 모델<sub>MLM; Masked Language Model</sub>을 주로 사용한다. 학습 알고리즘이 스스로 단어를 가린 다음에 그것을 레이블로 사용해 모델을 학습하기 때문에 자율지도라 부른다. 이외에도 같은 문서에서 연속된 두 문장을 뽑아 만든 쌍과 서로 다른 문서에서 뽑아 만든 쌍을 구별하도록 학습하는 다음 문장 예측<sub>NSP; Next Sentence Prediction</sub> 기법 등이 있다. 
### 컴퓨터 비전을 위한 MIM
컴퓨터 비전의 자율지도 학습에는 영상을 패치로 나누고 섞어 직소 퍼즐로 만든 다음 모델이 바로잡게 학습하는 기법, 명암 영상으로 바꾼 다음 컬러로 복원하도록 학습하는 방법 등 다양하다. 마스크 영상 모델<sub>MIM; Masked Image Modeling</sub>은 MLM과 비슷하게 영상의 일부를 가린 다음 모델이 알아내도록 학습하는 기법이다. 어디를 어떻게 가릴 지에 따라 여러 방식으로 구현할 수 있다. 
# 5. 비전 트랜스포머 프로그래밍 실습
## 5.1. ViT 프로그래밍 실습: 영상 분류
``` python
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

img = [Image.open('BSDS_242078.jpg'), Image.open('BSDS_361010.jpg'), Image.open('BSDS_376001.jpg')]

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

inputs = feature_extractor(img, return_tensors="pt")
res = model(**inputs)

for i in range(res.logits.shape[0]):
    plt.imshow(img[i])
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    predicted_label = int(torch.argmax(res.logits[i], dim=-1))
    prob = float(F.softmax(res.logits[i], dim=-1)[predicted_label] * 100)
    print(f'{i}번째 영상의 분류: {model.config.id2label[predicted_label]} ({prob:.2f}%)')
```
```
0번째 영상의 분류: umbrella (95.56%)
1번째 영상의 분류: croquet ball (99.97%)
2번째 영상의 분류: croquet ball (35.53%)
```
## 5.2. DETR 프로그래밍 실습: 물체 검출
``` python
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from PIL import Image
import numpy as np
import cv2 as cv

img = Image.open('BSDS_361010.jpg')

feature_extraction = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

inputs = feature_extraction(images=img, return_tensors="pt")
res = model(**inputs)

colors = np.random.uniform(0, 255, size=(100, 3))
im = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)

img_width, img_height = img.size

for i in range(res.logits.shape[1]):
    predicted_label = res.logits[0, i].argmax(-1).item()
    confidence = float(res.logits[0, i].softmax(dim=0)[predicted_label])
    
    if predicted_label != 91 and confidence > 0.5:
        name = model.config.id2label[predicted_label]
        prob = '{:.2f}%'.format(confidence * 100)
        
        cx, cy = int(img_width * res.pred_boxes[0, i, 0]), int(img_height * res.pred_boxes[0, i, 1])
        w, h = int(img_width * res.pred_boxes[0, i, 2]), int(img_height * res.pred_boxes[0, i, 3])
        
        cv.rectangle(im, (cx - w//2, cy - h//2), (cx + w//2, cy + h//2), colors[predicted_label], 2)
        
        cv.putText(im, name + ' ' + prob, (cx - w//2, cy - h//2 - 5), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, colors[predicted_label], 1)

cv.imshow('DETR', im)
cv.waitKey(0)
cv.destroyAllWindows()
```
![[Pasted image 20250709143956.png]]
## 5.3. CLIP 프로그래밍 실습: 멀티 모달
트랜스포머는 영상과 자연어처럼 특성이 서로 다른 데이터, 즉 멀티 모달<sub>multi-modal</sub> 데이터를 모델링하는 능력이 뛰어나다.
### CLIP: 영상과 자연어를 같이 모델링한 멀티 모달 모델
CLIP<sub>Contrastive Language-Image Pre-Training</sub>은 (image, text) 샘플을 웹에서 4억장 가량 모아 학습에 사용한다. 영상 인코더와 텍스트 인코더를 동시에 학습해 멀티 모달 임베딩 공간을 구성하는데, 옳은 쌍의 코사인 유사도는 최대화하고 틀린 쌍의 코사인 유사도는 최소화하도록 대조 학습<sub>contrastive learning</sub>을 적용한다. 
``` python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

img = Image.open('BSDS_361010.jpg')

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

captions = ["a woman holding a bat", "students are eating", "croquet playing", "golf playing"]
inputs = processor(text=captions, images=img, return_tensors="pt", padding=True)
res = model(**inputs)

import matplotlib.pyplot as plt
plt.imshow(img); plt.xticks([]); plt.yticks([])
plt.show()

logits = res.logits_per_image
probs = logits.softmax(dim=1)
for i in range(len(captions)):
    print(f'{i}번째 캡션: {captions[i]} ({probs[0][i].item() * 100:.2f}%)')
```
```
0번째 캡션: a woman holding a bat (0.09%)
1번째 캡션: students are eating (0.00%)
2번째 캡션: croquet playing (99.89%)
3번째 캡션: golf playing (0.02%)
```
# 6. 트랜스포머 특성
트랜스포머는 여러 과업에서 컨볼루션 신경망의 성능을 능가할 뿐만 아니라 전이 학습의 폭과 깊이를 놀라울 정도로 확장하고 있다. 이런 현상에 토대 모델<sub>foundation model</sub>이라는 이름을 붙여 새로운 조류를 설명하려는 연구 그룹이 생기고 있다.
### 장거리 의존
컨볼루션 신경망에서는 화소들이 컨볼루션 연산을 통해 이웃 화소와 정보를 교환한다. 주로 $3 \times 3$이나 $5 \times 5$의 작은 필터를 사용하기 때문에 영상 내에서 멀리 떨어져 있는 물체끼리 상호 작용하려면 층이 깊어야 한다. 컨볼루션 신경망과 순환 신경망은 모두 깊은 층을 통해 상호작용을 일으키는데 층이 깊어짐에 따라 여러 요소 사이에서 발생한 정보가 혼합되어 중요한 정보가 흐릿해진다.
트랜스포머는 시작부터 끝까지 자기 주목을 이용해 명시적으로 장거리 의존<sub>long-range dependency</sub>을 처리한다. 비전 트랜스포머는 자기 주목을 통해 특정 패치를 많이 주목하게 함으로써 영상을 이해한다.
컨볼루션 신경망과 순환 신경망은 어느 순간 이웃 요소끼리만 교류하는 지역성을 벗어나지 못한 반면에 트랜스포머는 입력 신호 전체를 한꺼번에 표현한 자기 주목 행렬을 통해 전역 정보를 처리한다. 트랜스포머는 전역 정보를 명시적으로 표현하고 처리함과 동시에 지역 정보 추출에도 능숙하다는 사실을 입증한 연구 결과가 있다.
### 확장성
트랜스포머는 모델의 크기와 데이터셋의 크기 측면에서 확장성<sub>scalability</sub>이 뛰어나다. 원래 트랜스포머가 4000만개를 조금 넘는 가중치를 가졌는데 불과 3년만에 GPT-3은 1750억개를 가지며 이후 1.6조개 매개변수를 가진 Switch 트랜스포머가 발표된다. 자연어 처리를 위한 트랜스포머에서 모델 크기, 데이터셋 크기, 학습에 투입한 계산 자원 사이의 관계를 체계적으로 실험 분석한 연구가 있다. 이 실험은 큰 모델을 사용하는 것이 유리하다는 근거를 제시한다.
자연어 처리와 달리 컴퓨터 비전은 자율지도 학습보다 지도 학습에 의존하는 경향이 커서 사람이 레이블링한 데이터셋 크기에 영향을 더 받는다. 비전 트랜스포머의 모델 크기는 인코더 블록의 개수를 늘리거나 패치 크기를 줄이거나 패치 인코딩 차원 $d_{model}$을 늘리거나 헤드의 개수를 늘리는 방법으로 키울 수 있다. 실험 결과로 모델 크기와 데이터셋 크기, 계산 시간을 같이 늘리면 확실하게 성능 향상이 나타나는 현상을 확인한다. 또한 모델 크기가 병목으로 작용한다는 사실을 밝힌다. 
### 설명 가능
트랜스포머는 영상을 구성하는 요소끼리 자기 주목을 명시적으로 표현하기 때문에 설명 가능을 구현하기에 훨씬 유리하다. 아래 그림은 미세 분류를 수행하는 TransFG라는 트랜스포머 모델이 새의 종을 분류하는 사례다. 미세 분류에서는 서로 다른 종인데 전체 모양이 아주 비슷해서 부리나 코끝처럼 특정한 곳에 집중해야 구별할 수 있는 경우가 많다.
![[Pasted image 20250709164301.png]]
### 멀티 모달
모든 기계학습 모델에는 귀납 편향<sub>inductive bias</sub>이 있다. 귀납 편향은 기계학습 모델이 학습 단계에서 보지 않았던 새로운 샘플을 옳게 추론하기 위해 사용하는 가정이다. 이웃 화소는 비슷한 특성을 가진다는 지역성<sub>locality</sub>과 물체가 이동하면 이동 위치에서 같은 특징이 추출된다는 이동 등변<sub>translation-equivariant</sub>은 컨볼루션 신경망이 사용하는 대표적인 귀납 편향이다. 컨볼루션 신경망의 귀납 편향은 강한 편이라 이를 만족하지 못하는 자연어 문장과 같이 다른 모달리티 데이터에 적용하는 일은 부자연스런 결과를 낳는다. 순환 신경망은 시간축에 관련된 강한 귀납 편향을 갖는다.
이들 모델에 비해 트랜스포머의 귀납 편향은 약한 편이다. 약한 귀납 편향은 트랜스포머를 여러 가지 입력 형태, 즉 멀티 모달 데이터에 적용할 여지를 열어준다.
사람은 멀티 모달<sub>multi-modal</sub>을 능숙하게 처리하고 유용하게 이용한다. 인공지능도 멀티 모달 연구를 활발하게 진행한다.
### 토대 모델
토대 모델<sub>foundation model</sub>은 전이 학습을 통해 컴퓨터 비전과 자연어 처리, 음성 분석 등의 넓은 범주의 응용을 지원할 수 있어야 하는데, BERT와 GPT-3, CLIP을 제시한다. 토대 모델은 그 자체로 응용이 완성되지 않았지만 전이 학습을 통해 다양한 과업에 적용할 수 있어야 하는 만큼, 기초가 튼튼하고 다양한 응용 헤드를 붙일 수 있는 유연함을 갖추어야 한다.