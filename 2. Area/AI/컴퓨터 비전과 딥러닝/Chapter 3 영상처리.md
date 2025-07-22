---
date: 2025-06-08
tags:
  - ai
  - computer_vision
  - deep_learning
---
# 1. 디지털 영상 기초
## 1.1. 영상 획득과 표현
디지털 카메라는 실제로 존재하는 피사체를 샘플링하고 명암을 양자화하는 과정을 통해 디지털 영상을 획득한다.
### 핀홀 카메라와 디지털 변환
핀홀 카메라(pinhole camera) 모델은 물체에서 반사된 빛이 카메라의 작은 구멍을 통해 안으로 들어가 뒷면에 있는 영상 평면(image plane)에 맺힌다.
![[Pasted image 20250716125446.png]]
빛이라는 아날로그 신호를 받은 CCD 센서<sub>이미지 센서</sub>는 디지털 신호로 변환한 영상을 메모리에 저장한다. 디지털로 변환하는 과정에서 샘플링<sub>sampling</sub>과 양자화<sub>quantization</sub>를 수행한다. 샘플링은 2차원 공간을 가로 방향으로 $N$개, 세로 방향으로 $M$개 구간으로 나눈다. 이렇게 형성된 한 점을 화소<sub>pixel</sub>라고 하고, $M \times N$을 영상의 크기<sub>size</sub> 또는 해상도<sub>resolution</sub>라고 한다.
![[Pasted image 20250716125817.png]]
### 영상 좌표계: (y,x)와 (x,y) 표기
가로 방향의 $x$축은 $0, 1, \cdots , N-1$의 정수 좌표를 가지며 세로 방향의 $y$축은 $0, 1, \cdots , M-1$의 정수 좌표를 가진다. $j$행 $i$열의 명암을 $f(j, i)$로 표기한다.
![[Pasted image 20250716130100.png]]
디지털 영상의 좌표는 원점이 왼쪽 위다. 또한 $(x, y)$ 대신 $(y, x)$ 표기를 사용한다. 따라서 가로 방향을 $y$축, 세로 방향을 $x$축으로 설정하고 $f(x, y)$ 표기를 사용한다.
## 1.2. 다양한 종류의 영상
명암 영상<sub>grayscale image</sub>은 2차원 구조의 배열로 표현된다. 컬러 영상은 RGB(red, green, blue) 3개 채널로 구성되어 있으며 3차원 구조의 배열로 표현된다.
![[Pasted image 20250716130557.png]]
## 1.3. 컬러 모델
**RGB 컬러 모델**은 빨강(R), 녹색(G), 파랑(B)의 세 요소를 섞으면 세상의 모든 색을 표현할 수 있다는 이론이다. 세 요소의 범위를 0부터 1로 설정해 세상의 모든 색을 RGB 큐브에 넣는다.
![[Pasted image 20250716130617.png]]
**HSV 컬러 모델**은 H은 색상, S는 채도, V는 명암을 표현한다. RGB 모델에서는 빛이 약해지면 R, G, B 값이 모두 작아진다. HSV 모델에서는 색상은 H 요소, 빛의 밝기 V 요소에 집중되어 있다.
![[Pasted image 20250716130825.png]]
## 1.4. RGB 채널별로 디스플레이
`numpy` 배열을 이용하여 RGB 영상의 일부를 R, G, B 채널로 분리한다. 여기서는 `ndarray` 클래스의 슬라이싱 기능을 사용하여 영상의 왼쪽 위 1/4만큼을 `img[0:img.shape[0]//2, 0:img.shape[1]//2, :]`로 지정한다.
![[Pasted image 20250716131036.png]]
``` python
import cv2
import sys

img = cv2.imread('soccer.jpg')

if img is None:
	sys.exit('파일을 찾을 수 없습니다.')
    
cv2.imshow('original_RGB', img)
cv2.imshow('Upper left half', img[0:img.shape[0]//2,0:img.shape[1]//2, :])
cv2.imshow('Center half', img[img.shape[0]//4:3*img.shape[0]//4, img.shape[1]//4:3*img.shape[1]//4,:])

cv2.imshow('R channel', img[:,:,2])
cv2.imshow('G channel', img[:,:,1])
cv2.imshow('B channel', img[:,:,0])

cv2.waitKey()
cv2.destroyAllWindows()
```
![[Pasted image 20250716131147.png]]
# 2. 이진 영상
**이진 영상**<sub>binary image</sub>은 화소가 0(흑) 또는 1(백)인 영상이다.
## 2.1. 이진화
명암 영상을 이진화하려면 임계값 T보다 큰 화소는 1, 그렇지 않은 화소는 0으로 바꾸면 된다.
$$
b(j,i)= \begin{cases} 1, f(j,i) \ge T \\ 0, f(j,i) < T \end{cases}
$$
히스토그램은 $0, 1, \cdots , L-1$의 명암 단계 각각에 대해 화소의 발생 빈도를 나타내는 1차원 배열이다.
![[Pasted image 20250716132047.png]]
아래 코드는 컬러 영상에서 R 채널을 명암 영상으로 간주해 히스토그램을 구한다.
``` python
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('soccer.jpg')
h = cv.calcHist([img], [2], None, [256], [0, 256])
plt.plot(h, color='r', linewidth=1)
plt.show()
```
![[Pasted image 20250716132120.png]]
## 2.2. 오츄 알고리즘
이 식은 모든 명암값에 대해 목적 함수 $J$를 계산하고 $J$가 최소인 명암값을 최적값 $\hat{t}$으로 정한다. 이렇게 결정한 $\hat{t}$을 임계값 $T$로 사용해 이진화한다.
$$
\hat{t}= \text{argmin}_{t\in \{0,1,2,\cdots ,L-1\}} J(t)
$$
**목적 함수**<sub>objective function</sub> $J(t)$는 $t$의 좋은 정도를 측정하는데 $J$가 작을수록 좋다. $t$로 이진화했을 때 0이 되는 화소의 분산과 1이 되는 화소의 가중치 합을 $J$로 사용했다. 가중치는 해당 화소의 개수이다. 분산이 적을수록 0인 화소 집합과 1인 화소 집합이 균일하기 때문에 좋은 이진 영상이 된다는 발상이다.
아래 식은 $J$를 정의한다. $n_0$과 $n_1$은 $t$로 이진화된 영상에서 0인 화소와 1인 화소의 개수고 $v_0$과 $v_1$인 0인 화소와 1인 화소의 분산이다.
$$
J(t)=n_0(t)v_0(t)+n_1(t)v_1(t)
$$
### 이진화
``` python
import cv2 as cv
import sys

img = cv.imread('soccer.jpg')

t, bin_img = cv.threshold(img[:, :, 2], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
print('오츄 알고리즘이 찾은 최적 임계값 =', t)

cv.imshow('R channel', img[:, :, 2])                # R 채널 영상
cv.imshow('R channel binarization', bin_img)        # R 채널 이진화 영상

cv.waitKey()
cv.destroyAllWindows()
```
![[Pasted image 20250716132810.png]]
### 최적화 문제
컴퓨터 비전은 주어진 문제를 최적화<sub>optimization</sub> 문제로 공식화해 푸는 경우가 많다. 오츄 알고리즘의 식은 이진화를 최적화 문제로 공식화한 예이다. 매개변수 $t$는 해 공간<sub>solution space</sub>을 구성한다. 해 공간이란 발생할 수 있는 모든 해의 집합이다. 이와 같이 모든 해를 다 검사하는 방법을 낱낱 탐색 알고리즘<sub>exhausive search algorithm</sub>이라 한다.
하지만 문제가 공간의 크기가 큰 경우가 대부분이기 때문에 낱낱 탐색 알고리즘은 시간이 너무 많이 걸린다. 따라서 보다 똑똑한 탐색 알고리즘을 사용한다. 예를 들어 물체의 외곽선을 찾는 스테이크 알고리즘은 명암 변화와 곡선의 매끄러운 정도가 최대가 되는 최적해를 탐욕 알고리즘<sub>greedy algorithm</sub>으로 찾는다. 신경망을 학습할 때는 **역전파**<sub>back-propagation</sub> 알고리즘을 사용해 최적해를 찾는다.
## 2.3. 연결 알고리즘
아래 그림은 화소의 연결성을 보여준다. 맨 왼쪽은 $(j, i)$에 위치한 화소의 8개 이웃 화소를 표시한다. 가운데는 상화좌우에 있는 4개 이웃만 연결된 것으로 간주하는 4-연결성이고 오른쪽은 대각선에 위치한 화소도 연결된 것으로 간주하는 8-연결성이다.
![[Pasted image 20250716141023.png]]
이진 영상에서 1의 값을 가진 연결된 화소의 집합을 **연결 요소**<sub>connected component</sub>라 한다.
## 2.4. 모폴로지
영상을 변환하는 과정에서 하나의 물체가 여러 영역으로 분리되거나 다른 물체가 한 영역으로 붙는 경우 등이 발생한다. 이런 부작용을 해결하기 위해 모폴로지 영산을 사용한다. **모폴로지**<sub>morphology</sub>는 구조 요소<sub>structuring element</sub>를 이용해 영역의 모양을 조작한다.
![[Pasted image 20250716141256.png]]
### 팽창, 침식, 열림, 닫힘
**팽창**<sub>dilation</sub>은 구조 요소의 중심을 1인 화소에 씌운 다음 구조 요소에 해당하는 모든 화소를 1로 바꾼다. 반면 **침식**<sub>erosion</sub>은 구조 요소의 중심을 1인 화소 $p$에 씌운 다음 구조 요소에 해당하는 모든 화소가 1인 경우에 $p$를 1로 유지하고 그렇지 않으면 0으로 바꾼다.
![[Pasted image 20250716141357.png]]
![[Pasted image 20250716141408.png]]
팽창은 영역을 키우고 침식은 영역을 작게 만든다. 침식을 수행한 영상에 팽창을 작용하면 대략 원래 크기를 유지한다. 침식한 결과에 팽창을 적용하는 연산을 **열림**<sub>opening</sub>, 팽창을 결과에 침식을 적용하는 연산을 **닫힘**<sub>closing</sub>이라 한다.
### 모폴로지 계산
``` python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('JohnHancocksSignature.png', cv.IMREAD_UNCHANGED)

t, bin_img = cv.threshold(img[:, :, 3], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
plt.imshow(bin_img, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

b = bin_img[bin_img.shape[0] // 2:bin_img.shape[0], 0:bin_img.shape[0] // 2 + 1]
plt.imshow(b, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

se = np.uint8([[0,0,1,0,0],
                [0,1,1,1,0],
                [1,1,1,1,1],
                [0,1,1,1,0],
                [0,0,1,0,0]])

b_dilation = cv.dilate(b, se, iterations=1)
plt.imshow(b_dilation, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

b_erosion = cv.erode(b, se, iterations=1)
plt.imshow(b_erosion, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

b_closing = cv.erode(cv.dilate(b, se, iterations=1), se, iterations=1)
plt.imshow(b_closing, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()
```
![[Pasted image 20250716141807.png]]
![[Pasted image 20250716141819.png]]
![[Pasted image 20250716141835.png]]
![[Pasted image 20250716141845.png]]
![[Pasted image 20250716141853.png]]
# 3. 점 연산
화소의 입장에서 바라본 연상 처리 연산이란 화소가 새로운 값을 받는 과정이다. 어디서 새로운 값을 받느냐에 따라 점 연산, 영역 연산, 기하 연산의 세 종류로 구분할 수 있다.
![[Pasted image 20250716142113.png]]
## 3.1. 명암 조절
아래 식을 이용해 영상을 밝게 또는 어둡게 조정할 수 있다. 맨 위 식은 원래 영상에 양수 a를 더해 밝게 만드는데, 화소가 가질 수 있는 최댓값 $L-1$을 넘지 않게 $min$을 취한다. 가운데 식은 원래 영상에 양수 $a$를 빼서 어둡게 만드는데 $max$를 취해 음수를 방지한다. 마지막 식은 $L-1$에서 원래 명암값을 빼서 반전시킨다. 이 식은 모두 선형<sub>linear</sub> 연산이다.
$$
f'(j,i)=\begin{cases} \min (f(j,i)+a, L-1) \\ \max (f(j,i)-a,0) \\ (L-1)-f(j,i) \end{cases}
$$
아래 식은 감마 보정<sub>gamma correction</sub>을 정의하며 비선형적인 사각 반응을 수학적으로 표현한다. $\gamma$는 1이면 원래 영상을 유지하고 1보다 작으면 밝아지고, 1보다 크면 어두워진다.
$$
f'\quad j,i = L-1\ \times \dot{f} \quad j,i ^\gamma
$$
``` python
import cv2 as cv
import numpy as np

img = cv.imread('soccer.jpg')
img = cv.resize(img, dsize=(0,0), fx=0.25, fy=0.25)

def gamma(f, gamma=1.0):
    f1 = f / 255.0
    return np.uint8(255 * (f1 ** gamma))

gc = np.hstack((gamma(img, 0.5), gamma(img, 0.75), gamma(img, 1.0), gamma(img, 2.0), gamma(img, 3.0)))
cv.imshow('gamma', gc)

cv.waitKey(0)
cv.destroyAllWindows()
```
![[Pasted image 20250716142528.png]]
## 3.2. 히스토그램 평활화
**히스토그램 평활화**<sub>histogram equalization</sub>는 히스토그램이 평평하게 되도록 영상을 조작해 명암 대비를 높이는 기법이다. 히스토그램 평활화는 모든 칸의 값을 더하면 1.0이 되는 정규화 히스토그램 $h$와 $i$번 칸은 $0\sim i$번 칸을 더한 값을 가진 누적 정규화 히스토그램 $h$를 가지고 식을 수행한다. $l$은 원래 명암값이고 $l’$은 평활화로 얻은 새로운 명암값이다.
$$
l'= \mathrm{round} (\ddot{h}(l) \times (L-1))
$$
``` python
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('mistyroad.png')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.show()

h = cv.calcHist([gray], [0], None, [256], [0, 256])
plt.plot(h, color='r', linewidth=1), plt.show()

equal = cv.equalizeHist(gray)
plt.imshow(equal, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.show()

h = cv.calcHist([equal], [0], None, [256], [0, 256])
plt.plot(h, color='r', linewidth=1), plt.show()
```
![[Pasted image 20250716142843.png]]
![[Pasted image 20250716142857.png]]
![[Pasted image 20250716142905.png]]
![[Pasted image 20250716142913.png]]
# 4. 영역 연산
## 4.1. 컨벌루션
**컨벌루션**<sub>convolution</sub>은 입력 영상 f의 각 화소에 필터를 적용해 곱의 합을 구하는 연산이다. 필터를 입력 영상의 $1, 2, 3, \cdots , 8$ 위치에 차례로 씌우고 곱의 합을 구해 출력 영상 $f’$에 쓴다. 곱의 합은 해당하는 화소끼리 곱한 다음 결과를 더한다.
$$
f'(x)=\sum^{(w-1)/2}_{i=-(w-1)/2} u(i)f(x+i)
$$
$$
f'(y,x)=\sum^{(h-1)/2}_{j=-(h-1)/2} \sum^{(w-1)/2}_{i=-(w-1)/2} u(j,i) f(y+j,x+i)
$$
![[Pasted image 20250716143019.png]]
## 4.2. 다양한 필터
컨볼루션 자체는 특정한 목적이 없는 일반 연산<sub>generic operation</sub>이다. 필터가 정해지면 목적이 결정된다.
![[Pasted image 20250716143051.png]]
![[Pasted image 20250716143101.png]]
밝은 영역에 어두운 작은 반점이 나타나는 경우 잡음일 수 있다. 그렇다면 스무딩 필터<sub>smoothing filter</sub>로 컨볼루션하면 잡음을 누그러뜨릴 수 있다. 왼쪽 필터 크기는 $3\times3$인데 모든 화소가 1/9이기 때문에 입력 영상의 해당 9개 화소의 평균을 취하는 셈이다. 따라서 어떤 점의 값이 주위에 비해 낮을 때 자신은 커지고 주위는 작아져서 잡음을 누그러뜨리는 효과를 발휘한다.
아래 식은 1차원과 2차원 가우시안 함수<sub>Gaussian function</sub>을 정의한다. 함수의 모양은 표준편차 $\sigma$에 따른다. $\sigma$가 크면 봉우리가 낮아지고 작으면 봉우리가 높아진다. 가우시안 함수는 중심에서 멀어지면 0은 아니지만 가깝기 때문에 디지털 필터를 만들 때는 필터 크기를 한정 짓는다.
$$
g(x)=\frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{x^2}{2\sigma^2}}
$$
$$
g(y,x)=\frac{1}{\sigma^2 2\pi}e^{-\frac{y^2 +x^2}{2\sigma^2}}
$$
단, 스무딩 필터는 잡음을 제거하는 효과가 있지만 물체의 경계를 흐릿하게 만드는 블러링<sub>blurring</sub>이라는 부작용이 있다. 이와 반대 작용을 하는 샤프닝<sub>sharpening</sub> 필터가 존재한다. 샤프닝 필터는 에지를 선명하게 해서 식별을 돕지만 잡음을 확대할 수 있다. 또한 물체에 돋을새김 느낌을 주는 엠보싱<sub>embossing</sub> 필터도 있다.
![[Pasted image 20250716143703.png]]
![[Pasted image 20250716143712.png]]
## 4.3. 데이터 형과 컨볼루션
### 데이터 형
`img`을 구성하는 화소의 데이터 형을 확인하면 `numpy.uint8`이다. `uint8`은 unsigned integer 8 bits의 약어로써 $0 \sim 255$ 범위를 표현할 수 있는 부호 없는 8 비트 데이터 형이다.
### 컨볼루션 적용
``` python
import cv2 as cv
import numpy as np

img = cv.imread('soccer.jpg')
img = cv.resize(img, dsize=(0,0), fx=0.4, fy=0.4)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.putText(gray, 'soccer', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv.imshow('Original', gray)

smooth = np.hstack((cv.GaussianBlur(gray, (5, 5), 0.0), cv.GaussianBlur(gray, (9, 9), 0.0), cv.GaussianBlur(gray, (15, 15), 0.0)))
cv.imshow('Smooth', smooth)

femboss = np.array([[-1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0]])

gray16 = np.int16(gray)
emboss = np.uint8(np.clip(cv.filter2D(gray16, -1, femboss)+128, 0, 255))
emboss_bad = np.uint8(cv.filter2D(gray16, -1, femboss) + 128)
emboss_worse = cv.filter2D(gray, -1, femboss)

cv.imshow('Emboss', emboss)
cv.imshow('Emboss Bad', emboss_bad)
cv.imshow('Emboss Worse', emboss_worse)

cv.waitKey(0)
cv.destroyAllWindows()
```
![[Pasted image 20250716143831.png]]
# 5. 기하 연산
## 5.1. 동차 좌표와 동차 행렬
동차 좌표<sub>homogeneous coordinate</sub>는 2차원 점의 위치$(x,y)$에 1을 추가해 아래 식처럼 3차원 벡터 $p$로 표현한다. 동차 좌표에서는 3개 요소에 같은 값을 곱하면 같은 좌표를 나타낸다.
$$
\bar{p}=(x,y,1)
$$
3가지 기하 변환은 이동, 회전, 크기다. 이들 연산을 $3\times 3$ 동차 행렬<sub>homogeneous matrix</sub>로 표현한다.
- 이동: $x$ 방향으로 $tx$, $y$ 방향으로 $ty$만큼 이동
$$
T(t_x,t_y)=\begin{pmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{pmatrix}
$$
- 회전: 원점을 중심으로 반시계 방향으로 $\theta$만큼 회전
$$
R(\theta)=\begin{pmatrix} \cos\theta & \sin\theta & 0 \\ -\sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{pmatrix}
$$
- 크기: $x$ 방향으로 $sx$, $y$ 방향으로 $sx$만큼 크기 조정 (1보다 크면 확대, 1보다 작으면 축소)
$$
S(s_x, s_y)=\begin{pmatrix} s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & 1 \end{pmatrix}
$$
변환 행렬은 아무리 여러 개를 곱해도 직선은 직선으로 유지되고 평행선은 평행을 유지한다. 이런 성질을 가진 변환을 그렇지 않은 변환과 구분하기 위해 **어파인 변환**<sub>affine transformation</sub>이라 한다. 투영<sub>projection</sub>은 멀리 있는 물체가 작게 보이기 위해 평행을 유지하지 못하는 변환으로 어파인 변환이 아니다.
## 5.2. 영상의 기하 변환
영상을 구성하는 점, 즉 화소에 동차 변환을 적용해 영상을 회전하거나 크기를 조정할 수 있다. 하지만 화소 위치를 정수로 지정하기 때문에 문제가 생긴다.
아래 그림은 동차 행렬 A로 원래 영상을 새로운 영상으로 변환하는 과정이다. 이때 실수 좌표를 정수로 반올림하면 값을 받지 못하는 경우가 발생하고 결과적으로 구멍이 뚫린 영상이 된다. 이와 같은 형상을 **에일리어싱**<sub>aliasing</sub>이라 하고 이 현상을 누그러뜨리는 방법을 **안티 에일리어싱**<sub>anti-aliasing</sub>이라 한다.
![[Pasted image 20250716145042.png]]## 5.3. 영상 보간
## 5.3. 영상 보간
영상에 기하 연산을 적용할 때 후방 변환을 사용하면 구멍이 생기는 현상을 방지할 수 있다. 하지만 여전히 실수 좌표를 정수로 변환하는 과정이 필요하다. 이때 반올림을 사용해 가장 가까운 화소에 배정하는 방법을 **최근접 이웃**<sub>nearest neighbor</sub> 방법이라고 한다. 다만 엘리어싱이 심하기 때문에 보간을 사용하면 개선된다.
$$
f(j'i')=\alpha\beta f(j,i)+(1-\alpha)\beta f(j,i+1)+\alpha(1-\beta)f(j+1,i)\\+(1-\alpha)(1-\beta)f(j+1,i+1)
$$
![[Pasted image 20250716145626.png]]
이 보간은 $x$와 $y$의 두 방향에 걸쳐 계산하므로 양선형 보간법<sub>bilinear interpolation method</sub>이라 한다.
### 보간을 이용한 영상 변환
``` python
import cv2 as cv

img = cv.imread('rose.jpg')
patch = img[250:350, 170:270, :]

img = cv.rectangle(img, (170, 250), (270, 350), (255, 0, 0), 3)
patch1 = cv.resize(patch, dsize=(0,0), fx=5, fy=5, interpolation=cv.INTER_NEAREST)
patch2 = cv.resize(patch, dsize=(0,0), fx=5, fy=5, interpolation=cv.INTER_LINEAR)
patch3 = cv.resize(patch, dsize=(0,0), fx=5, fy=5, interpolation=cv.INTER_CUBIC)

cv.imshow('Original', img)
cv.imshow('Resize Nearest', patch1)
cv.imshow('Resize Bilinear', patch2)
cv.imshow('Resize Bicubic', patch3)

cv.waitKey(0)
cv.destroyAllWindows()
```
![[Pasted image 20250716203901.png]]
# 6. OpenCV의 시간 효율
``` python
import cv2 as cv
import numpy as np
import time

def my_cvtGray1(bgr_img):
    g = np.zeros([bgr_img.shape[0], bgr_img.shape[1]])
    for r in range(bgr_img.shape[0]):
        for c in range(bgr_img.shape[1]):
            g[r, c] = 0.114 * bgr_img[r, c, 0] + 0.587 * bgr_img[r, c, 1] + 0.299 * bgr_img[r, c, 2]
    return g.astype(np.uint8)

def my_cvtGray2(bgr_img):
    g = np.zeros([bgr_img.shape[0], bgr_img.shape[1]])
    g = 0.114 * bgr_img[:, :, 0] + 0.587 * bgr_img[:, :, 1] + 0.299 * bgr_img[:, :, 2]
    return np.uint8(g)

img = cv.imread('rose.jpg')

start = time.time()
my_cvtGray1(img)
print('My time1:', time.time() - start)

start = time.time()
my_cvtGray2(img)
print('My time2:', time.time() - start)

start = time.time()
cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print('OpenCV time:', time.time() - start)
```
```
My time1: 0.938173770904541
My time2: 0.005000591278076172
OpenCV time: 0.010982751846313477
```
