---
date: 2025-07-10
tags:
  - ai
  - computer_vision
  - deep_learning
---
# 1. 3차원 기하와 캘리브레이션
## 1.1. 3차원 기하 변환
![[Pasted image 20250710122321.png]]
3차원에서의 이동과 회전을 위한 병렬 행렬은 다음과 같다.
$$
T(t_x,t_y,t_z)=\begin{pmatrix} 1 & 0 & 0 & t_x \\ 0 & 1 & 0 & t_y \\ 0 & 0 & 1 & t_z \\ 0 & 0 & 0 & 1 \end{pmatrix},\ R(\mathbf{n},\theta)=\begin{pmatrix} r_1 & r_2 & r_3 & 0 \\ r_4 & r_5 & r_6 & 0 \\ r_7 & r_8 & r_9 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}
$$2차원과 마찬가지로 여러 단계의 변환을 하려면 변환 행렬을 미리 곱해 하나의 변환 행렬을 만들고 3차원 점에 곱하면 된다. 
위 식은 컴퓨터 비전보다 3차원 물체를 모델링하고 렌더링하는 컴퓨터 그래픽스에서 더 많이 사용한다. 컴퓨터 비전에서는 3차원 공간 상에서 물체를 기하 변환하는 일보다 어떤 좌표계 상의 점을 다른 좌표계의 점으로 변환하는 일에 관심이 더 많다. 이 일을 하려면 캘리브레이션<sub>calibration</sub>이 필요하다.
## 1.2. 영상 생성 기하
3차원 세계가 2차원 영상 평면에 투영되는 과정, 즉 영상 생성<sub>image formation</sub>을 위한 기하를 단순하게 표현한 핀홀 카메라 모델이다.
![[Pasted image 20250710122339.png]]
### 세계 좌표계와 카메라 좌표계
위 그림에는 3개의 좌표계가 있다. 파란색으로 표시한 3차원 세계 좌표계<sub>world coordinate</sub>, 빨간색으로 표시한 3차원 카메라 좌표계<sub>camera coordinate system</sub>, 카메라에 붙어 있는, 주황색으로 표시된 영상 평면을 나타내는 2차원 영상 좌표계다. 세계 좌표계를 방의 구석점이라고 볼 때 방 바닥에 $x$축과 $y$축이 있고 위쪽 방향이 $z$축이다. 3차원 물체의 점 $\mathbf{p}$는 세계 좌표계를 기준으로 $(x_w,y_w,z_w)$, 카메라 좌표계를 기준으로, $(x_c,y_c,z_c)$로 표기한다. 영상 좌표계에 투영된 2차원 점은 $(u,v)$로 표기한다. 
세계 좌표계의 $(x_w,y_w,z_w)$를 카메라 좌표계의 $(x_c,y_c,z_c)$로 변환하는 방법은 다음과 같다. $(\mathbf{R}|\mathbf{t})$ 행렬은 카메라가 외부 세계와 상호작용하는 기하 관계를 표현하기 때문에 외부 행렬<sub>extrinsic matrix</sub>이라 부른다.
$$
\begin{pmatrix} x_c \\ y_c \\ z_c \end{pmatrix} = \begin{pmatrix} r_1 & r_2 & r_3 & t_x \\ r_4 & r_5 & r_6 & t_y \\ r_7 & r_8 & r_9 & r_z \end{pmatrix} \begin{pmatrix} x_w \\ y_w \\ z_w \\ 1 \end{pmatrix}= (\mathbf{R}|\mathbf{t}) \begin{pmatrix} x_w \\ y_w \\ z_w \\ 1\end{pmatrix}
$$
### 영상 좌표계
3차원 점 $(x_c,y_c,z_c)$는 영상 공간의 $(u,v)$ 점으로 투영된다. $(u,v)$는 $u-v$ 좌표계를 기준으로 한 좌표이고 $(x,y)$는 $x-y$ 좌표계를 기준으로 한 좌표다. $f$는 카메라 좌표계에서 영상 평면까지 거리로 초점 거리<sub>focal length</sub>라 부른다. 삼각비에 따라 아래 식이 성립된다.
$$
x=\frac{fx_c}{z_c},\ y=\frac{fy_c}{z_c}
$$
위 식을 행렬 식으로 쓰면 아래 식이 된다.
$$
\begin{pmatrix} x' \\ y' \\ z' \end{pmatrix}=\begin{pmatrix} f & 0 & 0 \\ 0 & f & 0 \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} x_c \\ y_c \\ z_c \end{pmatrix}
$$
카메라에 따라 둘이 다를 수 있기 때문에 일반성 확보를 위해 두 방향의 초점 거리를 $f_x$와 $f_y$로 구분한다. $u-v$ 좌표계의 원점은 $(c_x, c_y)$이며 $x$와 $y$축이 기울 수 있어 $\gamma$를 반영한다. 아래 식은 이런 사항을 반영한다. $\mathbf{K}$는 카메라의 내부 동작을 표현한다는 뜻에서 내부 행렬<sub>intrinsic matrix</sub>이라 부른다.
$$
\begin{pmatrix} u' \\ v' \\ w' \end{pmatrix}=\begin{pmatrix} f_x & \gamma & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} x_c \\ y_c \\ z_c \end{pmatrix} = \mathbf{K} \begin{pmatrix} x_c \\ y_c \\ z_c \end{pmatrix}
$$
아래 식은 위 식의 동차 좌표를 보통 좌표로 변환한다.
$$
u=\frac{u'}{w'},\ v=\frac{v'}{w'}
$$
아래 식은 지금까지 한 일 전체를 정리한다.
$$
\begin{pmatrix} u' \\ v' \\ w' \end{pmatrix}=\begin{pmatrix} f_x & \gamma & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} x_c \\ y_c \\ z_c \end{pmatrix} =\begin{pmatrix} f_x & \gamma & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} r_1 & r_2 & r_3 & t_x \\ r_4 & r_5 & r_6 & t_y \\ r_7 & r_8 & r_9 & r_z \end{pmatrix} \begin{pmatrix} x_w \\ y_w \\ z_w \\ 1 \end{pmatrix}= \mathbf{K}(\mathbf{R}|\mathbf{t}) \begin{pmatrix} x_w \\ y_w \\ z_w \\ 1\end{pmatrix} = \mathbf{P} \begin{pmatrix} x_w \\ y_w \\ z_w \\ 1\end{pmatrix}
$$
## 1.3. 카메라 캘리브레이션
카메라는 3차원 세계를 2차원으로 투영해 영상을 획득한다. 컴퓨터 비전으로 영상을 인식해 정보를 알아내면 외부 세계와 상호작용해 유용한 일을 수행해야 한다. 중요한 일은 카메라가 외부 세계와 어떤 기하학적 관계를 가지는지 알아내는 것이다. 이 기하학적 관계의 추정은 캘리브레이션을 통해 이루어진다. 다시 말해 캘리브레이션은 위 식에서 외부 행렬 $(\mathbf{R}|\mathbf{t})$와 내부 행렬 $\mathbf{K}$를 알아내는 일이다. 내부 행렬을 알아내는 일을 내부 카메라 캘리브레이션<sub>intrinsic camera calibration</sub>, 외부 행렬을 알아내는 일을 외부 카메라 캘리브레이션<sub>extrinsic camera calibration</sub>이라 한다.
### Zhang 방법
영상을 획득하는 상황을 제어할 수 있다면 아래 그림과 같은 격자 패턴을 사용하는 방법이 가장 정확하고 편하다. 
1. 격자 패턴을 프린트해 평평한 곳에 붙인다. 격자를 구성하는 칸의 길이를 알아야 한다.
2. 카메라 또는 칠판을 이동하면서 여러 장의 영상을 획득한다.
3. `findChessboardCorners` 함수를 이용해 영상에서 격자의 교차점을 검출하여 1단계에서 수집한 3차원 점과의 2차원 대응점을 수집한다.
4. `calibrateCamera` 함수를 이용해 내부 행렬의 5개 매개변수와 외부 행렬의 6개 매개변수를 구한다.
### 렌즈 왜곡 바로잡기
![[Pasted image 20250710130645.png]]
## 1.4. 손눈 캘리브레이션
아래 그림에서 내부 행렬 $\mathbf{K}$를 알아낼 때 로봇 눈에 해당하는 카메라가 로봇 손과 협동하는 데 필요한 일이라는 뜻에서 손눈 캘리브레이션<sub>hand-eye calibration</sub>이라 부른다. 
![[Pasted image 20250710131311.png]]
### 영상 좌표를 로봇 좌표로 변환
위 그림에서 손눈 캘리브레이션을 완성하면 위 식을 이용해 로봇 좌표계의 3차원 점 $(x_w,y_w,z_w)$를 영상 공간의 좌표 $(u,v)$로 변환할 수 있다. 그런데 로봇이 작업할 때는 컴퓨터 비전이 알아낸 영상 공간의 좌표 $(u,v)$에 해당하는 3차원 점 $(x_w,y_w,z_w)$로 로봇 손을 이동해서 물체를 잡아야 하기 때문에 역방향으로 식을 적용해야 한다.
![[Pasted image 20250710131951.png]]
# 2. 깊이 추정
## 2.1. 스테레오
### 깊이 추정을 위한 기하 공식
삼각형의 닮은비를 적용하면 쉽게 아래 식을 유도할 수 있다. 영상 평면의 좌표인 $x_i$와 카메라의 초점거리 $f$는 알 수 있지만 3차원 점의 좌표인 $x_w$는 알 수 없다.
$$
z_w=\frac{x_wf}{x_i}
$$
![[Pasted image 20250710132008.png]]
카메라 두 대 사용하면 거리를 확정할 수 있다. 이런 방식으로 거리를 알아내는 기술을 스테레오 비전<sub>stereo vision</sub>이라 한다.
$$
z_w=\frac{x_wf}{x_i},\ z_w=\frac{(b+x_w)f}{x_{right}}
$$
위 식을 정리하면 아래 식을 얻는다. $p$는 변위<sub>disparity</sub>라 부른다.
$$
z_w=\frac{bf}{x_{right}-x_{left}}=\frac{bf}{p}
$$
### 대응점 찾기와 에피폴리 기하
첫 번째와 두 번째 영상에서 서로 대응되는 두 점을 표시하는데, $x_{left}$와 $x_{right}$는 대응점의 $x$ 좌표다.
![[Pasted image 20250710132759.png]]
스테레오에서는 틀린 대응점에 위 식을 적용하면 엉뚱한 깊잇값이 되어버린다. 게다가 SIFT가 검출된 몇몇 화소에 대해서만 깊이를 계산하면 희소 깊이 영상<sub>sparse depth image</sub>
이 된다. 스테레오 카메라는 위 그림의 오른쪽과 같이 모든 화소가 값을 가지는 밀집 깊이 영상<sub>dense depth map</sub>을 생성해야 한다.
스테레오 비전에서는 대응점을 찾을 때 에피폴라 기하<sub>epipolar geometry</sub>라는 매우 유용한 조건이 있다. 왼쪽 영상의 점 $(x_{left}, y_{left})$에 대응하는 오른쪽 영상의 점 $(x_{right}, y_{right})$는 반드시 에피폴라 선분이라 부르는 초록색 선분 위에 나타난다. 보통 $x$축과 수직하지 않은 경우가 많기에, 임의의 방향을 가진 에피포라 선분을 따라 대응점을 탐색한다.
$$
\begin{align}y_{right}=y_{left}\\ x_{right}=x_{left}+p \end{align} \biggl\}
$$
### 밀집 깊이 영상 획득
아래 알고리즘은 밀집 깊이 영상을 얻는 알고리즘이다. 대응점을 찾으려면 에피폴라 선분에 있는 점에 대해 아래 식의 제곱차 합<sub>SSD; Sum of Squared Difference</sub>을 계산하고 최소가 되는 점을 찾으면 된다. 
$$
\text{SSD}(\mathbf{p},\mathbf{q})=\sum_{p_n\in patch(\mathbf{p}),\mathbf{q}_n \in patch(\mathbf{q})}||I(\mathbf{p}_n)-I(\mathbf{q}_n)||^2_2
$$
대응점 $I_{right}$을 찾을 때, 왼쪽 오른쪽 영상에서 어떤 점 $\mathbf{p},\mathbf{q}$에 대해서 그 점을 중심으로 하는 패치 $\mathbf{p}_n, \mathbf{q}_n$에 대해서 SSD를 시행해 가장 값이 가장 작은 $q$를 대응점으로 채택하는 방식이 있다. 
> [!info] 스테레오 비전에서 밀집 깊이 영상 추정
> 입력: 왼쪽 영상 $I_{left}$와 오른쪽 영상 $I_{right}$
> 출력: 밀집 깊이 영상 $Z$
> 1. for 모든 화소 $(x,y)$에 대해
> 	1. $x_{left}=x,\ y_{left}=y$
> 	2. $I_{left}$의 $(x_{left}, y_{left})$에 대응하는 $I_{right}$의 $(x_{right},y_{right})$를 에피폴라 선분에서 찾는다.
> 	3. 깊이 $z_w$를 계산해 $Z(x,y)$에 기록한다.

원래 영상에서 잘라낸 패치에서 유사도를 계산하는 대신에 컨볼루션 신경망이 추출한 특징에서 유사도를 계산하는 전략을 사용할 수 있다. MC-CNN은 샴 신경망<sub>Siamese network</sub>을 사용해 특징을 추출하고 완전연결층 $1 \times 1$ 컨볼루션층을 사용해 $[0,1]$ 범위의 유사도를 출력한다.
![[Pasted image 20250710135015.png]]
### 단안 깊이 추정
단안 깊이 추정<sub>monocular depth estimation</sub>은 한 장의 컬러 또는 명암 영상으로부터 깊이 영상을 추정하는 문제로 활발히 연구되고 있다.
![[Pasted image 20250710135152.png]]
## 2.2. 능동 센서
거리 측정에 유리한 신호를 장면에 발사하고 물체가 반사하는 신호를 되받아서 거리를 계산한다. 무엇을 쏘는지에 따라 구조 광<sub>structured light</sub>과 비행 거리<sub>TOF; Time Of Flight</sub> 방식으로 구별할 수 있다.
![[Pasted image 20250710135334.png]]
TOF 기법은 빛 또는 소리 신호를 발사하고 물체에서 반사되어 돌아오는 시간을 측정해 거리를 계산한다.
$$
z=\frac{ct}{2}
$$
TOF를 실제 구현할 때는 근적외선<sub>NIR; Near InfRared</sub>, 초음파<sub>ultrasound</sub>, 레이저<sub>laser</sub> 빛을 주로 사용하는데, 사용하는 빛의 종류와 쏘는 방식, 반사된 빛을 받는 방식 등에 따라 다양한 기법이 있다.
## 2.3. 상용 깊이 카메라
![[Pasted image 20250710135620.png]]
RGB-D 영상에서 깊이에 해당하는 D 채널은 잡음이 심한 편이다. Zhang은 RGB 영상에서 물체 표면의 법선 벡터<sub>normal vector</sub>와 물체 경계를 검출한 다음 이들 정보를 이용해 깊이 영상의 빈 곳을 채우는 알고리즘을 제안했다.
![[Pasted image 20250710135820.png]]
# 3. RGB-D 영상 인식
D는 두 가지 측면에서 컴퓨터 비전에 쓸모가 있는데, 첫 번째는 RGB와 D를 융합해 인식 성능을 높이는 것이고 두 번째는 로봇 주행에 필수적인 SLAM과 같은 새로운 응용을 창출하는 것이다.
### RGB와 D를 융합하여 인식 성능 향상
RGB-D에서 RGB는 물체 외관<sub>appearance</sub>에 대한 정보를 가졌고 D는 물체 형상<sub>shape</sub>에 대한 정보를 가졌기 때문에 상호 약점을 보완할 수 있다. 
RGB-D 영상을 의미 분할하기 위해 쉬운 접근 방법은 RGB 채널과 D 채널을 별도의 신경망에 넣어 특징 맵을 추출한 다음 적절한 순간에 융합을 하는 것이다. D채널은 불완전한 특성이 있기 때문에 신중하게 결합해야 좋은 성능을 얻을 수 있다. 
![[Pasted image 20250710140352.png]]
### SLAM
자신의 위치를 부분 지도 속에서 인식하며 동시에 지도를 완성해나가는 일을 SLAM<sub>Simultaneous Localization And Mapping</sub>이라고 한다. SLAM에서 mapping은 3차원 모델이고 localization은 위치와 방향, 즉 자세다.
![[Pasted image 20250710140618.png]]
# 4. 점 구름 인식
## 4.1. 데이터의 이해
점 구름 영상은 아래 식처럼 3차원 좌표로 점을 표현한다. 점은 번호가 매겨져 있지만 순서를 바꿔도 같은 영상이다. 따라서 점 구름은 순열 불변<sub>permutation-invariant</sub>이라고 말한다.
$$
I_{point cloud}=\{(x_i,y_i,z_i,\alpha_i)|1 \le i \le n\}
$$
``` python
import os
import trimesh
import tensorflow as tf
import matplotlib.pyplot as plt

classes = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
           'monitor', 'night_stand', 'sofa', 'table', 'toilet']

path = 'http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'
data_dir = tf.keras.utils.get_file('modelnet.zip', path, extract=True)
data_dir = os.path.join(os.path.dirname(data_dir), 'ModelNet10')

fig = plt.figure(figsize=(50, 5))
for i in range(len(classes)):
    mesh = trimesh.load(os.path.join(data_dir, classes[i] + '/train/' + classes[i] + '_0001.off'))
    points = mesh.sample(4096)
    ax = fig.add_subplot(1, 10, i + 1, projection='3d')
    ax.set_titile(classes[i], fontsize=30)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='g')
```
## 4.2. 인식 모델
### PointNet
PointNet의 구조는 점 구름 데이터의 세 가지 성질을 반영했다. 첫 번째 성질은 순열 불변이다. 순열 불변은 max pool 표시된 최대 풀링으로 달성한다. 두 번째는 집합에서 이웃 점을 찾아 물체의 지역 구조<sub>local structure</sub>를 파악할 수 있어야 한다. 세 번째는 물체에 회전이나 크기의 기하 변환이 일어나도 같은 출력을 내야 한다.
![[Pasted image 20250710160234.png]]
### PointNet: 분류 프로그래밍
``` python 
import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

classes = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
           'monitor', 'night_stand', 'sofa', 'table', 'toilet']

path = 'http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'
data_dir = tf.keras.utils.get_file('modelnet.zip', path, extract=True)
data_dir = os.path.join(os.path.dirname(data_dir), 'ModelNet10')

def parse_dataset(num_points=4096):
    train_points, train_labels = [], []
    test_points, test_labels = [], []

    for i in range(len(classes)):
        folder = os.path.join(data_dir, classes[i])
        print('데이터 읽기: 부류 {}'.format(os.path.basename(folder)))
        train_files = glob.glob(os.path.join(folder, 'train/*'))
        test_files = glob.glob(os.path.join(folder, 'test/*'))

        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    return (np.array(train_points), np.array(test_points), np.array(train_labels), np.array(test_labels))

NUM_POINTS = 2048
NUM_CLASSES = 10
batch_siz = 32

x_train, x_test, y_train, y_test = parse_dataset(num_points=NUM_POINTS)

def conv_bn(x, filters):
    x = keras.layers.Conv1D(filters, kernel_size=1, padding='valid')(x)
    x = keras.layers.BatchNormalization(momentum=0.0)(x)
    return keras.layers.ReLU()(x)

def dense_bn(x, filters):
    x = keras.layers.Dense(filters)(x)
    x = keras.layers.BatchNormalization(momentum=0.0)(x)
    return keras.layers.ReLU()(x)

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)
    
    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2,2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))
    
def tnet(inputs, num_features):
    bias = keras.initializers.Constant(value=np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features=num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = keras.layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = keras.layers.Dense(num_features * num_features, kernel_initializer='zeros', bias_initializer=bias, activity_regularizer=reg)(x)

    feat_T = keras.layers.Reshape((num_features, num_features))(x)
    return keras.layers.Dot(axes=(2, 1))([inputs, feat_T])

inputs = keras.Input(shape=(NUM_POINTS, 3))
x = tnet(inputs, 3)
x = conv_bn(x, 32)
x = conv_bn(x, 32)
x = tnet(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 128)
x = keras.layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 256)
x = keras.layers.Dropout(0.3)(x)
x = dense_bn(x, 128)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='PointNet')

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['sparse_categorical_accuracy'])
model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

chosen = np.random.randint(0, len(x_test), 8)
points = x_test[chosen]
labels = y_test[chosen]

fig = plt.figure(figsize=(15, 4))
for i in range(8):
    ax = fig.add_subplot(2, 4, i+1, projection='3d')
    ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2], s=1, c='g')
    ax.set_title('pred: {}, true: {}'.format(
        classes[np.argmax(model.predict(points[i:i+1]))],
        classes[labels[i]]), fontsize=15)
    ax.set_axis_off()
```