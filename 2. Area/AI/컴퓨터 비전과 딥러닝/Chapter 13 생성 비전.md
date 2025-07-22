---
date: 2025-07-13
tags:
  - ai
  - computer_vision
  - deep_learning
---
# 1. 생성 모델 기초
## 1.1. 생성 모델이란?
생성 모델을 가장 단순하게 표현하면 아래 식이다. $\mathbf{x}$가 발생할 확률 분포<sub>probability distribution</sub>, $p(\mathbf{x})$를 알면 새로운 샘플을 생성할 수 있다. 확률 이론에서는 $\mathbf{x}$를 랜덤 벡터<sub>random vector</sub>라 부르는데, 컴퓨터 비전에서는 $\mathbf{x}$가 영상 또는 영상에서 추출한 특징 벡터에 해당한다. 
$$
생성\ 모델:\ p(\mathbf{x})
$$
### 장난감 세상의 생성 모델
#### 데이터에 기반한 생성 모델
6개 면의 면적이 같지 않은 찌그러진 주사위가 있다고 가정한다. 주사위를 10번 던져 다음과 같은 데이터를 얻었다고 가정한다.
$$
X=\{5,3,5,5,2,4,1,6,3,1\}
$$
데이터로부터 확률 분포 $p(x)$를 추정하면 다음과 같다. 이제 $p(x)$를 가지고 $x$를 생성하여 주사위 게임을 만들 수 있다.
$$
p(x)=
\begin{cases}
p(x=1)=0.2 \\ p(x=2)=0.1 \\ p(x=3)=0.2 \\ p(x=4)=0.1 \\ p(x=5)=0.3 \\ p(x=6)=0.1
\end{cases}
$$
#### 특징 벡터가 2차원인 경우의 생성 모델
특징 벡터가 2차원이고 각 특징은 0 또는 1 값을 가진다고 가정하자. 샘플링 실험을 통해 아래 데이터를 얻었다고 한다.
$$
X=\{(0,0),(1,1),(0,0),(1,1),(1,1),(1,0),(0,1),(1,0),(1,0),(1,1)\}
$$
데이터로부터 확률 분포 $p(\mathbf{x})$를 추정하면 다음과 같다.
$$
p(\mathbf{x})=
\begin{cases} 
p((0,0))=0.2 \\ p((0,1))=0.1 \\ p((1,0))=0.3 \\ p((1,1))=0.4
\end{cases}
$$
데이터를 만들어내는 진짜 확률 분포를 $p_{data}$, 주어진 데이터로부터 추정한 확률 분포를 $p_{model}$이라 표기한다. 보통 $p_{data}$는알 수 없고, 단지 데이터에서 근사 추정한 $p_{model}$만 알 수 있다. 확률 이론에 따르면 데이터의 크기가 커지면 $p_{model}$은 $p_{data}$에 점점 가까워진다. 이런 경우를 다룸 가능<sub>tractable</sub>이라 말한다.
생성 모델에 대한 연구는 다룸 가능을 벗어나지 않으면서 가능한 $p_{model}$이 $p_{data}$에 가깝도록 모델링하는 기법을 찾는 일이다. 이 일을 수학으로 표현하면 아래 식이 된다.
$$
p_{model}(\mathbf{x}) \cong p_{data}(\mathbf{x})
$$
## 1.2. 가우시안 혼합 모델
### 단순한 상황: 특징 벡터가 2차원 실수 공간
변수가 2개인 단순한 상황에서 시작한다. 다양한 사람은 모집해 키와 몸무게를 잰 뒤 데이터셋을 얻었다고 가정한다. 데이터셋은 가우시안 분포(정규 분포)를 한다고 가정하고 아래 식의 확률 분포를 추정한다.
$$
p(\mathbf{x})=N(\mu, \sum)
$$
``` python
import numpy as np

X = np.array([[169,70],[172,68],[175,78],[163,58],[180,80],[159,76],[158,52],[173,69],[180,75],[155,50],[187,90],[170,66]])

m = np.mean(X, axis=0)
cv=np.cov(X, rowvar=False)

gen = np.random.multivariate_normal(m,cv,5)

print(gen)
```
```
[[168.91089304  77.92444031]
 [192.81347248  89.98557644]
 [179.91189326  70.78925061]
 [177.16325288  75.45314173]
 [163.9673125   65.79395881]]
```

``` python
import numpy as np
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
X = x_train[np.isin(y_train, [0])]
X = X.reshape((X.shape[0], 28*28))

m = np.mean(X, axis=0)
cv = np.cov(X, rowvar=False)

gen = np.random.multivariate_normal(m,cv,5)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(gen[i].reshape((28,28)), cmap='gray'); plt.xticks([]); plt.yticks([])

plt.show()
```
![[Pasted image 20250713161249.png]]
### 가우시안 혼합 모델(GMM)
아래 그림은 가상의 데이터 분포를 예시한다. 3개 모드가 있는데 아래 그림은 모드를 찾아 각각을 가우시안으로 표현하는 가우시안 혼합 모델<sub>GMM; Gaussian Mixture Model</sub>을 개념적으로 보여준다. 아래 식은 GMM 모델을 표현한다.
$$
p(\mathbf{x})=\sum_{i=1,k}\pi_iN(\mu_i,\sum_i)
$$
![[Pasted image 20250713162001.png]]
``` python
import numpy as np
import sklearn.mixture
from tensorflow import keras
import sklearn

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
X = x_train[np.isin(y_train, [0])]
X = X.reshape((X.shape[0], 28*28))

k = 8

gm = sklearn.mixture.GaussianMixture(n_components=k).fit(X)

gen = gm.sample(n_samples=10)

import matplotlib.pyplot as plt

plt.figure(figsize=(20,4))
for i in range(k):
    plt.subplot(1,10,i+1)
    plt.imshow(gm.means_[i].reshape((28,28)),cmap='gray'); plt.xticks([]); plt.yticks([])
plt.show()

plt.figure(figsize=(20,4))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(gen[0][i].reshape((28,28)),cmap='gray'); plt.xticks([]); plt.yticks([])
plt.show()
```
![[Pasted image 20250713163823.png]]
![[Pasted image 20250713163927.png]]
### 가우시안 혼합 모델의 한계
가우시안과 달리 평균을 중심으로 평균을 중심으로 대칭 모양을 형성하지 않는다. 가우시안 혼합 모델의 가정은 실제 데이터를 너무 단순화한다. 
가우시안 혼합 모델은 데이터가 얼추 가우시안 분포를 한다는 가정을 통해 다룸 가능성을 확보하지만, 가우시안에 기반한 $p_{model}$이 진짜 확률 분포 $p_{data}$와 차이가 커서 생성 모델로서 한계가 있다.
## 1.3. 최대 우도
생성 모델은 $n$개의 샘플을 가진 데이터셋 $X=\{\mathbf{x}^1,\mathbf{x}^2, \cdots, \mathbf{x}^n\}$으로 학습한다. 생성 모델의 학습 알고리즘은 데이터 $X$를 발생할 가능성이 가장 높은 매개변수 $\theta$를 추정하는 방식으로 동작한다. 이를 식으로 쓰면 아래 식이다.
$$
\hat{\theta}=\text{argmax}_\theta \ p_{\theta}(X)
$$
$p_\theta (X)$를 우도<sub>likelihood</sub>라 부르고 위 식을 푸는 학습 알고리즘을 최대 우도법<sub>maximum likelihood method</sub>이라 부른다.
![[Pasted image 20250713181436.png]]
위 식에서 $X$를 구성하는 $n$개의 샘플은 독립적으로 샘플링되었으므로 우도 $p_\theta (X)$를 아래 식으로 쓸 수 있다.
$$
p_\theta (X)= \prod^n_{i=1}p_\theta (\mathbf{x}^i)=p_\theta(\mathbf{x}^1)p_\theta(\mathbf{x}^2)\cdots p_\theta(\mathbf{x}^n)
$$
log 함수는 곱셈을 덧셈으로 바꾸기 때문에 보통 log 함수를 적용한 아래 식의 로그 우도<sub>log likelihood</sub>를 대신 사용한다.
$$
\hat{\theta}=\text{argmax}_\theta \sum^n_{i=1} \log p_\theta (\mathbf{x}^i)
$$
## 1.4. 고전 생성 모델
### 은닉 마르코프 모델
은닉 마르코프 모델<sub>HMM; Hidden Markov Model</sub>은 딥러닝 이전에 있던 생성 모델이다. 
![[Pasted image 20250713184309.png]]
### 제한 볼츠만 머신
신경망은 근사 추정에 뛰어나기 때문에 자연스럽게 신경망과 최대 우도를 결합하는 아이디어가 탄생했다. 제한 볼츠만 머신<sub>RBM; Restricted Boltzmann Machine</sub>은 초기 시도다.
![[Pasted image 20250713184500.png]]
# 2. 오토인코더를 이용한 생성 모델
``` python
import numpy as np
import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = np.reshape(x_train, (len(x_train), 28,28,1))
x_test = np.reshape(x_test, (len(x_test), 28,28,1))

z_dim=32

encoder_input=keras.layers.Input(shape=(28,28,1))
x=keras.layers.Conv2D(32,(3,3),activation='relu',padding='same',strides=(1,1))(encoder_input)
x=keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',strides=(2,2))(x)
x=keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',strides=(2,2))(x)
x=keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',strides=(1,1))(x)
x=keras.layers.Flatten()(x)
encoder_output=keras.layers.Dense(z_dim)(x)
model_encoder=keras.models.Model(encoder_input, encoder_output)

decoder_input=keras.layers.Input(shape=(z_dim,))
x=keras.layers.Dense(3136)(decoder_input)
x=keras.layers.Reshape((7,7,64))(x)
x=keras.layers.Conv2DTranspose(64,(3,3),activation='relu',padding='same',strides=(1,1))(x)
x=keras.layers.Conv2DTranspose(64,(3,3),activation='relu',padding='same',strides=(2,2))(x)
x=keras.layers.Conv2DTranspose(32,(3,3),activation='relu',padding='same',strides=(2,2))(x)
x=keras.layers.Conv2DTranspose(1,(3,3),activation='relu',padding='same',strides=(1,1))(x)
decoder_output=x
model_decoder=keras.models.Model(decoder_input, decoder_output)

model_input=encoder_input
model_output=model_decoder(encoder_output)
model=keras.models.Model(model_input,model_output)

model.compile(optimizer='Adam',loss='mse')
model.fit(x_train, x_train, epochs=10, batch_size=128, shuffle=True, validation_data=(x_test,x_test))

import matplotlib.pyplot as plt

i=np.random.randint(x_test.shape[0])
j=np.random.randint(x_test.shape[0])
x=np.array((x_test[i], x_test[j]))
z=model_encoder.predict(x)

zz=np.zeros((11,z_dim))
alpha=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
for i in range(11):
    zz[i]=(1.0-alpha[i])*z[0]+alpha[i]*z[1]

gen=model_decoder.predict(zz)

plt.figure(figsize=(20,4))
for i in range(11):
    plt.subplot(1,11,i+1)
    plt.imshow(gen[i].reshape(28,28),cmap='gray'); plt.xticks([]); plt.yticks([])
    plt.title(str(alpha[i]))
plt.show()
```
![[Pasted image 20250714091004.png]]
## 2.2. 변이                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
뛰어난 언어-비전 생성 모델은 변이 오토인코더 또는 확산 모델을 이용한다. 변이 오토인코더와 확산 모델은 복잡한 확률 분포를 추정해야 하는데, 이 과정을 위해 변이 추론<sub>variational inference</sub>을 활용한다. 
확률 분포 $p(\mathbf{x})$를 알면 완벽한 생성 모델을 가진 셈이다. 하지만 우리가 가진 것은 $p(\mathbf{x})$가 아니라 $p(\mathbf{x})$에서 샘플링한 데이터셋 $X=\{\mathbf{x}^1,\mathbf{x}^2,\cdots,\mathbf{x}^n\}$ 뿐이다. 따라서 잠복 공간<sub>latent space</sub> $\mathbf{z}$를 도입하여 문제 해결의 실마리를 찾는다. 변이 추론에서는 역 방향의 확률 분포 $p(\mathbf{z}|\mathbf{x})$를 구하는 일을 추론<sub>inference</sub>이라 부른다. 아래 식의 베이즈 정리<sub>Bayes' theorem</sub>는 추론을 위한 수식을 제공한다. $p(\mathbf{z}|\mathbf{x})$를 사후 확률<sub>posterior probability</sub>, $p(\mathbf{x}|\mathbf{z})$를 우도<sub>likelihood</sub>, $p(\mathbf{z})$를 사전 확률<sub>prior probability</sub>이라 부른다.
$$
p(\mathbf{z}|\mathbf{x})=\frac{p(\mathbf{x}|\mathbf{z})p(\mathbf{z})}{p(\mathbf{x})}=\frac{p(\mathbf{x},\mathbf{z})}{p(\mathbf{x})}
$$
변이 추론에서는 확률 분포의 집합 $Q$을 가정한 다음, $Q$에 속하는 확률 분포 중에서 참 확률 $p(\mathbf{z}|\mathbf{x})$에 가장 가까운 확률 분포 $\hat{q}$을 찾는 최적화 방법을 시도한다. 아래 식은 최적화 방법을 정의한다. $KL(a||b)$는 확률 분포 $a$가 $b$와 다른 정보를 측정해주는 쿨백-라이블러 발산<sub>Kullback-Leibler divergence</sub>이다.
$$
\hat{q}(\mathbf{z})=\text{argmin}_{q\in Q}\text{KL}(q(\mathbf{z})||p(\mathbf{z}|\mathbf{x}))
$$
변이 추론은 위 식의 $KL(q(\mathbf{z})||p(\mathbf{z}|\mathbf{x}))$에서 출발하여 출발하여 아래 식의 수식 전개를 통해 ELBO<sub>Evidence Lower BOund</sub>를 유도한다.
$$
\begin{align}
KL(q(\mathbf{z})||p(\mathbf{z}|mathbf{x})) =\mathbb{E}(\log q(\mathbf{z}))-\mathbb{E}(\log p(\mathbf{z}|\mathbf{x})) \\ 
= \mathbb{E}(\log q(\mathbf{z}))-\mathbb{E}(\log p(\mathbf{z},\mathbf{x}))+\log p(\mathbf{x}) \\
=-\underbrace{(\mathbb{E}(\log p(\mathbf{z},\mathbf{x}))-\mathbb{E}(\log q(\mathbf{z})))}_{ELBO(q)}+\log p(\mathbf{x})
\end{align}
$$
위 식을 아래 식으로 바꿔 쓸 수 있다.
$$
\log p(\mathbf{x})=ELBO(q)+KL(q(\mathbf{z})||p(\mathbf{z}|\mathbf{x}))
$$
ELBO를 아래 식과 같이 다시 쓸 수 있다.
$$
\begin{align}
ELBO(q)=\mathbb{E}(\log p(\mathbf{z},\mathbf{x}))-\mathbb{E}(\log q(\mathbf{z})) \\
=\mathbb{E}(\log p(\mathbf{x}|\mathbf{z}))+\mathbb{E}(\log p(\mathbf{z}))-\mathbb{E}(\log q(\mathbf{z})) \\
=\mathbb{E}(\log p(\mathbf{x}|\mathbf{z}))-KL(q(\mathbf{z})||p(\mathbf{z}))
\end{align}
$$
## 2.3. 변이 오토인코더
오토인코더가 생성 모델로 변신한 대표적 사례로 변이 오토인코더<sub>VAE; Variational AutoEncoder</sub>를 들 수 있다. 변이 인코더는 잠복 공간이 가우시안 확률 분포를 이루도록 규제함으로써 생성 모델로 발돋움한다. 
### 구조와 동작
오토인코더는 같은 영상을 다시 입력하면 같은 점으로 매핑하기 때문에 결정론적인 신경망이다. 반면 변이 오토인코더에 같은 영상을 입력하면 같은 확률 분포가 나오지만, 샘플링 과정에서 다른 $\mathbf{z}$가 발생하기 때문에 디코더로 복원한 $\mathbf{x}'$는 예전 입력 때와 다를 수 있다.
![[Pasted image 20250714133932.png]]
손실 함수는 ELBO를 사용한다. 변이 오토인코더를 위한 ELBO를 유도하면 아래 식이 된다.
$$
\log p_\theta (\mathbf{x}^i)=\underbrace{\mathbb{E}_{z\sim q_\phi (\mathbf{z}|\mathbf{x}^i)}(\log p_\theta (\mathbf{x}^i|\mathbf{z}))-KL(q_\phi(\mathbf{z}|\mathbf{x}^i)||p_\theta(\mathbf{z}))}_{ELBO\ L(\theta,\phi,\mathbf{x}^i)}+KL(q_\phi(\mathbf{z}|\mathbf{x}^i)||(p_\theta (\mathbf{z}|\mathbf{x}^i)))
$$
음수 ELBO를 손실 함수로 정의하면 아래 식의 최소화 문제가 된다. 이제 옵티마이저는 이 최적화 문제를 풀면 된다.
$$
\hat{\theta},\hat{\phi}=\text{argmin}_{\theta,\phi}-L(\theta,\phi,X)=\text{argmin}_{\theta,\phi}(\mathbb{E}_{\mathbf{z}\sim q_\phi (\mathbf{z}|\mathbf{x}^i)}(\log p_\theta (\mathbf{x}^i|\mathbf{z}))+KL(q_\phi (\mathbf{z}|\mathbf{x}^i)||p_\theta(\mathbf{z})))
$$
### MNIST로 변이 오토인코더 학습
``` python
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

zdim = 32

# Custom VAE Model Class
class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=(1, 1)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=(1, 1)),
            tf.keras.layers.Flatten(),
        ])
        
        self.fc_mu = tf.keras.layers.Dense(latent_dim)
        self.fc_log_var = tf.keras.layers.Dense(latent_dim)
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(3136),
            tf.keras.layers.Reshape((7, 7, 64)),
            tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=(1, 1)),
            tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=(2, 2)),
            tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=(2, 2)),
            tf.keras.layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same', strides=(1, 1))
        ])

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        batch_size = tf.shape(mu)[0]
        epsilon = tf.random.normal(shape=(batch_size, self.latent_dim), mean=0.0, stddev=0.1)
        return mu + tf.exp(0.5 * log_var) * epsilon

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

# Custom training step
@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        reconstruction, mu, log_var = model(x)
        
        # Reconstruction loss
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(x, reconstruction), axis=(1, 2)
            )
        )
        
        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)
        )
        
        total_loss = reconstruction_loss + kl_loss
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return total_loss, reconstruction_loss, kl_loss

# Create and train the model
vae = VAE(latent_dim=zdim)
optimizer = tf.keras.optimizers.Adam()

# Training
epochs = 10
batch_size = 128

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    
    # Shuffle data
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train_shuffled = x_train[indices]
    
    total_loss_avg = 0
    num_batches = len(x_train) // batch_size
    
    for i in range(0, len(x_train_shuffled), batch_size):
        batch_x = x_train_shuffled[i:i+batch_size]
        total_loss, recon_loss, kl_loss = train_step(vae, batch_x, optimizer)
        total_loss_avg += total_loss
    
    total_loss_avg /= num_batches
    print(f"  Loss: {total_loss_avg:.4f}")

# Create separate encoder and decoder models for inference
def sampling_layer(args):
    z_mean, z_log_var = args
    batch_size = tf.shape(z_mean)[0]
    epsilon = tf.random.normal(shape=(batch_size, zdim), mean=0.0, stddev=0.1)
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

encoder_input = tf.keras.layers.Input(shape=(28, 28, 1))
h = vae.encoder(encoder_input)
z_mean = vae.fc_mu(h)
z_log_var = vae.fc_log_var(h)
z = tf.keras.layers.Lambda(sampling_layer, output_shape=(zdim,))([z_mean, z_log_var])
model_encoder = tf.keras.models.Model(encoder_input, [z_mean, z_log_var, z])

decoder_input = tf.keras.layers.Input(shape=(zdim,))
decoder_output = vae.decoder(decoder_input)
model_decoder = tf.keras.models.Model(decoder_input, decoder_output)

# Interpolation
i = np.random.randint(x_test.shape[0])
j = np.random.randint(x_test.shape[0])
x = np.array([x_test[i], x_test[j]])
z = model_encoder.predict(x)[2]

zz = np.zeros((11, zdim))
alpha = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for i in range(11):
    zz[i] = (1.0 - alpha[i]) * z[0] + alpha[i] * z[1]

gen = model_decoder.predict(zz)
plt.figure(figsize=(15, 3))
for i in range(11):
    plt.subplot(1, 11, i+1)
    plt.imshow(gen[i].reshape(28, 28), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title(str(alpha[i]))
plt.tight_layout()
plt.show()
```
![[program13-5.png]]
## 2.4. 이산 변이 오토인코더
지금까지 오토인코더는 모두 실수 공간에서 동작한다. 이산 변이 오토인코더<sub>dVAE; discrete VAE</sub>는 실수 공간을 이산 공간으로 바꿈으로써 성능 향상을 꾀한다.
![[Pasted image 20250714144458.png]]
# 3. 생성 적대 신경망
생성 적대 신경망<sub>GAN; Generative Adversarial Network</sub>은 두 신경망이 적대적인 관계에서 학습하는 생성 모델이다.
## 3.1. 구조와 학습
GAN은 생성망 G와 분별망 D가 대립 관계를 가진다. 생성망의 목적은 분별망을 속이기 위해 진짜 같은 가짜를 생성하는 것이고 분별망의 목적은 생성망이 만들어낸 가짜를 훈련 집합에서 들어오는 진짜와 정교하게 구별하는 것이다. GAN에서 G와 D는 학습을 통해 점점 고도화된다. 학습을 마치면 생성망을 사용해 가짜 샘플을 만든다.
![[Pasted image 20250714144850.png]]
### 구조
GAN의 원래 논문은 완전연결층을 사용했는데 2016년에 컨볼루션층을 사용하는 DCGAN으로 발전한다.
![[Pasted image 20250714145041.png]]
### 학습
분별망은 입력된 영상을 가짜와 진짜의 두 부류로 분류하는 신경망이므로 그 자체로는 compile과 fit 함수를 사용해 학습하면 된다. 하지만 진짜는 훈련 집합에서 취하고 가짜는 생성망에서 취해야 하므로 생성망과 연결된 상태에서 학습이 이루어져야 한다.
생성망은 잠복 공간에서 난수로 생성한 벡터를 통과시켜 가짜 샘플을 생성한다. 그리고 생성된 가짜 샘플에 진짜를 부여하고 학습을 진행한다. 현재 분별망이 가중치를 고정해 생성망만 학습이 일어나게 설정하는 일이 중요하다. 현재 분별망이 가짜 샘플을 진짜로 오인하도록 생성망을 학습함으로써 생성망이 더욱 진짜 같은 샘플을 만들도록 개선하는 전략이다. 
생성망과 분별망은 각자 손실 함수를 가지고 대립 관계에서, 즉 둘이 승리 게임을 하면서 학습한다. 따라서 손실 함수 하나를 최적화하는 다른 문제에 비해 학습이 까다롭다. GAN 학습에서 가장 큰 문제는 모드 붕괴<sub>mode collapse</sub>다. 모드 붕괴가 나타나는 이유는 생성망과 분별망이 몇 개 모드만 잘 학습해도 손실 함수를 충분히 낮출 수 있기 때문이다. GAN의 궁극적인 목적은 진짜와 구별할 수 없는 가짜를 생성하는 것인데 궁극적인 목적을 직접적으로 표현하기 어렵기 때문에 승리 게임을 표현하는 수식을 손실 함수로 대신 사용하기 때문에 모드 붕괴가 발생한다.
## 3.2. GAN의 프로그래밍 실습
### fashion MNIST에서 한 부류 모델링
``` python
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = x_train[np.isin(y_train, [8])]  # Filter for class 8 (bags)
x_train = (x_train.astype('float32')/255.0)*2.0-1.0  # Normalize to [-1, 1]
x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension: (n, 28, 28, 1)

zdim = 100

def make_discriminator(in_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(64, (3,3), padding='same', 
                                  activation=keras.layers.LeakyReLU(alpha=0.2), 
                                  input_shape=in_shape))
    model.add(keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same', 
                                  activation=keras.layers.LeakyReLU(alpha=0.2)))
    model.add(keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same', 
                                  activation=keras.layers.LeakyReLU(alpha=0.2)))
    model.add(keras.layers.Conv2D(256, (3,3), strides=(2,2), padding='same', 
                                  activation=keras.layers.LeakyReLU(alpha=0.2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    # Fixed optimizer parameters
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                  metrics=['accuracy'])
    return model

def make_generator(zdim):
    model = keras.models.Sequential()
    
    # Start with 7x7x256
    model.add(keras.layers.Dense(7*7*256, activation=keras.layers.LeakyReLU(alpha=0.2), 
                                input_dim=zdim))
    model.add(keras.layers.Reshape((7, 7, 256)))
    
    # 7x7x256 -> 14x14x128
    model.add(keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', 
                                          activation=keras.layers.LeakyReLU(alpha=0.2)))
    
    # 14x14x128 -> 28x28x64
    model.add(keras.layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', 
                                          activation=keras.layers.LeakyReLU(alpha=0.2)))
    
    # 28x28x64 -> 28x28x1
    model.add(keras.layers.Conv2D(1, (3,3), padding='same', activation='tanh'))
    
    return model

def make_gan(G, D):
    D.trainable = False
    model = keras.models.Sequential()
    model.add(G)
    model.add(D)
    # Fixed optimizer parameters
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))
    return model

def generate_real_samples(dataset, n_samples):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    x = dataset[ix]
    y = np.ones((n_samples, 1))
    return x, y

def generate_latent_points(zdim, n_samples):
    return np.random.randn(n_samples, zdim)

def generate_fake_samples(G, zdim, n_samples):
    x_input = generate_latent_points(zdim, n_samples)
    x = G.predict(x_input, verbose=0)  # Added verbose=0 to reduce output
    y = np.zeros((n_samples, 1))
    return x, y

def train(G, D, GAN, dataset, zdim, n_epochs=200, batch_size=128, verbose=0):
    n_batch = int(dataset.shape[0] / batch_size)
    
    for epoch in range(n_epochs):
        for b in range(n_batch):
            # Train discriminator on real samples
            x_real, y_real = generate_real_samples(dataset, batch_size//2)
            d_loss1, _ = D.train_on_batch(x_real, y_real)
            
            # Train discriminator on fake samples
            x_fake, y_fake = generate_fake_samples(G, zdim, batch_size//2)
            d_loss2, _ = D.train_on_batch(x_fake, y_fake)
            
            # Train generator via GAN
            x_gan = generate_latent_points(zdim, batch_size)
            y_gan = np.ones((batch_size, 1))
            g_loss = GAN.train_on_batch(x_gan, y_gan)
            
        if verbose == 1:
            print('E%d: loss D(real)=%.3f, D(fake)=%.3f GAN=%.3f' % 
                  (epoch+1, d_loss1, d_loss2, g_loss))
            
        # Generate and display samples every 10 epochs
        if (epoch+1) % 10 == 0:
            x_fake, y_fake = generate_fake_samples(G, zdim, 12)
            plt.figure(figsize=(20, 2))
            plt.suptitle('Epoch ' + str(epoch+1))
            for k in range(12):
                plt.subplot(1, 12, k+1)
                # Convert from [-1, 1] to [0, 1] for display
                plt.imshow((x_fake[k].squeeze() + 1) / 2.0, cmap='gray')
                plt.xticks([])
                plt.yticks([])
            plt.show()

# Create models
D = make_discriminator((28, 28, 1))
G = make_generator(zdim)
GAN = make_gan(G, D)

# Print model summaries
print("Discriminator:")
D.summary()
print("\nGenerator:")
G.summary()

# Start training
print(f"Training dataset shape: {x_train.shape}")
print("Starting GAN training...")
train(G, D, GAN, x_train, zdim, verbose=1)
```
``` text
E1: loss D(real)=0.714, D(fake)=0.714 GAN=0.655
E2: loss D(real)=0.732, D(fake)=0.732 GAN=0.623
E3: loss D(real)=0.753, D(fake)=0.754 GAN=0.589
E4: loss D(real)=0.770, D(fake)=0.771 GAN=0.564
E5: loss D(real)=0.784, D(fake)=0.784 GAN=0.546
E6: loss D(real)=0.795, D(fake)=0.795 GAN=0.531
E7: loss D(real)=0.803, D(fake)=0.804 GAN=0.519
E8: loss D(real)=0.810, D(fake)=0.810 GAN=0.510
E9: loss D(real)=0.816, D(fake)=0.816 GAN=0.503
E10: loss D(real)=0.820, D(fake)=0.820 GAN=0.497
```
![[program13-6.png]]
## 3.3. GAN의 전개: CycleGAN, ProGAN, SAGAN
CycleGAN은 영상을 내용과 스타일로 분해한 다음에 스타일만 변화시키는 방법으로 사진을 특정 화가의 화풍으로 변환하거나 무늬를 변환해 말을 얼룩말로 변환하는 일을 한다. 무작위로 고른 사진 집합과 무작위로 고른 고흐 그림 집합을 주고 학습하면 사진을 고흐풍의 그림으로 변환해주는 GAN을 얻는다.
ProGAN은 고해상도 얼굴 영상을 생성한다. 한꺼번에 고해상도 영상을 생성하는 일이 어렵기 때문에 먼저 $4 \times 4$ 저해상도 영상을 생성하는 신경망을 학습한 후 학습된 신경망에 $8 \times 8$을 생성하는 층을 추가하고 다시 학습한다. 이런 과정을 $1024 \times 1024$를 얻을 때까지 반복한다.
SAGAN은 자기 주목을 GAN에 적용해 성능을 개선한다. SAGAN은 컨볼루션 신경망을 백본으로 사용한 채 자기 주목을 적용했는데 이후에 트랜스포머를 백본으로 사용하는 GAN으로 넘어간다.
## 3.4. 트랜스포머 GAN
### TransGAN
TransGAN은 인코더 블록만 가져다 생성망과 분별망에 사용한다. 입력은 잠복 공간의 벡터고 출력은 영상이라는 사실은 변함이 없다. 또한 잠복 공간의 벡터를 출력 영상의 크기에 맞추어 한번에 키운 다음에 같은 크기를 유지할 수 있는데 그렇게 하면 주목 행렬의 크기가 커서 메모리를 감당할 수 없게 된다.
TransGAN은 두 가지 아이디어로 이 문제를 해결한다. 첫 번째 아이디어는 작은 해상도에서 시작해 업 샘플링을 통해 점점 키워나간다. 생성망에는 높은 해상도를 다루기 위한 또 다른 아이디어로 격자 자기 주목<sub>grid self-attention</sub>이 있는데 이게 두 번째 아이디어다. 
# 4. 확산 모델
비평형 열역학<sub>non-equilibrium thermodynamics</sub>은 기체 운동을 연구하는 학문이다. 확산 모델<sub>diffusion model</sub>은 비평형 열역학에서 아이디어를 얻어 개발된 생성 모델이다.
$t-1$ 순간의 영상 $\mathbf{x}_{t-1}$에 잡음을 조금 섞어 $t$ 순간의 영상 $\mathbf{x}_t$를 만든다. 이런 연산을 원래 영상 $\mathbf{x}_0$에서 출발하여 $\mathbf{x}_T$가 될 따까지 반복한다. 시간 단계<sub>time steps</sub> T를 충분히 길게 하면 $\mathbf{x}_T$는 원래 영상의 내용을 모두 잃고 가우시안 잡음으로만 구성된다. 이 과정 전반을 잔방 확산<sub>forward diffusion</sub>이라 부른다. 전방 확산은 학습 없이 일정한 확률 규칙에 따라 동작한다. 전방 확산이 끝나면 역 디노이징<sub>reverse denoising</sub> 과정을 이용하여 $\mathbf{x}_T$에서 출발하여 원래 영상 $\mathbf{x}_0$을 복원한다. 역 디노이징은 신경망을 이용하여 달성하기 때문에 학습이 필요하다.
![[Pasted image 20250715094625.png]]
### 전방 확산
아래 식은 $\mathbf{x}_{t-1}$에 잡음을 섞어 $\mathbf{x}_t$를 만드는 수식이다. 이 식은 바로 직전 순간 $t-1$의 상태만 보고 현재 순간 $t$의 상태를 결정할 수 있다는 가정을 근거로 하기 때문에 마르코프 체인<sub>Markov chain</sub>을 형성한다. 잡음 스케줄<sub>noise schedule</sub> $\beta_t$는 [0,1] 사이의 실수로서 $t$ 순간에 추가하는 잡음의 정도를 조정한다.
$$
q(\mathbf{x}_t|\mathbf{x}_{t-1})=N(\sqrt{1-\beta_t}\mathbf{x}_{t-1},\beta_t\mathbf{I})
$$
위 식을 연속으로 적용하면 입력 영상에서 $\mathbf{x}_0$에서 출발하여 $\mathbf{x}_1,\mathbf{x}_2, \cdots,\mathbf{x}_t$를 만들어서 임의 순간 $t$의 영상 $\mathbf{x}_t$를 만들 수 있다. 그런데 이런 순차적인 방법은 시간이 많이 걸리기 때문에 $\mathbf{x}_0$에서 바로 $\mathbf{x}_t$를 만드는 방법이 필요하다. 이 일은 $\alpha_t=1-\beta_t$로 재매개변수화<sub>reparmetrization</sub>하고 아래 식을 유도함으로써 가능해진다.
$$
q(\mathbf{x}_t|\mathbf{x}_0)=N(\sqrt{\bar{a}_t}\mathbf{x}_0,(1-\bar{\alpha})\mathbf{I})
$$
$\mathbf{x}_t$의 실제 샘플링은 위 식에서 유도된 아래 식을 이용하여 수행한다.
$$
\mathbf{x}_t=\sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\boldsymbol{\varepsilon}
$$
### 역 디노이징
역 디노이징 과정을 구현하려면 $\mathbf{x}_t$에서 $\mathbf{x}_{t-1}$이 발생할 확률 분포를 계산해야 한다.
전방 확산에서 $\beta$를 충분히 작게 설정하면 역 디노이징 과정의 확률 분포를 가우시안으로 간주할 수 있다는 정리가 있다. 이에 따르면 신경망이 추정하는 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$는 가우시안이고 아래 식으로 표현할 수 있다.
$$
p_\theta=(\mathbf{x}_{t-1}|\mathbf{x}_t)=N(\mu_\theta(\mathbf{x}_t,t),\sigma_\theta(\mathbf{x}_t,t))=N(\mu_\theta(\mathbf{x}_t,t),C)
$$
신경망이 할 일은 $\mathbf{x}_t$가 주어졌을 때 전방 확산 과정에서 섞은 잡음을 제거하여 $\mathbf{x}_{t-1}$을 복원하는 것이다. 위 식에 따르면 가우시안 평균 $\mu_\theta$를 통해 손실 함수를 정의할 수 있는데, 실제로는 잡음 $\varepsilon_\theta$를 통해 아래 식과 같이 손실 함수를 정의한다.
$$
J(\theta)=||\boldsymbol{\varepsilon}-\varepsilon_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t},\varepsilon,t)||^2_2=||\boldsymbol{\varepsilon}-\varepsilon_\theta(\mathbf{x}_t,t)||^2_2
$$
### 학습 알고리즘과 추론 알고리즘
아래 알고리즘은 DDPM이 사용하는 학습 알고리즘이다.
> [!info] DDPM의 학습
> 입력: 데이터셋 $X$, 시간 단계 $T$
> 출력: 잡음을 추정하는 신경망의 최적 가중치 $\hat{\theta}$
> 1. 신경망의 가중치 $\theta$를 초기화한다.
> 2. repeat
> 	1. $X$에서 랜덤하게 샘플 하나를 취해 $\mathbf{x}_0$이라 한다.
> 	2. $1,2,\cdots,T$에서 랜덤하게 하나를 취해 $t$라 한다.
> 	3. 표준 가우시안 $N(\mathbf{0,I})$에서 한 점을 샘플링하여 $\boldsymbol{\varepsilon}$이라 한다.
> 	4. $\sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\boldsymbol{\varepsilon}$를 계산하여 $\mathbf{x}_t$를 구한다.
> 	5. 신경망으로 $\varepsilon_\theta(\mathbf{x}_t,t)$를 예측하여 $\boldsymbol{\varepsilon}'$라 한다.
> 	6. $||\boldsymbol{\varepsilon}-\boldsymbol{\varepsilon}'||^2_2$의 그레이디언트를 구하여 $\Delta\theta$에 저장한다.
> 	7. $\theta=\theta-\rho\Delta\theta$
> 3. until(수렴)
> 4. $\hat{\theta}=\theta$

위 알고리즘으로 학습을 마치고 U-net을 이용하여 새로운 샘플을 생성한다.
> [!info] DDPM에서 추론(생성)
> 입력: 잡음을 추정하는 신경망의 가중치 $\theta$, 시간 단계 $T$
> 출력: 새로운 샘플 $\mathbf{x}$
> 1. 표준 가우시안 $N(\mathbf{0,I})$에서 한 점을 샘플링하여 $\mathbf{x}_T$라 한다.
> 2. `for t = T, T-1, ... , 2, 1`
> 	1. `if (t > 1),` $N(\mathbf{0,I})$에서 한 점을 샘플링하여 $\mathbf{z}$라 한다.
> 	2. `else z=0`
> 	3. $\mathbf{x}_{t-1}=\frac{1}{\sqrt{\alpha}_t}\left(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha_t}}}\varepsilon_\theta(\mathbf{x}_t,t)\right)+\alpha_t\mathbf{z}$
> 3. $\mathbf{x}=\mathbf{x}_0$
### 확산 모델의 특성
생성 모델은 주어진 데이터셋을 충실히 모델링하여 빗스한 샘플을 생성하는 충실성<sub>fidelity</sub>과 데이터셋을 골고루 발생시키는 다양성<sub>diversity</sub>을 동시에 만족해야한다. 
# 5. 생성 모델의 평가
- 충실성: 주어진 데이터셋을 잘 모방하여 비슷하지만 같지 않은 샘플을 생성한다.
- 다양성: 데이터셋을 이루는 모든 영상을 빼놓지 않고 골고루 반영한다.
자동 평가에는 인셉션 점수<sub>IS; Inception Score</sub>, 프레셰 인셉션 거리<sub>FID; Frechet Inception Distance</sub>, 커널 인셉션 거리<sub>KID; Kernel Inception Distance</sub>를 주로 사용한다.
### 인셉션 점수
인셉션 점수 IS는 생성 모델이 출력한 영상 $\mathbf{x}$를 사전 학습된 신경망에 입력하여 분류를 수행하고, 분류 결과를 $\mathbf{y}$로 표기한다.
![[Pasted image 20250715104413.png]]
$$
IS=\text{exp}\left(\frac{1}{n}\sum_{i=1,n}\text{KL}(p(\mathbf{y}^i|\mathbf{x}^i)||p(\mathbf{y}))\right)=\text{exp}\left(\mathbb{E}_X\text{KL}(p(\mathbf{y}|\mathbf{x})||p(\mathbf{y}))\right)
$$
IS는 생성 모델을 위한 자동 평가 척도를 처음 제시했다는 데 의미가 있지만 여러가지 한계점이 있다. 예를 들어 한 부류에 대해 항상 똑같은 영상 하나만 생성하는 경우 아주 높은 점수를 받는다. 즉 부류 간의 다양성은 고려하지만 부류 내 다양성은 무시하는 문제가 있다. 프레셰 인셉션 거리와 커널 인셉션 거리는 IS의 한계점을 개선한다.
### 프레셰 인셉션 거리
프레셰 인셉션 거리 FID는 생성된 샘플의 특징 벡터와 데이터셋에 있는 진짜 샘플의 특징 벡터를 비교하여 점수를 계산함으로써 한계를 극복한다. 두 특징 벡터가 비슷할수록 좋은 품질을 의미하기 때문에 FID는 점수가 낮을수록 좋다.
$$
d^2(N(\boldsymbol{\mu},\boldsymbol{\Sigma}),N(\boldsymbol{\mu}_w,\boldsymbol{\Sigma}_w))=||\boldsymbol{\mu}-\boldsymbol{\mu}_w||^2_2+\text{Tr}\left(\boldsymbol{\Sigma}+\boldsymbol{\Sigma}_w-2(\boldsymbol{\Sigma\Sigma}_w)^{1/2}\right)
$$
### 커널 인셉션 거리
FID는 샘플의 개수를 달리 하여 측정하면 심하게 다른 값이 나오는 편향 문제를 알고 있다. 커널 인셉션 거리 KID는 편향 문제를 크게 누그러뜨린다. KID는 아래 식이 제시하는 MMD<sub>Maximum Mean Distance</sub>로 측정한다.
$$
\begin{align}
\text{KID} =\text{MMD}(P,Q)  =\frac{1}{m(m-1)}\sum_{\mathbf{x}^i\in P}\sum_{\mathbf{x}^j\in P,j\neq i}k(\mathbf{x}^i,\mathbf{x}^j)+\frac{1}{m(m-1)}\sum_{\mathbf{x}^i\in Q}\sum_{\mathbf{x}^j\in Q, j\neq i}k(\mathbf{x}^i,\mathbf{x}^j)-2\frac{1}{mm}\sum_{\mathbf{x}^i\in P}\sum_{\mathbf{x}^j \in Q}k(\mathbf{x}^i,\mathbf{x}^j)
\end{align}
$$
# 6. 멀티 모달 생성 모델: 언어와 비전의 결합
생성 모델이 발전하면서 보다 적극적인 결합으로 전환되었다. 이러한 결합은 더욱 어려운 역방향 문제, 즉 설명 문장을 주면 걸맞은 영상을 생성하는 언어-비전 멀티 모달 연구를 촉발했다.
DALLE, Imagen, Midjourney와 같은 생성 모델은 제로샷<sub>zero-shot</sub> 학습을 한다고 말한다. 
