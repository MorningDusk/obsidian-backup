---
tags:
  - ai
  - computer_vision
  - article
  - 6d
date: 2025-07-15
link: https://arxiv.org/abs/2312.08344
연구 목적: 새로운 객체에 대해 6D 자세 추정과 추적을 통합적으로 수행하는 foundation model 개발 (model-based와 model-free 설정 모두 지원)
연구 방법: LLM 기반 texture augmentation과 대규모 synthetic training, neural implicit representation을 통한 novel view synthesis, transformer 기반 architecture와 contrastive learning을 결합한 통합 프레임워크 구축
결과 변수: AUC of ADD/ADD-S metrics, ADD-0.1d recall, BOP challenge의 VSD/MSSD/MSPD 평균 recall (AR) 점수를 통한 6D pose estimation과 tracking 성능 평가
주요 결과: model-based/model-free pose estimation 및 tracking 4개 task 모두에서 기존 specialized methods 대비 큰 폭의 성능 향상을 달성하고 BOP leaderboard 1위를 기록하며 instance-level methods와 유사한 성능을 더 적은 가정으로 실현
---
# Introduction
## 연구 배경과 문제점
**6D pose estimation**은 물체의 3D 위치 (x,y,z)와 3D 회전 (roll, pitch, yaw)을 모두 추정하는 핵심적인 컴퓨터 비전 기술이다. 이는 로봇 조작, 가상 현실, 자율 주행 등 다양한 실제 응용에서 필수적인 기능이다. 하지만 기존 연구들은 여러 근본적인 한계점들을 가지고 있다.
- **Instance-level**: 학습 시점에 결정된 특정 물체 인스턴스에만 작동한다. 새로운 물체에는 전혀 적용할 수 없고, 보통 textured CAD 모델이 필요하다.
- **Category-level**: 미리 정의된 카테고리 내의 물체들만 처리 가능하다. 카테고리 수준의 training data 확보가 매우 어렵고, pose canonicalization 과정이 복잡하다.
- **설정 분리 문제**: Model-based(CAD 모델 제공)와 model-free(참조 이미지 제공) 설정이 완전히 분리되어 있어, 실제 응용에서 유연성이 부족하다.
## 기여점
이 논문은 **FoundationPose**라는 통합 foundation model을 제안하여 위의 문제들을 해결한다. 
1. 처음 보는 새로운 물체에 대해 model-based와 model-free 설정을 모두 지원하는 단일 시스템을 구축했다. 이는 실제 응용에서 매우 중요한 유연성을 제공한다.
2. 단일 프레임에서 pose estimation과 비디오 시퀀스에서 tracking을 동시에 수행할 수 있는 **통합 framework**를 개발했다.
3. large-scale synthetic training과 LLM을 활용하여 **강력한 일반화 성능**을 달성했다. 기존 방법들이 실제 데이터의 부족으로 어려움을 겪었다면, 이 연구는 합성 데이터의 품질과 다양성을 획기적으로 향상시켜 이 문제를 해결했다. 
4. 각 task별로 특화된 기존 방법들을 크게 앞서는 **state-of-the-art 성능**을 달성했다. 특히 instance-level 방법들과 비교할 만한 결과를 더 적은 가정으로 실현했다.
# Related Work
## CAD Model-based Object Pose Estimation
Instance-level 방법들은 6D pose estimation의 전통적인 접근법으로, 특정 물체의 textured CAD 모델을 필요로 한다. 이들의 주요 접근법은 크게 세 가지로 나뉜다. 
1. **Direct regression**: neural network가 직접 6D pose를 예측하는 방식이다.
2. **2D-3D correspondence**: 2D keypoint와 3D model point를 매칭한 후 PnP solve를 이용한다. 
3. **3D-3D correspondence**: point cloud 간 매칭 후 least squares fitting을 수행한다.
하지만 이 모든 방법들은 학습 시점에 본 특정 물체에만 적용 가능하다는 근본적인 한계가 있다.
Category-level 방법들은 이런 한계를 어느정도 완화하여 같은 카테고리 내의 새로운 인스턴스에 적용할 수 있다. NOCS, 6-PACK 등이 대표적인 예이지만, 여전히 미리 정의된 카테고리를 벗어날 수 없다는 제약이 있다. 최근 연구들은 이런 문제를 해결하기 위해 임의의 새로운 물체에 대해 CAD 모델만 주어지면 즉시 적용 가능한 방법들을 개발하고 있다. MegaPose, OSOP 등이 이런 접근법의 대표적인 예이다.
## Few-shot Model-free Object Pose Estimation
Model-free 방법들은 CAD 모델 대신 참조 이미지 몇 장만으로 물체를 인식하려는 시도들이다. 
- **RLLG와 NeRF-Pose**: instance-wise training을 수행하지만 CAD 모델 없이 학습하는 방법을 제안했다. 하지만 이들은 여전히 특정 물체에 대한 학습이 필요하다는 한계가 있다.
- **Gen6D**: detection, retrieval, refinement pipeline을 사용하지만, out-of-distribution 문제로 인해 fine-tuning이 필요하다. 
- **OnePose 시리즈**: Structure-from-Motion을 활용한 object modeling과 2D-3D matching network를 사용하는 흥미로운 접근법을 제시했다. 
- **FS6D**: RGBD 정보를 활용하지만 여전히 target dataset에서 fine-tuning이 필요하다는 단점이 있다.
이런 기존 방법들의 공통적인 한계점은 대부분 correspondance에 의존한다는 것이다. 이로 인해 textureless 물체나 심한 occulation 상황에서 매우 취약하다. 또한 새로운 물체에 대한 일반화 성능이 부족하여 실제 응용에서 제약이 많다.
## Object Pose Tracking
Object Pose Tracking은 시간적 정보를 활용해 더 효율적이고 부드러운 pose 예측을 수행하는 분야이다. 이 분야 역시 pose estimation과 마찬가지로 instance-level, category-level, novel object 방법들로 나뉜다.
- **Instance-level**: se(3)-TrackNet, PoseRBPF
- **Category-level**: 6-PACK
- **Novel object tracking**: BundleTrack, Wuthrich et al
기존 tracking 방법들의 문제점은 대부분 pose estimation과 분리되어 있어서 초기 pose를 외부에서 제공받아야 한다는 것이다. 또한 tracking이 실패했을 때 자동으로 re-initialization을 수행하는 능력이 부족하다. 이런 문제들로 인해 실제 응용에서는 robust한 성능을 보장하기 어렵다.
# Approach
## Language-aided Data Generation at Scale
### 3D Assets 수집 전략
강력한 일반화 성능을 달성하기 위해서는 대규모의 다양한 물체와 장면이 필요하다. 하지만 실제 세계에서 이런 데이터를 수집하고 정확한 ground-truth 6D pose를 annotation하는 것은 시간과 비용 측면에서 매우 비현실적이다. 반면 기존의 합성 데이터는 3D asset의 크기와 다양성이 부족했다.
이 연구에서는 최신 대규모 3D 데이터베이스들을 활용하는 전략을 채택했다. **Objaverse**에서는 40K개 이상의 물체를 포함하는 Objaverse-LVIS subset을 사용했다. 이 subset은 1156개 LVIS 카테고리에 속하는 가장 관련성이 높은 일상 물체들로 구성되어 있으며, 합리적인 품질과 다양한 형태 및 외관을 보장한다. 또한 각 물체에는 카테고리를 설명하는 tag가 포함되어 있어 후속 LLM-aided texture augmentation 단계에서 자동 language prompt 생성에 활용된다. **GSO**(Google Scanned Objects)에서도 고품질로 스캔된 일상 물체들을 추가로 확보했다.
### LLM-aided Texture Augmentation의 혁신
기존 FS6D의 random texture blending 방식은 심각한 문제점들을 가지고 있었다. Random UV mapping으로 인해 결과 texture mesh에서 이상한 seam이 생성되고, ImageNet이나 MS-COCO의 holistic scene 이미지를 물체에 적용하면 매우 비현실적인 결과가 나온다.
이 연구에서는 최근 large language model과 diffusion model의 발전을 활용하여 더 현실적이고 완전 자동화된 texture augmentation을 탐구했다. 핵심은 hierarchial prompt 전략이다.
1. ChatGPT에게 물체의 tag를 제공하여 가능한 외관을 설명하는 자연스러운 텍스트를 생성하도록 한다. 예를 들어 `A traditional wooden armoire in a rich mahogany finish, showcasing intricate carvings and brass hardware for an elegant look`와 같은 설명이 생성된다.
2. ChatGPT가 생성한 설명을 **TexFusion** diffusion model에 입력하여 augmented textured model을 생성한다. 이 과정은 object shape과 randomly initialized noisy texture도 함께 제공된다. 이런 방식은 완전히 자동화되어 있어 대규모 물체들에 대한 다양한 스타일의 texture 증강을 가능하게 한다. 수동으로 프롬프트를 제공하는 것은 확장성이 없기 때문에, 이런 게층적 전략이 매우 중요하다.
### 합성 데이터 생성 파이프라인
실제 데이터 생성은 **NVIDIA Isaac Sim**에서 path tracing을 활용한 고품질 photorealistic rendering으로 구현되었다. 물리적으로 타당한 장면을 생성하기 위해 gravity와 physics simulation을 수행한다. 각 scene에서는 원본과 texture가 증강된 버전을 포함하여 물체들을 무작위로 샘플링한다. 물체 크기, 재질, 카메라 pose, 조명도 모두 무작위하여 최대한 다양한 상황을 포함하도록 했다.
전체적으로 약 60만 개의 scene과 120만 개의 이미지를 생성했으며, 각 이미지에는 RGBD 정보뿐만 아니라 object segmentation, camera parameter, object pose정보도 함께 저장된다. 이런 대규모 고품질 합성 데이터가 모델의 강력한 일반화 성능의 기반이 된다.
## Neural Object Modeling
### Field Representation의 핵심 설계
Model-free 설정에서 3D CAD 모델이 없을 때 핵심적인 도전은 물체를 효과적으로 표현하여 downstream module에 충분한 품질의 이미지를 렌더링하는 것이다. Neural implicit representation은 novel view synthesis에 효과적이면서 GPU에서 병렬 처리가 가능하여, multiple pose hypothesis를 렌더링할 때 높은 게산 효율성을 제공한다.
이 연구에서는 object-centric neural field representation을 도입했다. Geometric function $\Omega$: $x$ -> $s$는 3D 점 $x \in \mathbb{R}^3$을 입력으로 받아 signed distance value $s \in \mathbb{R}$을 출력한다. Appearance function $\Phi$: $(f_\Omega(x),n,d)$ -> $c$는 geometric network의 intermediate feature vector $f_\Omega(x)$, point normal $n \in \mathbb{R}^3$, view direction $d \in \mathbb{R}^3$을 입력으로 받아 color $c \in \mathbb{R}_+^3$를 출력한다.
실제 구현에서는 $x$에 multi-resolution hash encoding을 적용한 후 network에 전달한다. Normal $n$과 view direction $d$는 고정된 second-order spherical harmonic coefficient 집합으로 embedding된다. Implicit object surface는 signed distance field의 zero level set으로 정의된다.
$$
S=\{x\in \mathbb{R}^3|\Omega(x)=0\}
$$
NeRF와 비교했을 때 SDF representation의 장점은 더 높은 품질의 depth rendering이 가능하고 density threshold를 수동으로 선택할 필요가 없다는 것이다.
### Field Learning의 수학적 기초
Texture learning을 위해 truncated near-surface region에서 volumetric rendering을 따른다. Color rendering은 다음과 같이 정의된다.
$$
c(r)=\int_{z(r)-\lambda}^{z(r)+0.5\lambda}w(x_i)\Phi(f_{\Omega(x_i)},n(x_i),d(x_i))dt
$$
여기서 $w(x_i)$는 bell-shaped probability density function으로 다음과 같다.
$$
w(x_i)=\frac{1}{1+e^{-\alpha\Omega(x_i)}}\frac{1}{1+e^{\alpha\Omega(x_i)}}
$$
이 확률은 implicit object surface와의 signed distance $\Omega (x_i)$에 의존하며, $\alpha$는 distribution의 softness를 조정한다. 확률은 surface intersection에서 peak를 가진다. $z(r)$은 depth image에서 ray의 depth value이고, $\lambda$는 truncation distance이다. 더 효율적인 학습을 위해 surface에서 $\lambda$보다 멀리 떨어진 empty space의 기여도는 무시하고 self-occulasion을 모델링하기 위해 $0.5\lambda$ penetrating distance까지만 적분한다.
학습 과정에서는 여러 loss function을 사용한다. 
#### Color supervision loss
$$
\mathcal{L}_c = \frac{1}{|\mathcal{R}|}\sum_{r \in \mathcal{R}} \|c(r) - \bar{c}(r)\|_2
$$
여기서 $\bar{c}(r)$은 ray $r$이 지나가는 pixel에서의 ground-truth color이다. 
#### Empty space loss
$$
\mathcal{L}_e = \frac{1}{|\mathcal{X}_e|}\sum_{x \in \mathcal{X}_e} |\Omega(x) - \lambda| 
$$
#### Near surface loss
$$
\mathcal{L}_s = \frac{1}{|\mathcal{X}_s|}\sum_{x \in \mathcal{X}_s} (\Omega(x) + d_x - d_D)^2
$$
#### 전체 training loss
$$
\mathcal{L} = w_c\mathcal{L}_c + w_e\mathcal{L}_e + w_s\mathcal{L}_s + w_{eik}\mathcal{L}_{eik}
$$
학습은 prior 없이 물체별로 최적화되며 몇 초 안에 효율적으로 수행된다. Neural field는 새로운 물체에 대해 한 번만 학습하면 된다.
### Efficient Rendering Pipeline
학습이 완료되면 neural field는 기존 graphics pipeline을 대체하여 subsequent render-and-compare iteration에서 효율적인 물체 렌더링에 사용될 수 있다. 기존 NeRF의 color rendering 외에도 RGBD 기반 pose estimation과 tracking을 위해 depth rendering이 필요하다. 이를 위해 marching cubes를 수행하여 SDF의 zero level set에서 textured mesh를 추출하고, color projection을 결합한다. 이 과정은 각 물체마다 한 번만 수행하면 된다.
Inference 시에는 주어진 object pose에 대해 rasterization process를 따라 RGBD 이미지를 렌더링한다. 대안으로는 sphere tracing을 사용하여 $\Omega$로 직접 depth image를 온라인으로 렌더링할 수도 있지만, 특히 병렬로 렌더링해야 할 pose hypothesis가 많을 때는 효율성이 떨어진다는 것을 발견했다.
## Pose Hypothesis Generation
### Pose Initialization 전략
RGBD 이미지가 주어지면 Mask R-CNN이나 CNOS 같은 기성 방법을 사용하여 물체를 detection한다. Translation 초기화는 detected 2D bounding box 내의 median depth에 위치한 3D point를 사용한다. Rotation을 초기화하기 위해서는 물체를 중심으로 하는 icosphere에서 $N_s=42$개의 viewpoint를 uniformly sampling한다. 이때 카메라는 중심을 향하도록 배치된다.
이런 camera pose들은 $N_i=12$개의 discretized in-plane rotation으로 추가 증강되어, 총 $N_s \times N_i=504$개의 global pose initialization이 pose refiner의 입력으로 전달된다. 이런 systematic sampling 전략은 다양한 viewpoint를 고르게 커버하면서도 computational load를 관리 가능한 수준으로 유지한다.
### Pose Refinement Network Architecture
이전 단계의 coarse pose initialization들은 종종 상당히 noisy하기 때문에 pose 품질을 개선하는 refinement module이 필요하다. Pose refinement network는 coarse pose에 conditioning된 물체의 rendering과 카메라에서 관찰된 input의 crop을 입력으로 받아 pose 품질을 개선하는 pose update를 출력한다.
MegaPose와 달리 coarse pose 주변의 multiple view를 렌더링하여 anchor point를 찾는 방식 대신, 이 연구에서는 coarse pose에 해당하는 single view를 렌더링하는 것으로 충분하다는 것을 발견했다. Input observation의 경우 constant한 2D detection에 기반한 cropping 대신 pose-conditioned cropping 전략을 수행하여 translation update에 피드백을 제공한다.
구체적으로는 object origin을 image space에 projection하여 crop center를 결정한다. 그 다음 slightly enlarged object diameter(물체 표면의 임의의 point pair 간 최대 거리)를 projection하여 물체와 pose hypothesis 주변의 nearby context를 포함하는 crop size를 결정한다. 이런 crop은 coarse pose에 conditioning되어 있으며, network가 translation을 업데이트하여 crop이 observation과 더 잘 정렬되도록 장려한다.
Refinement process는 최신 업데이트된 pose를 다음 inference의 입력으로 feeding하여 여러 번 반복될 수 있으며, 이를 통해 pose 품질을 iteratively 개선한다. Network는 두 RGBD input branch에서 single shared CNN encoder로 feature map을 추출한다. Feature map들이 concatenate되고, residual connection이 있는 CNN block들에 입력된 후, positioning embedding과 함께 patch로 나누어 tokenization된다.
최종적으로 network는 translation update $\Delta t \in \mathbb{R}^3$과 rotation update $\Delta R\in SO(3)$을 예측하며, 각각 individual transformer encoder로 처리되고 output dimension으로 linearly project된다. $\Delta t$는 camera frame에서 물체의 translation shift를 나타내고, $\Delta R$은 camera frame에서 orientation update를 나타낸다. 실제로는 rotation이 axis-angle representation으로 parameterize된다.
Input coarse pose $[R|t] \in SE(3)$는 다음과 같이 업데이트된다.
$$
t^+=t+\Delta t
$$
$$
R^+=\Delta R \otimes R
$$
여기서 $\otimes$는 SO(3)에서의 update를 나타낸다. Single homogeneous pose update를 사용하는 대신 이런 disentangled representation은 translation update를 적용할 때 updated orientation에 대한 dependency를 제거한다. 이는 update와 input observation을 모두 camera coordinate frame에서 통일하여 학습 과정을 단순화한다.
Network training은 L2 loss로 supervise된다.
$$
\mathcal{L}_{refine} = w_1 \|\Delta t - \Delta \bar{t}\|_2 + w_2 \|\Delta R - \Delta \bar{R}\|_2
$$
여기서 $\Delta \bar{t}$와 $\Delta\bar{R}$은 ground truth이고, $w_1$과 $w_2$는 loss들의 균형을 맞추는 weight로 epirically 1로 설정된다.
## Pose Selection
### Hierarchical Comparison의 2단계 전략
