---
date: 2025-07-03
tags:
  - ai
  - computer_vision
  - deep_learning
---
# 1. 모션 분석
비디오<sub>video</sub>는 시간 순서에 따라 정지 영상을 나열한 구조로, 동영상<sub>dynamic image</sub>이라 부르기도 한다. 비디오를 구성하는 영상 한 장을 프레임<sub>frame</sub>이라고 한다. 프레임은 2차원 공간에 정의되는데, 비디오는 시간 축이 추가되어 3차원 시공간<sub>spatio-temporal</sub>을 형성한다. 컬러 영상은 채널이 세 장이므로 $m \times n \times 3$ 텐서이고 $T$장의 프레임을 담은 비디오는 $m \times n \times 3 \times T$의 4차원 구조 텐서다.
![[Pasted image 20250703102553.png]]
비디오를 분석하는 초창기 연구는 카메라와 조명, 배경이 고정된 단순한 상황을 가정했다. 배경이 고정된 아래 식의 차영상<sub>difference image</sub>을 분석해 쓸 만한 정보를 얻을 수 있다. $f(j,i,t)$는 $t$순간 프레임의 $(j,i)$ 화솟값이다. $f(,,0)$은 배경만 두고 획득한 영상으로 기준 프레임 역할을 한다.
$$
d(j,i,t)=|f(j,i,0)-f(j,i,t)|,0 \leq j \lt m, 0\leq i \lt n, 1 \leq t < T
$$
이후로는 일반적인 비디오를 처리하는 연구로 발전했는데, 초창기에는 광류<sub>optical flow</sub>를 이용하는 접근 방법이 주류를 이루었다.
## 1.1. 모션 벡터와 광류
움직이는 물체는 연속 프레임에 명암 변화를 일으킨다. 따라서 명암 변화를 분석하면 역으로 물체의 모션 정보를 알아낼 수 있다. 화소별로 모션 벡터<sub>motion vector</sub>를 추정해 기록한 맵을 광류<sub>optical flow</sub>라고 한다. 
### 광류 추정 방법
모션 벡터를 추정하는 알고리즘은 애매함을 해결하려고 연속한 두 영상에서 같은 물체는 같은 명암으로 나타난다는 밝기 형상성<sub>brightness constancy</sub> 조건을 가정한다. 두 영상의 시간 차이인 $dt$가 충분하 작다면 테일러 급수에 따라 아래 식이 성립한다.
$$
f(y+dy, x+dy, t+dt)=f(y,x,t)+\frac{\partial f}{\partial y}dy + \frac{\partial f}{\partial x}dx + \frac{\partial f}{\partial t}dt + 2차\ 이상의\ 합
$$
$dt$가 작다는 가정에 따라 2차 이상을 무시한다. 밝기 향상성 가정을 적용하면 $dt$동안 $(dy,dx)$만큼 이동하여 형성된 $f(y+dy,x+dy,t+dt)$는 $f(y,x,t)$와 같다. 이런 가정에 따라 위의 식을 아래 식으로 바꿔 쓸 수 있다.
$$
\frac{\partial f}{\partial y} \frac{dy}{dt} + \frac{\partial f}{\partial x} \frac{dy}{dt} + \frac{\partial f}{\partial t}=0
$$
$\frac{dy}{dt}$와 $\frac{dx}{dt}$는 $dt$동안 $y$와 $x$ 방향으로 이동한 양으로 모션 벡터 $\mathbf{v}=(v,u)$에 해당한다. 이 사실에 따라 아래 식이 성립한다.
$$
\frac{\partial f}{\partial y}v + \frac{\partial f}{\partial x}u + \frac{\partial f}{\partial t}=0
$$
위 식을 광류 조건식<sub>optical flow constraint equation</sub>이라 부르는데 대부분 광류 계산 알고리즘은 이 식을 이용한다. 
> [!example] 광류 조건식
> 계산 편의상 $\frac{\partial f}{\partial y}$와 $\frac{\partial f}{\partial x}$는 바로 이웃에 있는 화소와 명암 차이로 계산한다. 실제로는 소벨 연산자처럼 더 큰 연산자를 사용한다.
> $$
> \frac{\partial f}{\partial y}=f \quad y+1,x,t\ -f \quad y,x,t \quad
> \frac{\partial f}{\partial x}=f(y,x+1,t)-f(y,x,t) \quad
> \frac{\partial f}{\partial t}=f(y,x,t+1)f(y,x,t)
> $$
> 이 식을 광류 조건식에 대입하면 다음 식을 얻는다. 
> $$
> -v+2u+1=0
> $$
### Farneback 알고리즘으로 광류 추정
``` python
import numpy as np
import cv2 as cv
import sys

def draw_OpticalFlow(img, flow, step=16):
    for y in range(step//2, img.shape[0], step):
        for x in range(step//2, img.shape[1], step):
            dx, dy = flow[y,x].astype(np.int32)
            if (dx*dx + dy*dy):
                cv.line(img, (x,y), (x+dx, y+dy), (0,0,255),2)
            else:
                cv.line(img, (x,y), (x+dx, y+dy), (0,255,0),2)

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened(): sys.exit("카메라 연결 실패")

prev = None

while (1):
    ret, frame = cap.read()
    if not ret: sys('프레임 획득에 실패하여 루프를 나갑니다')

    if prev is None:
        prev = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        continue

    curr = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    draw_OpticalFlow(frame, flow)
    cv.imshow('Optical flow', frame)

    prev = curr
  
    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
```
![[Pasted image 20250703114545.png]]
### 희소 광류 측정을 이용한 KLT 추적
```python
import numpy as np
import cv2 as cv

cap = cv.VideoCapture('slow_traffic_small.mp4')  

feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15,15), maxLevel=2, criteria=(cv.TermCriteria_EPS|cv.TermCriteria_COUNT, 10, 0.03))

color = np.random.randint(0, 255, (100, 3))

ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

mask = np.zeros_like(old_frame)

while(1):
    ret, frame = cap.read()
    if not ret: break

    new_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    p1, match, err = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

    if p1 is not None:
        good_new = p1[match==1]
        good_old = p0[match==1]

    for i in range(len(good_new)):
        a,b = int(good_new[i][0]), int(good_new[i][1])
        c,d = int(good_old[i][0]), int(good_old[i][1])
        mask = cv.line(mask, (a,b), (c,d), color[i].tolist(),2)
        frame = cv.circle(frame, (a,b), 5, color[i].tolist(), -1)

    img = cv.add(frame, mask)
    cv.imshow('LTK tracker', img)
    cv.waitKey(30)

    old_gray = new_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv.destroyAllWindows()
```
![[Pasted image 20250703121150.png]]
## 1.2. 딥러닝 기반 광류 추정
FlowNet은 광류 추정에 컨볼루션 신경망을 적용한 최초 논문이다. 분할을 위해 사용했던 DeConvNet 같은 신경망을 광류에 적용하는 것이다. 그런데 광류에서는 입력이 $t$와 $t+1$ 순간의 영상 두 장이다. FlowNet은 컬러 영상 두 장을 쌓은 $384 \times 512 \times 6$ 텐서를 입력하는 단순한 신경망을 제안한다. 
딥러닝으로 인해 영상 분할 알고리즘의 성능이 높아지자 분할 결과를 활용하여 광류 추정의 성능을 개선하려는 시도가 여러 이루어진다. Sevilla-Lara는 입력 영상을 의미 분할한 다음 평평하고 움직이지 않는 부류, 스스로 움직이는 부류, 텍스처를 가지고 제자리에서 움직이는 부류로 구분하고 서로 다른 수식으로 광류를 추정한 다음 결합하는 방법을 제안했다. Bai는 자율주행 영상을 대상으로 알고리즘을 구상했는데, 의미 분할 대신 사례 분할을 적용했다. 
# 2. 추적
## 2.1. 문제의 이해
추적에서는 물체가 사라졌다 다시 나타나는 상황이 종종 발생한다. 이런 경우 끊긴 추적을 매칭하여 같은 물체로 이어주는 과정이 필요한데 이 과정을 재식별<sub>re-identification</sub>이라 부른다. 
물체 추적 문제는 추적할 물체의 개수에 따라 VOT<sub>Visual Object Tracking</sub>(시각 물체 추적)와 MOT<sub>Multiple Object Tracking</sub>(다중 물체 추적)로 나눈다. VOT는 첫 프레임에서 물체 하나를 지정하면 이후 프레임에서 그 물체를 추적하는 문제다. MOT는 영상에 있는 여러 물체를 찾아야 하는데 첫 프레임에서 물체 위치를 지정해주지 않고 추적할 물체의 부류는 지정해준다.
![[Pasted image 20250703135056.png]]
![[Pasted image 20250703135108.png]]
추적 문제는 배치 방식과 온라인 방식으로 나눌 수 있다. 배치 방식은 t 순간을 처리할 때 미래에 해당하는 $t+1,t+2, \cdots, T$ 순간의 프레임을 활용할 수 있다. 온라인 방식은 지난 순간의 프레임만 활용할 수 있다.
한 대의 카메라에서 입력된 비디오로 한정하는데, 문제는 다중 카메라 추적<sub>MCT; Multi-Camera Tracking</sub>이 있다. 다중 카메라 추적에서는 수초 내지 수분이 지난 후 다른 카메라에 나타나는 동일 물체를 찾아 이어주는 장기 재식별이 중요하다. 
## 2.2. MOT의 성능 척도
추적을 위한 성능 척도는 프레임 간의 연관성까지 고려해야 하므로 검출이나 분할보다 복잡하다. MOT를 위한 여러 가지 척도가 있는데 주로 아래 식의 MOTA<sub>MOT Accuracy</sub>를 사용한다. $GT_t$와 $FN_t,FP_t,IDSW_t$는 $t$ 순간에서 참값과 거짓 부겅, 거짓 긍정, 번호 교환 오류의 개수다.
$$
\text{MOTA}=1-\frac{\sum_{t=1,T}(FN_t+FP_t+IDSW_t)}{\sum_{t=1,T}GT_t}
$$
## 2.3. 딥러닝 기반 추적
### 일반적인 처리 절차
보통 딥러닝 기반 추적 알고리즘은 네 단계의 처리 과정을 거친다. 
1. 현재 프레임에 물체 검출 알고리즘을 적용해 박스를 찾는다. 
2. 박스에서 특징을 추출한다. 
3. 이전 프레임의 박스와 현재 프레임 박스의 거리 행렬을 구한다. 
4. 거리 행렬을 분석해 이전 프레임의 박스와 현재 프레임의 박스를 쌍으로 맺어 물체의 이동 궤적을 구성한다.
![[Pasted image 20250703154526.png]]
### SORT와 DeepSORT
SORT<sub>Simple Online and Real-time Tracking</sub>는 네 단계를 모두 단순한 알고리즘으로 처리하기 때문에 이해하기 쉽고 빠르다.
1. 물체 검출은 faster RCNN으로 해결한다. 현재 순간 $t$의 프레임에 faster RCNN을 적용하여 물체를 검출한다. 이렇게 구한 박스를 $B_{detection}$에 담는다.
2. 이전 순간 $t-1$에서 결정해 놓은 목표물의 위치 정보와 이동 이력 정보를 사용한다. 아래  식은 목표물의 정보를 표현한다. $x$와 $y$는 목표물의 중심 위치, $s$는 크기, $r$은 높이와 너비 비율이다.
$$
\mathbf{b}=(x,y,s,r,\dot{x},\dot{y},\dot{s})
$$
3. $B_{detection}$에 있는 박스와 $B_{predict}$에 있는 박스의 IOU를 계산하고 1에서 IoU를 빼서 거리로 변환한다. 이렇게 얻은 거리를 거리 행렬에 채운다.
4. 거리 행렬을 이용하여 매칭 쌍을 찾는다. 이때 헝가리안 알고리즘<sub>Hungarian Algorithm</sub>을 적용하여 최적의 매칭 쌍을 찾는다.
이렇게 네 단계를 마친 다음 후처리를 수행하고 다음 프레임으로 넘어간다. 가장 중요한 후처리는 $B_{predict}$에 있는 목표물의 식 정보를 갱신하는 일이다. 매칭이 일어난 목표물은 쌍이 된 박스 정보로 $x,y,s,r$을 대치한다. 이동 이력 정보에 해당하는 $\dot{x}, \dot{y}, \dot{s}$는 칼만 필터<sub>Kalman Filter</sub>를 사용하여 갱신한다. 
## 2.4. 프로그래밍 실습: SORT로 사람 추적
``` python
import numpy as np
import cv2 as cv
import sys

def construct_yolo_v3():
    f = open('coco_names.txt', 'r')
    class_names = [line.strip() for line in f.readlines()]

    model = cv.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    layer_names = model.getLayerNames()
    out_layers = [layer_names[i-1] for i in model.getUnconnectedOutLayers()]

    return model, out_layers, class_names

def yolo_detect(img, yolo_model, out_layers):
    height, width = img.shape[0], img.shape[1]
    test_img = cv.dnn.blobFromImage(img, 1.0/256, (448,448), (0,0,0), swapRB=True)
    yolo_model.setInput(test_img)
    output3 = yolo_model.forward(out_layers)

    box, conf, id = [], [], []
    for output in output3:
        for vec85 in output:
            scores = vec85[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                centerx, centery = int(vec85[0]*width), int(vec85[0]*height)
                w,h = int(vec85[2]*width), int(vec85[2]*height)
                x,y = int(centerx-w/2), int(centery-h/2)
                box.append([x,y,x+w,y+h])
                conf.append(float(confidence))
                id.append(class_id)

    ind = cv.dnn.NMSBoxes(box, conf, 0.5, 0.4)
    objects = [box[i] + [conf[i]] + [id[i]] for i in range(len(box)) if i in ind]
    return objects

model, out_layers, class_names = construct_yolo_v3()
colors = np.random.uniform(0, 255, size=(100,3))

from sort import Sort

sort = Sort()

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened(): sys.exit('카메라 연결 실패')

while True:
    ret, frame = cap.read()
    if not ret: sys.exit('프레임 획득에 실패하여 루프를 나갑니다.')

    res = yolo_detect(frame, model, out_layers)
    persons = [res[i] for i in range(len(res)) if res[i][5]==0]

    if len(persons) == 0:
        tracks = sort.update()
    else:
        tracks = sort.update(np.array(persons))
    for i in range(len(tracks)):
        x1,y1,x2,y2,track_id = tracks[i].astype(int)
        cv.rectangle(frame, (x1,y1), (x2,y2), colors[track_id], 2)
        cv.putText(frame, str(track_id), (x1+10, y1+40), cv.FONT_HERSHEY_PLAIN, 3, colors[track_id], 2)
    cv.imshow('Person tracking by SORT', frame)

    key=cv.waitKey(1)
    if key==ord('q'): break

cap.release()
cv.destroyAllWindows()
```
![[Pasted image 20250703161135.png]]
# 3. MediaPipe를 이용해 비디오에서 사람 인식
현재 MediaPipe 프레임워크는 16개의 솔루션을 제공하는데 그중 파이썬 인터페이스가 가능한 것은 얼굴 검출<sub>face detection</sub>, 얼굴 그물망<sub>face mesh</sub>, 손<sub>hands</sub>, 셀피 분할<sub>selfie segmentation</sub>, 자세<sub>pose</sub>, 몸 전체<sub>holistic</sub>, 오브젝트론<sub>objectron</sub>의 7가지다.
## 3.1. 얼굴 검출
### BlazeFace로 얼굴 검출
``` python
import cv2 as cv
import mediapipe as mp

img = cv.imread('BSDS_376001.jpg')

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
res = face_detection.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))

if not res.detections:
    print('얼굴 검출에 실패했습니다. 다시 시도하세요.')
else:
    for detection in res.detections:
        mp_drawing.draw_detection(img, detection)

    cv.imshow('Face detection by Mediapipe', img)

cv.waitKey()
cv.destroyAllWindows()
```
![[Pasted image 20250703162522.png]]
### 비디오에서 얼굴 검출
``` python
import cv2 as cv
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break
    
    res = face_detection.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    if res.detections:
        for detection in res.detections:
            mp_drawing.draw_detection(frame, detection)

    cv.imshow('MediaPipe Face Detection from Video', cv.flip(frame, 1))
    if cv.waitKey() == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
```
![[Pasted image 20250703163015.png]]
### 얼굴을 장식하는 증강 현실
증강 현실<sub>AR; Augmented Reality</sub>은 카메라로 입력된 실제 영상에 가상의 물체를 혼합함으로써 현실감을 증대시키는 기술이다. 
``` python
import cv2 as cv
import mediapipe as mp

dice = cv.imread('dice.png', cv.IMREAD_UNCHANGED)
dice = cv.resize(dice, dsize=(0,0), fx=0.1, fy=0.1)
w,h = dice.shape[1], dice.shape[0]

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break
    
    res = face_detection.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    if res.detections:
        for detection in res.detections:
            p = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE)
            x1, x2 = int(p.x*frame.shape[1]-w//2), int(p.x*frame.shape[1]+w//2)
            y1, y2 = int(p.y*frame.shape[0]-h//2), int(p.y*frame.shape[0]+h//2)
            if x1>0 and y1>0 and x2<frame.shape[1] and y2<frame.shape[0]:
                alpha = dice[:,:,3:]/255
                frame[y1:y2,x1:x2] = frame[y1:y2,x1:x2]*(1-alpha)+dice[:,:,:3]*alpha

    cv.imshow('MediaPipe Face AR', cv.flip(frame, 1))
    if cv.waitKey() == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
```
![[Pasted image 20250703164215.png]]
## 3.2. 얼굴 그물망 검출
### FaceMesh로 얼굴 그물망 검출
``` python
import cv2 as cv
import mediapipe as mp

mp_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

mesh = mp_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    res = mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    if res.multi_face_landmarks:
        for landmarks in res.multi_face_landmarks:
            mp_drawing.draw_landmarks(image=frame, landmark_list=landmarks, connections=mp_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(image=frame, landmark_list=landmarks, connections=mp_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(image=frame, landmark_list=landmarks, connections=mp_mesh.FACEMESH_IRISES, landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_styles.get_default_face_mesh_iris_connections_style())
            
    cv.imshow('MediaPipe Face Mesh', cv.flip(frame, 1))
    if cv.waitKey() == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
```
![[Pasted image 20250703170326.png]]
## 3.3. 손 랜드마크 검출
### 손 랜드마크 검출 실습
``` python
import cv2 as cv
import mediapipe as mp

mp_hand = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hand = mp_hand.Hands(max_num_hands=2, static_image_mode=False,
                     min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    res = hand.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    if res.multi_hand_landmarks:
        for landmarks in res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hand.HAND_CONNECTIONS,
                                      mp_styles.get_default_hand_landmarks_style(), mp_styles.get_default_hand_connections_style())
            
    cv.imshow('MediaPipe Hands', cv.flip(frame, 1))
    if cv.waitKey() == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
```
![[Pasted image 20250703170859.png]]
# 4. 자세 추정과 행동 분류
## 4.1. 자세 추정
자세 추정<sub>pose estimation</sub>은 정지 영상 또는 비디오를 분석해 전신에 있는 관절<sub>joint</sub> 위치를 알아내는 일이다. 자세 추정에서는 관절을 랜드마크 또는 키포인트라 부른다. 
### 인체 모델
자세를 표현하려면 인체 모델이 있어야 하는데, 골격 표현법<sub>skeleton representation</sub>과 부피 표현법<sub>volumetric representation</sub>을 주로 사용한다. 골격 표현은 랜드마크의 연결 관계를 그래프로 표현한다. 골격 표현은 랜드마크를 2차원 좌표와 3차원 좌표로 표현하는 경우로 구분할 수 있다. 반면에 부피 표현법은 기본적으로 3차원 좌표를 사용한다. 
![[Pasted image 20250704115548.png]]
### 정지 영상에서 자세 추정
인체의 구성요소를 검출하기 위해 HOG를 주로 사용했고 수작업 특징과 결합한 부품 모델은 깨끗한 배경을 가정한 상황에서나 동작하는 정도였다. 
딥러닝은 자세 추정에 새로운 길을 열었다. DeepPose는 딥러닝을 자세 추정에 처음 적용한 모델이다. DeepPose는 $220 \times 220 \times 3$ 영상을 입력으로 받아 컨볼루션층 5개를 통해 $13 \times 13 \times 192$ 특징 맵으로 변환하고 2개의 완전연결층을 거쳐 $2k$개의 실수를 추력한다. $k$는 랜드마크 개수고 랜드마크는 $(x,y)$ 좌표로 표현된다. 따라서 DeepPose는 fast RCNN과 마찬가지로 회귀를 수행하는 샘이다.
랜드마크를 검출하는 또 다른 방법으로 열지도 회귀<sub>heatmap regression</sub>가 있다. DeepPose는 $i$번째 랜드마크를 $(x_i,y_i)$로 표현하고 신경망이 좌푯값 자체를 예측하는 반면, 열지도에서는 $(x_i,y_i)$ 위치에 가우시안을 씌운 맵을 예측한다. 
![[Pasted image 20250704135127.png]]
다수의 사람을 대상으로 자세를 추정하는 접근 방법에는 하향식<sub>top-down</sub>과 상향식<sub>botton-up</sub>이 있다. 하향식 방법은 faster RCNN과 같은 모델을 사용해 사람을 검출한 다음에 사람 부분을 패치로 잘라내고 각각의 패치에 모델을 적용해 자세를 추정한다. 상향식 방법에서는 랜드마크를 모두 측정한 다음에 랜드마크를 결합하여 사람별로 자세를 추정한다.
![[Pasted image 20250704140048.png]]
### 비디오에서 자세 추정
비디오는 정지 영상, 즉 프레임이 시간 순서로 흐르는 구조다. 따라서 프레임을 독립적으로 취급해 자세 추정 모델을 적용하면 비디오의 자세 추정 결과를 얻을 수 있다. 하지만 각 프레임에서 발생한 위치 오류로 인해 자세가 심하게 흔들리는 현상이 발생할 수 있는데, 사람의 동작은 변한다는 사실을 잘 이용하면 일관성 있는 자세 추정이 가능하다.
이웃 프레임을 고려하는 접근 방법은 다양한데, 여기서는 광류를 사용하는 방법과 순환 신경망 방법을 소개한다. 광류를 컨볼루션 신경망에 적용해 자세 추정을 시도한 초창기 논문으로 Jain2014를 들 수 있다. 이 논문에서는 컨볼루션 신경망에 RGB 영상과 광류 맵을 결합한 텐서를 입력한다. Pfister는 이웃한 여러 장의 프레임에서 랜드마크 위치를 표현한 열지도를 예측한 다음에 광류를 이용해 이웃 프레임의 랜드마크 열지도를 현재 프레임에 맞게 변환하고 변환된 열지도를 결합하는 방법을 사용해 자세 추정의 정확률을 개선했다. 
자세 추적<sub>pose tracking</sub>은 하나의 프레임에서 여러 명을 구분하고 개개인의 자세를 추정한 다음에 이후 프레임에서 자세 단위로 사람을 추적해야 한다. 박스 단위로 사람을 표시했던 MOT 문제가 골격 단위로 표시한 사람을 추적하는 문제로 확장된 셈이다. 
![[Pasted image 20250704142804.png]]
## 4.2. BlazePose를 이용한 자세 추정
``` python
import cv2 as cv
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(static_image_mode=False, enable_segmentation=True,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    res = pose.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
    
    cv.imshow('Mediapipe pose', cv.flip(frame, 1))
    if cv.waitKey() == ord('q'):
        mp_drawing.plot_landmarks(res.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        break

cap.release()
cv.destroyAllWindows()
```
![[Pasted image 20250704143413.png]]
![[Pasted image 20250704143444.png]]
## 4.3. 행동 분류
사람은 다른 사람의 행동을 정확히 인식한다. 인식 결과를 이전에 취득한 사전 지식과 현재 상황과 결합해 상대의 의도를 추론한다. 사람이 행동 분류<sub>action classification</sub>를 넘어 행동 이해<sub>action understanding</sub>를 어떻게 수행하는지 밝히는 일은 심리학에서 중요한 연구 주제다. 
Kinetics 데이터셋과 HAA500 데이터셋은 행동 부류에 널리 사용된다. Kinetics의 행동 부류는 '자전거 타기'와 같은 person 행동, '악수하기'와 같은 person-person 행동, '선물 풀어보기'와 같은 person-object 행동으로 구분되어 있으며 총 700 부류가 있다. 
HAA는 스포츠 행동이 212개, 악기 연주가 51개, 게임과 취미 활동이 82개, 일상 생활이 155개다. 더 이상 쪼갤 수 없는 원자 수준의 행동 부류<sub>atomic action class</sub>를 레이블링했다는 점에서 독특하다. 
![[Pasted image 20250704144105.png]]
![[Pasted image 20250704144138.png]]