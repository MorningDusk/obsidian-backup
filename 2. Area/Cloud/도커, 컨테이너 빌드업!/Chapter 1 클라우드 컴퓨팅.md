---
aliases:
  - "Chapter 1: 클라우드 컴퓨팅"
tags:
  - cloud
  - container
  - docker
  - kubernetes
  - devops
date: 2025-02-01
---
# 1. 클라우드 컴퓨팅 개요
## 1.1. 클라우드 컴퓨팅이란?
> [!info] 클라우드 컴퓨팅<sub>cloud computing</sub>
> 인터넷 기술을 이용해서 다수의 사용자에게 하나의 서비스로서 방대한 IT 능력을 제공하는 컴퓨팅 방식

시작은 유틸리티 컴퓨팅<sub>utility computing</sub>으로 기업 내의 IT 부서나 외부의 서비스 제공자<sub>provider</sub>가 고객에게 컴퓨팅에 사용되는 여러 자원과 기반 시설 등의 관리를 제공하는 형태이며 정액제 대신 사용량에 따라 요금을 부과하는 종량제 방식이다. 이 기술의 주요 기능으로는 클러스터<sub>cluster</sub>, 가상화<sub>virtualization</sub>, 분할<sub>partitioning</sub>, 프로비저닝<sub>provisioning</sub>, 자율 컴퓨팅<sub>autonomic computing</sub>, 그리드 컴퓨팅<sub>grid computing</sub> 등이 있다.

| 기술 기반    | 설명                                                                                                                                    |
| -------- | :------------------------------------------------------------------------------------------------------------------------------------ |
| 그리드 컴퓨팅  | 가상 네트워크를 이용하여 분산된 컴퓨팅 자원을 공유하도록 하는 기술 방식                                                                                              |
| 유틸리티 컴퓨팅 | 다양한 컴퓨팅 자원에 대한 사용량에 따라 요금을 부과하는 종량제 방식의 기술 기반으로, 필요할 때 가져다 쓴다는 온-디맨드(on-demand) 컴퓨팅 방식(기업 중심의 서비스)                                    |
| 클라우드 컴퓨팅 | 기술적으로는 그리드 컴퓨팅을 따르고, 비용적으로는 유틸리티 컴퓨팅을 혼합한 포괄적인 패러다임이다. 약간의 차이점은 다음과 같다.<br>- 기업과 개인이 모두 사용 가능한 서비스<br>- 클라우드 서비스를 제공하는 사업자의 컴퓨팅 자원 이용 |
### 특징
- **주문형 셀프서비스**<sub>on-demand self-service</sub>: 고객이 IT 서비스 제공자의 개입 없이 원하는 시점에 바로 서비스를 사용할 수 있다.
- **광대역 네트워크 접근**<sub>broad network access</sub>: 각 클라우드 서비스 업체<sub>Cloud Service Provider; CSP</sub>가 제공하는 광대역 네트워크를 이용하여 다양한 클라이언트 플랫폼이 빠르게 접속할 수 있다.
- **신속한 탄력성과 확장성**<sub>rapid elasticity and scalability</sub>: 자동 조정<sub>auto-scaling</sub> 기능을 통해 몇 분 안에 신속한 확장과 축소를 조정할 수 있다.
- **자원이 공동관리**<sub>resource pooling</sub>: 물리적 및 가상화된 자원을 풀<sub>pool</sub>로 관리하며, 탄력적으로 사용자 요구에 따라 동적으로 할당 또는 재할당된다.
- **측정 가능한 서비스**<sub>measured service</sub>: 자원 사용량이 실시간으로 수집되는 요금산정<sub>metering</sub> 기능을 통해 비용이 발생한다.
## 1.2. 클라우드 컴퓨팅 구조
클라우드 컴퓨팅 구조<sub>cloud computing architecture</sub>는 최하위 계층으로 자원<sub>resources</sub> 활용과 관련된 물리적 시스템 계층, 가상화 계층, 프로비저닝 계층이 있고, 클라우드 서비스와 관련된 클라우드 서비스 관리 계층, 클라우드 서비스 계층으로 구분한다. 그 위로는 사용자와 관련된 클라우드 접근<sub>access</sub> 계층과 사용자 역할에 연결성 구분을 설정할 수 있다.
![[Pasted image 20250718095704.png]]
클라우드 컴퓨팅의 물리적 시스템 계층은 여러 형태의 서버 계열을 활용하여 서버에 탑재된 수평적으로 확장 가능한<sub>scale out</sub> 스토리지 및 네트워크 등의 물리적 요소를 의미한다. 이를 기반으로 서버, 스토리지, 네트워크 가상화는 클라우드의 주요 이점 중 하나인 민첩성<sub>agility</sub>을 제공하고, 이를 통해 IT 서비스 공급자는 클라우드 서버 프로비저닝 또는 프로비저닝 해제를 신속히 수행하여 서비스 사용자의 요구를 충족하게 된다.
이렇게 구성된 클라우드 구성요소가 **서비스로서**<sub>as a service</sub> 제공되는 확장 가능한 컴퓨팅 자원을 사용한 양에 따라 비용을 지불하며, 클라우드 환경에 모든 자원에 인터넷과 모바일을 통해 언제든 접근할 수 있다. 
사용자는 주어진 역할에 따라 다양한 웹 애플리케이션 프로그램 인터페이스<sub>Application Programming Interface; API</sub>를 통해 클라우드 서비스를 호출할 수 있다.
## 1.3. 클라우드 컴퓨팅 제공 방식과 클라우드 서비스 종류
### 클라우드 컴퓨팅 제공 방식
#### 1. 온프레미스
클라우드 개념이 도입되기 전에는 대부분의 기업이 자체 데이터 및 솔루션 등을 저장하기 위해 자사에 데이터 센터를 구축하여 IT 서비스를 수행하였는데, 이를 온프레미스<sub>on-premise</sub>라고 한다. 하지만 이 방식은 모든 자원에 대한 초기 투자 비용과 탄력적이지 않은 제한된 용량으로 인해 지속적 관리 비용이 증가하는 단점이 있다.
온프레미스 방식으로 설계 시 자원량은 가급적 최대 사용량을 근거로 하고, 네트워크 트래픽 또한 최대 순간 트래픽<sub>peak traffic</sub>을 가정하기 때문에 고사양의 설계를 하게 된다. 클라우드 도입은 비용 측면보다는 서비스의 가용성과 품질을 높여서 기업의 이익을 높일 수 있다.
클라우드 접근 방식은 사용한 만큼 지불하는 정산 방식을 통해 필요에 따라 민첩하고 탄력적<sub>elastic</sub>으로 사용할 수 있다.
![[Pasted image 20250718103644.png]]
![[Pasted image 20250718103714.png]]
#### 2. 퍼블릭 클라우드
퍼블릭 클라우드<sub>public cloud</sub> 방식은 인터넷을 통해 다수의 사용자에게 서버 및 스토리지 등의 클라우드 자원을 AWS, GCP, Azure와 같은 클라우드 서비스 공급자로부터 제공받는 방식이다. 사용자 및 그룹 단위로 권한 관리를 통해 서비스 격리를 하기 때문에 사용자 간의 간섭이 발생하지 않는다.
![퍼블릭 클라우드와 프라이빗 클라우드 비교](https://cf-assets.www.cloudflare.com/slt3lc6tev37/2jBaVWKgbOUNLDNw7QJYPh/563316b4290e2919f7510ae59a3ae3ca/public-cloud-vs-private-cloud.svg)
#### 3. 프라이빗 클라우드
프라이빗 클라우드<sub>private cloud</sub> 방식은 제한된 네트워크에서 특정 사용자나 기업만을 대상으로 하는 클라우드 서비스 방식으로, 클라우드 자원과 데이터는 기업 내부에 저장되고 유지 관리에 대한 책임 또한 기업이 갖는다. 인터넷이 아닌 인트라넷<sub>intranet</sub> 방식으로 서비스에 접근하게 되므로 보안성이 높다.
#### 4. 하이브리드 클라우드
하이브리드 클라우드<sub>hybrid cloud</sub> 방식은 퍼블릭 클라우드와 프라이빗 클라우드 네트워크를 통해 결합하여 두 가지 서비스의 장점을 활용할 수 있도록 만든 클라우드 서비스 방식이다. 서로 다른 클라우드 간에 데이터와 애플리케이션 공유 및 이동이 유연하게 처리될 수 있고, 용도에 맞는 서비스 구현에 유리하다.
![[Pasted image 20250718104603.png]]
### 클라우드 서비스의 종류
클라우드 서비스란 언제 어디서나 별도의 소프트웨어 등을 설치하지 않고 인터넷 접속을 통해 저장해 놓은 데이터에 접근하여 사용할 수 있는 서비스를 말한다.
![[Pasted image 20250718105223.png]]
#### 1. 서비스로서의 인프라스트럭처<sub>Infrastructure as a Service; IaaS</sub>
서버, 스토리지, 네트워크와 같은 인프라 하드웨어 자원을 가상화하여 사용자 요구에 따라 인프라 자원을 사용할 수 있게 제공하는 클라우드 서비스 방식이다. IaaS는 자동화되고 신속한 확장성을 갖고 있는 IT 인프라를 의미하며, 비용은 사용량에 따라 지불하는 방식이다. 대표적으로 국내의 KT, LG U+ 등의 서비스와 외국의 AWS, GCP, Azure, 오라클 클라우드 플랫폼 등에서 IaaS를 제공한다.
#### 2. 서비스로서의 플랫폼<sub>Platform as a Service; PaaS</sub>
서비스 개발자가 애플리케이션 개발, 실행, 관리 등을 할 수 있도록 안정적인 플랫폼 또는 프레임워크를 제공하는 클라우드 서비스 방식이다. 따라서 개발자가 서비스 개발을 위한 복잡한 설치 과정이나 환경 설정을 하지 않고 완성된 개발 소스만 제공하면 바로 서비스를 올릴 수 있는 플랫폼 서비스를 말한다. 대표적으로 네이버 클라우드 플랫폼과 IaaS를 제공하는 AWS, GCP, Azure 등의 대표적인 클라우드 공급자가 있다.
#### 3. 서비스로서의 소프트웨어<sub>Software as a Service; SaaS</sub>
소프트웨어 사용자가 자신의 컴퓨터에 소프트웨어를 설치하지 않고 인터넷을 통해 클라우드에 접속하여 클라우드 기반 소프트웨어의 기능을 사용할 수 있게 해주는 클라우드 서비스 방식이다. 소프트웨어 버전업, 패치, 재설치 등의 작업 없이도 해당 기능을 사용할 수 있다. 대표적으로 이메일, CRM<sub>Customer Relationship Management</sub> 소프트웨어, 구글<sub>Google</sub> 앱 서비스 등이 있다.
# 2. 컨테이너 기술과 도커
## 2.1. 가상머신과 컨테이너
![[Pasted image 20250718110326.png]]
클라우드 컴퓨팅에서 가상화는 하드웨어 기능을 시뮬레이션하여 애플리케이션 서버, 스토리지, 네트워크와 같은 유용한 IT 서비스를 생성하는 소프트웨어 아키텍처 기술이다. 
최근 사용하고 있는 가상화는 **하이퍼바이저**를 이용한 가상머신과 **컨테이너**를 이용한 도커 방식이다. 가상머신은 호스트 운영체제 위에 가상화 소프트웨어를 이용하여 여러 개의 게스트 OS를 구동하는 방식이다. 하이퍼바이저<sub>hypervisor</sub>는 가상머신<sub>Virtual Machine; VM</sub>을 생성하고 실행하는 역할과 가상화된 하드웨어와 각각의 가상머신을 모니터링하는 중간 관리자다. 
컨테이너를 이용한 가상화는 리눅스 기반의 물리적 공간 격리가 아닌 프로세스 격리를 통해 경량의 이미지를 실행하고 서비스할 수 있는 **컨테이너**<sub>container</sub> 기술이다. 클라우드 서비스의 컨테이너는 애플리케이션을 구동하는 환경을 격리한 공간을 의미한다.
도커 엔진이 차용하고 있는 컨테이너 기술은 본래 리눅스 자체 기술인 `chroot`, 네임스페이스, `cgroup`을 조합한 리눅스 컨테이너<sub>LinuX Container; LXC</sub>에서 출발한다.
- `chroot`<sub>change root</sub>: 특정 디렉터리를 최상위 디렉터리 `root`로 인식하게끔 설정하는 리눅스 명령
- **네임스페이스**<sub>namespace</sub>: 프로세스 자원을 관리하는 기능으로, `mnt`, `pid`, `net`, `ipc`, `user` 등의 자원을 그룹화하여 할당하는 기술
- `cgroup`<sub>control group</sub>: CPU, 메모리, 디스크 I/O, 네트워크 등의 자원 사용량 제어를 통해 특정 애플리케이션의 과도한 자원 사용을 제한하는 기능
컨테이너 가상화는 프로세스 가상화다. 컨테이너 엔진인 도커와 오케스트레이션<sub>orchestration</sub> 도구인 쿠버네티스는 호스트 운영체제의 커널을 공유하고 그 위에 실행 파일 및 라이브러리<sub>Bins/Libs</sub>, 기타 구성 파일 등을 이미지로 빌드<sub>image build</sub>하여 패키지로 배포<sub>image run</sub>하는 방식이다.
### 장점
- 하이퍼바이저와 게스트 OS가 없기 때문에 가볍다.
- 경량이기 때문에 만들어진 이미지 복제, 이관, 배포가 쉽다.
- 게스트 OS를 부팅하지 않기 때문에 애플리케이션 시작 시간이 빠르다.
- 가상머신보다 경량이므로 더 많은 애플리케이션을 실행할 수 있다.
## 2.2. 도커
컨테이너는 코드와 모든 종속성을 패키지화하는 표준 소프트웨어 단위로, 애플리케이션이 한 컴퓨팅 환경에서 다른 컴퓨팅 환경으로 빠르고 안정적으로 실행되도록 한다. 
도커 컨테이너 이미지는 도커 허브<sub>docker hub</sub>로부터 내려받거나<sub>pull</sub> Dockerfile을 통해 생성<sub>build</sub>하여 도커 엔진을 이용해 실행하면 컨테이너 서비스가 된다.
### 기능
- **LXC를 이용한 컨테이너 구동**: `containerd`는 리눅스 및 윈도우용 데몬<sub>daemon</sub>으로, 이미지 전송 및 스토리지에서 컨테이너 실행 및 감독, 네트워크 연결까지 호스트 시스템 전체 컨테이너의 라이프사이클을 관리한다.
- **통합 Buildkt**: 빌드킷<sub>buildkit</sub>은 도커 파일의 설정 정보를 이용하여 도커 이미지를 빌드하는 오픈 소스 도구이며, 빠르고 정확하게 여러 가지 아키텍처 향상 기능을 제공한다.
- **도커 CLI 기반**: 도커 명령을 수행하는 기본적인 방법은 CLI<sub>Command Line Interface</sub>로 제공한다. 
도커를 사용하기 위해서는 우선 컨테이너, 이미지를 다룰 수 있는 **도커 엔진**이 필요하고, 다음으로 이미지 업로드<sub>push</sub>/다운로드<sub>pull</sub>을 통해 컨테이너 서비스에 필요한 이미지 배포를 지원하는 **도커 허브**에서 서비스를 제공받아야 한다.
``` docker
user@docker-host:~$ docker pull ubuntu:18.04

# push를 하려면 사전에 도커 허브에 가입이 되어 있어야 하고, docker login을 통해 접속한 후 수행할 수 있다.
user@docker-host:~$ docker push dbgurum/test_image:1.0
```
위의 예시는 도커 허브로부터 Ubuntu 18.04 버전을 호스트로 다운로드를 수행하고, `test_image`라는 새로운 이미지를 생성한 경우, 이 이미지를 `dbgurum`에 `test_image:1.0`이라는 태그<sub>tag</sub>로 업로드를 수행한 것이다.
도커 컨테이너 기술은 PaaS 서비스를 가능하게 하는 소프트웨어 개발환경을 제공하는 것이다. 다만, 컨테이너 서비스에 대한 자동화된 관리, 트래픽 라우팅, 로드 밸런싱 등을 쉽게 하려면 오케스트레이션 기능이 추가로 요구된다.
![[Pasted image 20250718114641.png]]
``` docker
# docker hub으로부터 node 도커 이미지를 다운로드
# 이미지명 뒤에 버전을 지정하지 않으면 자동으로 latest 버전으로 지정
user@docker-host:~$ docker pull node

# 다운로드한 이미지를 실행하여 컨테이너화함
user@docker-host:~$ docker run -d -it --name=nodejs_test node:latest

# 컨테이너 실행 확인
user@docker-host:~$ docker ps

# 미리 작성해둔 소스 코드를 컨테이너 내부로 복사 (nodejs_test.js)
# 내부에서 작성해도 되지만 별도 편집 프로그램 설치가 필요하여 복사함
user@docker-host:~$ docker cp nodejs_test.js nodejs_test:/nodejs_test.js

# 실행 중인 npm이 설치된 nodejs_test 컨테이너에 bash셀로 접속
user@docker-host:~$ docker exec -it nodejs_test /bin/bash

# 전달된 소스 코드 확인
root@579bcaa0d4d0:/# ls
bin    boot    dev    etc    home    lib    lib64    media    mnt    nodejs_test.js

# 설치된 npm 모듈 버전 확인
root@579bcaa0d4d0:/# node -v
v13.5.0

# node 프로그램을 이용하여 샘플 소스 코드 테스트 수행
root@579bcaa0d4d0:/# node nodejs_test.js
```
### 구성 요소
- **Docker Engine**: 도커를 이용한 애플리케이션 실행 환경 제공을 위한 핵심 요소
- **Docker Hub**: 전 세계 도커 사용자들과 함께 도커 컨테이너 이미지를 공유하는 클라우드 서비스
- **Docker-compose**: 의존성 있는 독립된 컨테이너에 대한 구성 정보를 야믈<sub>YAML</sub> 코드로 작성하여 일원환된 애플리케이션 관리를 가능하게 하는 도구
- **Docker Kitematic**: 컨테이너를 이용한 작업을 수행할 수 있는 **GUI**<sub>Graphic User Interface</sub> 제공
- **Docker Registry**: 도커 허브 사이트에 공개된<sub>public</sub> 레지스트리라고 보면 된다
- **Docker Machine**: 가상머신 프로그램 및 AWS EC2, MS Azure 환경에 도커 실행 환경을 생성하기 위한 도구
- **Docker Swarm**: 여러 도커 호스트를 클러스터로 구축하여 관리할 수 있는 도커 오케스트레이션 도구
## 2.3. 도커 맛보기: PWD
### 환경
- 가상 IP 주소와 함께 자원에 대한 리소스 현황을 볼 수 있다.
- SSH<sub>Secure Shell</sub>로 접속할 수 있는 주소를 지원한다.
- `[OPEN PORT]`는 도커 컨테이너를 외부로 노출 시 바인드되는 포트 번호를 보여준다.
### CentOS 7 버전이 필요한 가정
``` docker
# 제공된 환경에 이미지와 컨테이너가 있는지 조회해 본다. 아무것도 없을 것이다.
$ docker image ls
$ docker ps

# 도커 허브 사이트로부터 CentOS 7 버전의 도커 컨테이너 이미지를 다운로드한다.
$ docker pull centos:7
7: Pulling from library/centos
ab5ef0e58194: Pull complete
Digest: sha256:4a701376d03f6b39b8c2a8f4a8e499441b0d567f9ab9d58e4991de4472fb813c
Status: Download newer image for cenos:7
docker.io/library/centos:7

# 다운로드한 이미지를 확인해본다.
$ docker image ls
REPOSITORY    TAG    IMAGE ID        CREATED        SIZE
centos        7      5e35e350aded    4 months ago   203MB

# 이미지가 가지고 있는 CentOS 7 능력을 구경하기 위해 컨테이너를 시작한다.
$ docker run -it --name=centos7_test centos:7 /bin/bash

# 프롬프트(prompt)가 변경, CentOS 7 컨테이너 환경으로 들어온 것을 확인해본다.
[root@9b6c19fbc397  /]# cat /etc/os-release
```
![[Pasted image 20250718121837.png]]
`PWD Terminal`에 `docker run -dp 8080:docker/getting-started:pwd` 입력
![[Pasted image 20250718122043.png]]
상단 옆에 포트 번호 80이 생긴 것을 확인할 수 있다.
![[Pasted image 20250718122111.png]]
`80`을 클릭해 보면 제공된 SSH 주소에 해당 컨테이너 내부에 저장된 웹 화면을 80번 포트로 연결해서 보여주는 것을 확인할 수 있다.
![[Pasted image 20250718122209.png]]
# 3. 쿠버네티스
쿠버네티스는 대규모 클러스터 환경의 수많은 컨테이너를 쉽고 빠르게 확장, 배포, 관리하는 작업을 자동화해 주는 오픈 소스 플랫폼이다.
![kubernetes-manahed-service-circle-image|750](https://www.xenonstack.com/hubfs/Imported%20sitepage%20images/circle-image-2.svg)
도커 스윔, 아파치 메소스, AWS의 ECS<sub>Elastic Container Service</sub> 등 많은 오케스트레이션 도구가 나왔고, 그중 하나가 쿠버네티스다.
## 유용한 이유
- 온프레미스 환경에서 수행하는 서버 업그레이드, 패치, 백업 등의 작업을 자동화하여 인프라 관리보다는 서비스 관리에 집중할 수 있다.
- 서비스 사용자는 애플리케이션이 24/7/365 지속되기를 원한다. 컨테이너에 장애 발생 시 자가 회복<sub>self-healing</sub> 기능을 통해 곧바로 복제<sub>replica</sub> 컨테이너를 생성하여 서비스를 지속할 수 있다.
- 컨테이너화를 통해 소프트웨어를 패키지화하면 점진적 업데이트<sub>rolling update</sub>를 통해 다운타임 없이 쉽고 빠르게 릴리스 및 업데이트할 수 있다.
그 외에도 스토리지 오케스트레이션, 자동화된 빈 패킹<sub>bin packing</sub> 등 분산 시스템을 탄력적으로 운영하기 위한 프레임워크를 제공한다.
# 4. 데브옵스
데브옵스<sub>DevOps</sub>는 단순하게 업무, 부서, 방법론, 기술 형태로 제한하지 않는다. 업무적으로 상층관계<sub>trade-off</sub>에 있는 모든 형태에 적용할 수 있다. 조직 내의 모든 업무자 간의 소통과 협력은 효율성을 높이고 서비스 품질 향상을 통한 기업의 성장을 가져올 수 있다는 것이 데브옵스의 기본 철학이자 하나의 문화다.