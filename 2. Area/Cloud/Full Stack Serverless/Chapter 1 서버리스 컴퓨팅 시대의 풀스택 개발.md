---
date: 2025-07-18
tags:
  - react
  - aws
  - cloud
---
# 1. 현대적인 서버리스 철학
**서버리스**<sub>serverless</sub>라는 용어는 일반적으로 FaaS와 연관된다. 
> [!info] 서버리스
> FaaS로 대변되는 클라우드 기능들은 서버리스 컴퓨팅의 핵심이지만 클라우드 플랫폼은 **서비스형 백엔드**<sub>Backend as a Service</sub>(BaaS)와 같은 특정 애플리케이션 요구 사항을 충족하는 전문 서버리스 프레임워크도 제공한다. 간단히 말해 '**서버리스 컴퓨팅 = FaaS + BaaS**'이다.

**BaaS**는 일반적으로 데이터베이스, 인증 서비스, 인공지능 서비스와 같은 관리형 서비스를 의미한다. 
## 1.1. 서버리스 애플리케이션의 특성
### 운영 책임 감소
서버리스 아키텍처를 사용하면 일반적으로 클라우드 공급 업체 또는 서드 파티<sub>third-party</sub>로 더 많은 운영 책임을 이전할 수 있다.
FaaS를 구현하기로 했을 때 걱정해야 하는 것은 실행되는 코드 뿐이다. 모든 서버의 패치 적용, 업데이트, 유지 관리, 업그레이드는 더는 사용자 책임이 아니다.
### 관리형 서비스의 중용
관리형 서비스는 일반적으로 정의된 기능들의 모음을 책임지고 제공한다. 이러한 서비스는 서버가 원활하게 확장되며, 서버 운용이 필요하지 않고, 가동 시간을 관리할 필요도 없다.
## 1.2. 서버리스 아키텍처의 장점
### 확장성
애플리케이션을 구축할 때 애플리케이션의 사용량이 많아지고 사용자가 빠르게 증가하는 경우에 발생할 수 있는 문제들은 클라우드 공급 업체에서 처리해주므로 걱정할 필요가 없다.
애플리케이션은 사용자의 요청을 처리하기 위한 코드를 실행하면서 자동으로 확장된다. 서버리스 함수에서 코드는 병렬로 실행되며 각 트리거<sub>trigger</sub>를 개별적으로 처리한다.
### 비용
서버리스 기술을 이용하면 사용한 리소스에 대해서만 비용을 지불하면 된다. FaaS를 사용하면 기능에 대한 요청 횟수, 기능의 코드가 실행되는 데 걸리는 시간, 각 기능에 할당된 메모리를 기반으로 비용이 청구된다.
이를 통해 초기 인프라 비용 없이도 기능과 애플리케이션을 구축할 수 있다.
### 개발 속도
서버리스 기술을 사용하면 데이터베이스, 인증, 스토리지, API 같이 대부분의 애플리케이션에 필요한 기능들을 이용할 수 있기 때문에 더 빠르게 핵심 기능과 비즈니스 로직을 작성하는 데 집중할 수 있다.
### 실험
새로운 기능을 추가할 때 투자자본수익률<sub>return on investment</sub>(ROI)를 기준으로 기능 구축, 관련된 시간, 자본을 평가한다. 서버리스 기술을 사용하면 새로운 것을 시도할 때 발생하는 위험이 줄어들면서 과거에는 시도해보지 못했을 아이디어를 자유롭게 실험할 수 있다.
**A/B 테스트**는 여러 버전의 애플리케이션을 비교하여 가장 효과적인 버전을 확인하는 방법이다. 개발 속도 증가 덕분에 서버리스 애플리케이션은 더 빠르고 쉽게 A/B 테스트를 활용하여 다양한 아이디어를 실험할 수 있다.
### 보안과 안정성
서비스 제공 업체는 다운타임<sub>downtime</sub>을 가능한 한 최소한으로 만들려고 노력한다. 또한 구축, 배포, 유지 보수, 안정적인 서비스를 위해 할 수 있는 모든 것을 한다.
### 적은 코드
유지 관리에 필요한 코드의 양을 제한하거나 심지어 코드 없이도 기능을 출시하는 방법을 찾을 때 응용 프로그램의 전반적인 복합성은 줄어든다. 복잡성이 줄어들면 버그가 줄고, 새로운 엔지니어가 더 쉽게 적응할 수 있으며, 새로운 기능을 추가하고 유지하는 데 필요한 부하가 전반적으로 줄어든다.
## 1.3. 서버리스의 다양한 구현
### 서버리스 프레임워크
서버리스 프레임워크인 JAWS<sub>Javascript Amazon Web Services</sub>는 Node.js 기반의 무료 오픈 소스 프레임워크다. 
서버리스 프레임워크는 `serverless.yml` 같은 설정 파일, 명령줄 인터페이스<sub>command-line interface</sub>(CLI), 기능 코드의 조합으로 서버리스 함수 및 다른 AWS 서비스를 로컬 환경에서 클라우드에 배포하려는 사람들에게 좋은 사용자 경험을 제공한다.
### AWS Serverless Application Model
AWS Serverless Application Model을 이용하면 YAML 파일에 서버리스 애플리케이션에서 필요한 API Gateway의 정의, AWS Lambda 함수, Amazon DynamoDB 테이블 정의를 작성하여 서버리스 애플리케이션을 구축할 수 있다. 또한 YAML 설정, 함수 코드의 조합과 CLI를 이용하여 서버리스 애플리케이션의 생성, 관리, 배포할 수 있다.
SAM의 장점은 AWS의 거의 모든 서비스를 사용할 수 있는 AWS CloudFormation의 강력한 확장 기능을 사용할 수 있다는 것이다. 
### Amplify 프레임워크
Amplify 프레임워크는 CLI, 클라이언트 라이브러리, 툴체인<sub>toolchain</sub>, 웹 호스팅 플랫폼의 조합이다. Amplify의 목적은 개발자에게 풀스택 웹 애플리케이션과 모바일 애플리케이션을 클라우드상에서 쉽게 구축하고 배포하는 방법을 제공하는 것이다. 또한 서버리스 함수와 인증, 그래프QL<sub>GraphQL</sub> API, 머신러닝, 스토리지, 분석, 푸시 알림 등의 기능을 지원한다.
인증 서비스를 Amazon Cognito보다는 **auth**라고 하며 프레임워크는 내부적으로만 Amazon Cognito를 사용한다.
#### Amplify 프레임워크 네 가지 구성 요소의 역할
- CLI를 이용하여 명령줄에서 클라우드 서비스를 만들고, 구성하고, 배포할 수 있다.
- 클라이언트 라이브러리를 사용하면 웹 또는 모바일 애플리케이션에서 클라우드 서비스에 연결하고 상호작용할 수 있다.
- 툴체인은 코드 생성과 서버리스 함수 보일러플레이트<sub>boilerplate</sub> 같은 것을 통해 신속한 개발이 가능하도록 한다.
- 호스팅 플랫폼을 통해 원자 배포<sub>atomic deployment</sub>, 지속적 통합, 지속적 배포, 사용자 지정 도메인 등이 포함된 라이브 도메인에 배포할 수 있다.
### 다른 선택지
대부분 도구는 AWS 또는 다른 클라우드 공급 업체에서 사용할 수 있는 다른 서비스들을 다양하게 제공하지 않는다.
# 2. AWS 소개
## 2.1. AWS란
AWS는 개발자에게 온디맨드<sub>on-demand</sub> 클라우드 컴퓨팅 플랫폼을 최초로 제공한 회사다. Amazon SQS<sub>Amazon Simple Queue Service</sub>라는 단일 서비스로 처음 출시됬고, Amazon SQS, Amazon S3, Amazon EC2라는 총 3가지 서비스로 재출시되었다.
## 2.2. AWS의 풀스택 서버리스
**풀스택 서버리스**는 확장 가능한 애플리케이션을 최대한 빨리 구축한다는 목표를 달성하기 위해 개발자들에게 필요한 모든 것을 제공한다.
## 2.3. Amplify CLI
**Amplify CLI**를 사용하여 프런트엔드 환경에서 직접 클라우드 서비스들을 생성, 설정, 수정, 삭제할 수 있다.
CLI는 AWS Console 및 CloudFormation과 같은 도구에서 사용되는 서비스 이름 접근 방식 대신 카테고리 이름 접근 방식을 사용한다. CLI에서는 이러한 서비스를 생성, 설정하는 데 서비스 이름을 사용하는 대신 **storage**, **auth**, **analytics** 같은 이름을 사용하여 이를 통해 서비스가 어떤 일을 하는지 파악할 수 있도록 한다.
CLI에는 프런트엔드 환경을 벗어나지 않고도 서비스를 생성, 수정, 설정, 삭제할 수 있는 명령어들이 있다. 또한 CLI를 사용하여 운영 환경에 영향을 주지 않고도 새로운 기능을 출시하기 위한 새로운 환경을 배포할 수 있다.
### Amplify 클라이언트
Amplify 클라이언트는 AWS 서비스들과 상호작용해야 하는 자바스크립트 애플리케이션에서, 사용하기 쉬운 API를 제공하기 위해 특별히 제작된 라이브러리다. 또한 Amplify는 리액트 네이티브<sub>React Native</sub>, 네이티브 iOS, 네이티브 안드로이드용 클라이언트 SDK도 제공하고 있다.
Amplify 클라이언트의 방식은 더욱 높은 수준의 추상화를 제공하고 모범 사례를 활용한다. 이를 바탕으로 선언적이고 사용하기 쉬운 API를 제공하며 백엔드와 상호작용을 완전히 할 수 있게 제어한다. 또한 웹소켓<sub>WebSocket</sub> 및 그래프 QL 서브스크립션<sub>subscription</sub> 지원과 같은 기능을 가진 클라이언트를 염두에 두고 구축되었다.
Amplify는 리액트, 리액트 네이티브, 뷰<sub>vue</sub>, 앵귤러<sub>angular</sub>, 아이오닉<sub>ionic</sub>, 네이티브 안드로이드, 네이티브 iOS를 포함한 인기 있는 프런트엔드, 모바일 프레임워크에 UI 컴포넌트들을 제공한다.
Amplify 프레임워크는 AWS 서비스 전체를 지원하지는 않지만 서버리스 커테고리에 속하는 거의 대부분의 서비스를 지원한다. Amplify가 EC2와 상호작용을 지원하는 것은 적합하지 않지만 REST<sub>representational state transfer</sub>와 그래프QL API 사용에 대한 지원을 제공하는 것은 합리적이다.
### AWS AppSync
AWS AppSync는 그래프QL을 사용하여 애플리케이션이 모든 데이터 소스, REST API 또는 마이크로서비스와 쉽게 상호작용할 수 있도록 하는 관리형 **API 계층**이다.
**마이크로서비스 아키텍처**<sub>microservice architecture</sub>는 모듈식 구성 요소나 서비스들의 조합을 사용하여 구축된 대형 애플리케이션에 사용되는 일반적인 용어다. 
그래프QL은 **쿼리**<sub>query</sub>(조회), **뮤테이션**<sub>mutation</sub>(생성, 수정), **서브스크립션**(실시간 데이터)의 세 가지 작업 형태로 API와 상호작용하기 위한 일관된 규격을 정의했다. 이러한 작업들은 그래프QL **스키마**<sub>schema</sub>의 일부로 정의되어 있다. 스키마는 그래프QL 작업은 특정 데이터 소스에 종속되지 않으므로 데이터베이스, HTTP 엔드포인트<sub>endpoint</sub>, 마이크로서비스 또는 서버리스 함수 등이 제공하는 모든 기능과 상호작용하기 위해 자유롭게 사용할 수 있다.
AWS AppSync를 사용하면 서버, API 관리뿐만 아니라 보안까지 AWS에 위임할 수 있다.
# 3. AWS Amplify CLI 소개
## 3.1. Amplify CLI 설치와 설정
``` bash
npm install -g @aws-amplify/cli
```
CLI 설치가 완료되면 AWS 계정의 IAM<sub>Identity and Access Management</sub> 사용자를 이용하여 설정해야 한다.
새로운 사용자를 만들고 CLI를 설정하기 위해 `configure` 명령어를 실행한다.
``` bash
amplify configure
```
명령어 실행 후 다음 단계를 진행한다.
1. Specify the AWS region
2. Specify the username
입력 후 CLI가 AWS IAM 대시보드<sub>dashboard</sub> 페이지를 열고 사용자 생성을 진행한다.
액세스 키 ID와 보안 액세스 키의 값을 복사하여 CLI에 값을 붙여 넣으면 성공적으로 설정이 완료되고 서비스 생성을 할 수 있다.
## 3.2. 첫 번째 Amplify 프로젝트
먼저 새로운 리액트 프로젝트를 생성하는 것으로 시작한다.
``` bash
npx create-react-app amplify-app

# 리액트 애플리케이션 생성 후, 생성된 리액트 프로젝트 디렉터리로 이동한다.
cd amplify-app
```
이제 클라이언트에서 사용할 Amplify를 설치해야 한다.
``` bash
npm install aws-amplify @aws-amplify/ui-react
```
그 후 Amplify 프로젝트를 생성하기 위해 `init` 명령어를 실행한다.
``` bash
amplify init
```
Amplify Gen 1을 사용할 경우 다음 단계들을 진행하게 된다.
```
1. Enter a name for the project
2. Enter a name for the environment
3. Choose your default editor
4. Choose the type of app that you're building
5. What JavaScript framework are you using?
6. Source directory path
7. Distribution directory path
8. Build command
9. Start command
10. Select the authentication method you want to use
11. Please choose the profile you want to use
```
모든 단계를 마치면 Amplify CLI가 새 Amplify 프로젝트의 생성을 진행한다.
이제 `src` 디렉터리에 `aws-exports` 파일과 루트 디렉터리에 `amplify` 디렉터리가 추가된다.
### aws-exports 파일
`aws-exports` 파일은 사용자의 자격 증명을 가지고 CLI에서 생성한 리소스 카테고리의 키-값<sub>key-value</sub> 쌍이다.
### amplify 디렉터리
이 디렉터리에는 Amplify 프로젝트의 모든 코드와 설정 파일이 있고, 하위 디렉터리로 `backend` 디렉터리와 `#current-cloud-backend` 디렉터리가 있다.
#### backend 디렉터리
이 디렉터리에는 AppSync API 용 그래프QL, 스키마, 서버리스 함수를 위한 소스 코드, 현재 Amplify 프로젝트의 상태를 나타내는 인프라 정의 코드 같은 것들이 있다.
#### current-cloud-backend 디렉터리
이 디렉터리에는 마지막으로 Amplify의 push 명령어를 사용하여 클라우드에 배포된 리소스를 반영하는 코드와 설정이 있다. 이미 클라우드에 생성된 리소스의 설정과 현재 로컬 변경 사항이 반영된 `backend` 디렉터리의 설정을 CLI가 구별할 수 있도록 도와준다.
## 3.3. 첫 번째 서비스 생성과 배포
Amplify의 `add` 명령어를 이용하면 새 서비스를 추가할 수 있다.
``` bash
amplify add auth
```
이제 다음과 같은 단계를 진행한다.
``` 
1. Do you want to use the default authentication and security configuration?
2. How do you want users to be able to sign in?
3. Do you want to configure advanced settings?
4. Are you sure you want to continue?
```
배포가 완료되면 인증 서비스는 성공적으로 생성되었다.
인증 서비스를 이용하는 방법에는 여러 가지가 있다. `signUp`, `signIn`, `signOut` 등 30가지 이상의 메서드<sub>method</sub>를 사용할 수 있는 `Auth` 클래스를 사용하거나 사전 구성된 UI로 전체 인증 흐름을 진행하는 `withAuthenticator` 같은 프레임워크 컴포넌트를 사용할 수 있다.
먼저 Amplify와 함께 동작하도록 리액트 애플리케이션을 설정하기 위해 `src/index.js`를 열고 다음과 같이 수정한다.
``` js
import React from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import { Amplify } from 'aws-amplify';
import config from './aws-exports';

Amplify.configure(config);

const container = document.getElementById('root');
const root = createRoot(container);

root.render(
    <React.StrictMode>
        <App />
    </React.StrictMode>
);

reportWebVitals();
```
다음으로 `src/App.js`를 열고 다음과 같이 수정한다.
``` js
import React from 'react';
import { Authenticator } from '@aws-amplify/ui-react';
import '@aws-amplify/ui-react/styles.css';

function App() {
  return (
      <Authenticator>
        {({ signOut, user }) => (
            <div style={{ padding: '20px' }}>
              <h1>Hello from AWS Amplify</h1>
              <p>Welcome, {user?.signInDetails?.loginId || user?.username}!</p>
              <button onClick={signOut} style={{
                padding: '10px 20px',
                backgroundColor: '#ff4757',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer'
              }}>
                Sign Out
              </button>
            </div>
        )}
      </Authenticator>
  );
}

export default App;
```
이제 애플리케이션을 실행해서 테스트를 할 수 있다.
``` bash
npm start
```
![[Pasted image 20250718165714.png]]
## 3.4. 리소스 삭제
개별 기능을 제거하기 위해서는 `remove` 명령어를 사용한다.
``` bash
amplify remove auth
```
전체 Amplify 프로젝트를 삭제하려면 `delete` 명령어를 이용하면 된다.
``` bash
amplify delete
```
