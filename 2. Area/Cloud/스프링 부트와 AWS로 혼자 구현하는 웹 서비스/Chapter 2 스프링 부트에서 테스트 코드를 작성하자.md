---
date: 2025-07-20
tags:
  - cloud
  - aws
  - spring
---
# 1. 테스트 코드 소개
TDD는 **테스트가 주도하는 개발**을 이야기한다. **테스트 코드를 먼저 작성**하는 것부터 시작한다. 
![[Pasted image 20250720151023.png]]
단위 테스트는 TDD의 첫 번째 단계인 **기능 단위의 테스트 코드를 작성**하는 것을 이야기한다. 단위 테스트의 이점은 다음과 같다.
- 개발단계 초기에 문제를 발견해 도와준다
- 개발자가 나중에 코드를 리팩토링하거나 라이브러리 업그레이드 등에서 기존 기능이 올바르게 작동하는지 확인할 수 있다
- 기능에 대한 불확실성을 감소시킬 수 있다.
- 시스템에 대한 실제 문서를 제공한다. 즉, 단위 테스트 자체가 문서로 사용할 수 있다.
# 2. Hello Controller 테스트 코드 작성하기
Java 디렉토리를 마우스 오른쪽 버튼을 클릭하여 `새로 만들기 -> 패키지`를 차례로 선택해서 생성한다.
![[Pasted image 20250720161434.png]]
패키지와 마찬가지로 마우스 오른쪽 버튼으로 클릭, `새로 만들기 -> Java 클래스`를 선택한다.
![[Pasted image 20250720161658.png]]
클래스의 코드를 다음과 같이 작성한다.
``` java
package com.jojoldu.book.springboot;  
  
import org.springframework.boot.SpringApplication;  
import org.springframework.boot.autoconfigure.SpringBootApplication;  
  
@SpringBootApplication  
public class Application {  
    public static void main(String[] args) {  
        SpringApplication.run(Application.class, args);  
    }  
}
```
방금 생성한 Application 클래스는 앞으로 만들 프로젝트의 **메인 클래스**가 된다.
`@SpringBootApplication`으로 인해 스프링 부트의 자동 설정, 스프링 Bean 읽기와 생성을 모두 자동으로 설정된다. 특히나 `@SpringBootApplication`**이 있는 위치부터 설정을 읽기** 때문에 이 클래스는 **프로젝트의 최상단에 위치**해야만 한다.
main 메소드에서 실행하는 `SpringApplication.run`으로 인해 내장 WAS<sub>Web Application Server, 웹 애플리케이션 서버</sub>를 실행한다. 이렇게 되면 항상 서버에 **톰캣<sub>Tomcat</sub>을 설치할 필요가 없게 되고**, 스프링 부트로 만들어진 Jar 파일로 실행하면 된다.
이번에는 현재 패키지 하위에 web이란 패키지를 만들어 본다.
![[Pasted image 20250720164937.png]]
테스트해볼 컨트롤러를 만들고 간단한 API를 만든다.
``` java
package com.jojoldu.book.springboot.web;  
  
import org.springframework.web.bind.annotation.GetMapping;  
import org.springframework.web.bind.annotation.RestController;  
  
@RestController  
public class HelloController {  
    @GetMapping("/hello")  
    public String hello() {  
        return "hello world";  
    }  
}
```
작성한 코드가 제대로 작동하는지 테스트를 하기 위해 **테스트 코드로 검증**해본다. `src/test/java` 디렉토리에 앞에서 생성했던 패키지를 그대로 다시 생성한다.
![[Pasted image 20250720165851.png]]
테스트 코드를 작성할 클래스 `HelloControllerTest`를 생성한다.
``` java
package com.jojoldu.book.springboot;  
  
import org.junit.jupiter.api.Test;  
import org.junit.jupiter.api.extension.ExtendWith;  
import org.springframework.beans.factory.annotation.Autowired;  
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;  
import org.springframework.test.context.junit.jupiter.SpringExtension;  
import org.springframework.test.web.servlet.MockMvc;  
import com.jojoldu.book.springboot.web.HelloController;  
  
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;  
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;  
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;  
  
@ExtendWith(SpringExtension.class)  
@WebMvcTest(controllers = HelloController.class)  
public class HelloControllerTest {  
  
    @Autowired  
    private MockMvc mvc;  
  
    @Test  
    public void testHello() throws Exception {  
        String hello = "hello";  
  
        mvc.perform(get("/hello")).andExpect(status().isOk()).andExpect(content().string(hello));  
    }  
}
```
## 코드 설명
### 1. `@RunWith(SpringRunner.class)`
- 테스트를 진행할 때 JUnit에 내장된 실행자 외에 다른 실행자를 실행시킨다.
- 여기서는 SpringRunner라는 스프링 실행자를 사용한다.
- 즉, 스프링 부트 테스트와 JUnit 사이에 연결자 역할을 한다.
### 2. `@WebMvcTest`
- 여러 스프링 테스트 어노테이션 중, Web(Spring MVC)에 집중할 수 있는 어노테이션이다.
- 선언할 경우 `@Controller`, `@ControllerAdvice` 등을 사용할 수 있다.
### 3. `@Autowired`
- 스프링이 관리하는 빈(Bean)을 주입 받는다.
### 4. `private MockMVC mvc`
- 웹 API를 테스트할 때 사용한다.
- 스프링 MVC 테스트의 시작점이다.
- 이 클래스를 통해 HTTP GET, POST 등에 대한 API 테스트를 할 수 있다.
### 5. `mvc.perform(get("/hello"))`
- `MockMvc`를 통해 `/hello` 주소로 HTTP GET 요청을 한다.
- 체이닝이 지원되어 아래와 같이 여러 검증 기능을 선언할 수 있다.
### 6. `.andExpect(status().isOk())`
- `mvc.perform`의 결과를 검증한다.
- HTTP Header의 Status를 검증한다.
### 7. `andExpect(content().string(hello))`
- `mvc.perform`의 결과를 검증한다.
- 응답 본문의 내용을 검증한다.
- `Controller`에서 "hello"를 리턴하기 때문에 이 값이 맞는지 검증한다.
## 테스트 코드 실행
`HelloControllerTest를 실행`을 클릭한다.
![[Pasted image 20250720174111.png]]
이제 Application.java 파일로 이동해서 main 메소드의 `Application.main()을 실행`을 클릭해서 실행한다.
![[Pasted image 20250720174315.png]]
실행해보면 톰캣 서버가 8080 포트로 실행되었다는 것도 출력된다.
![[Pasted image 20250720174409.png]]
# 3. 롬복 소개 및 설치
**롬복**<sub>Lombok</sub>은 자바 개발할 때 자주 사용하는 코드 Getter, Setter, 기본생성자, toString 등을 어노테이션으로 자동 생성해준다.
프로젝트에 룸복을 추가하기 위해 build.gradle에 다음의 코드를 추가한다.
``` gradle
compileOnly 'org.projectlombok:lombok' 
annotationProcessor 'org.projectlombok:lombok'
```
롬복 플러그인은 이미 설치된 것이 확인되었으므로 설치는 건너뛴다.
# 4. Hello Controller 코드를 롬복으로 전환하기
먼저 web 패키지에 dto 패키지를 추가한다. 앞으로 **모든 응답 Dto는 이 Dto 패키지에 추가**한다. 이 패키지에 HelloResponseDto를 생성한다.
![[Pasted image 20250720175846.png]]
HelloResponseDo 코드를 작성한다.
``` java
package com.jojoldu.book.springboot.web.dto;  
  
import lombok.Getter;  
import lombok.RequiredArgsConstructor;  
  
@Getter  
@RequiredArgsConstructor  
public class HelloResponseDto {  
    private final String name;  
    private final int amount;  
}
```
## 코드 설명
### 1. `@Getter`
- 선언된 모든 필드의 `get` 메소드를 생성한다.
### 2. `@RequiredArgsConstructor`
- 선언된 모든 `final` 필드가 포함된 생성자를 생성해준다.
- `final`이 없는 필드는 생성자에 포함되지 않는다.
## 테스트 코드 작성 및 실행
``` java
package com.jojoldu.book.springboot.dto;  
  
import com.jojoldu.book.springboot.web.dto.HelloResponseDto;  
import org.junit.jupiter.api.Test;  
import static org.assertj.core.api.Assertions.assertThat;  
  
public class HelloResponseDtoTest {  
      
    @Test  
    public void testDto() {  
        String name = "test";  
        int amount = 1000;  
          
        HelloResponseDto dto = new HelloResponseDto(name, amount);  
          
        assertThat(dto.getName()).isEqualTo(name);  
        assertThat(dto.getAmount()).isEqualTo(amount);  
    }  
}
```
### 코드 설명
#### 1. `assertThat`
- `assertj`라는 테스트 검증 라이브러리의 검증 메소드이다.
- 검증하고 싶은 대상을 메소드 인자로 받는다.
- 에소드 체이닝이 지원되어 `isEqualTo`와 같이 메소드를 이어서 사용할 수 있다.
#### 2. `isEqualTo`
- `assertj`의 동등 비교 메소드이다.
- `assertThat`에 있는 값과 `isEqualTo`의 값을 비교해서 같을 때만 성공이다.
이제 테스트 메소드를 실행해 본다.
![[Pasted image 20250720180808.png]]
## 코드 추가
이제 HelloController에도 새로 만든 ResponseDto를 사용하도록 코드를 수정한다.
``` java
package com.jojoldu.book.springboot.web;  
  
import com.jojoldu.book.springboot.web.dto.HelloResponseDto;  
import org.springframework.web.bind.annotation.GetMapping;  
import org.springframework.web.bind.annotation.RequestParam;  
import org.springframework.web.bind.annotation.RestController;  
  
@RestController  
public class HelloController {  
    @GetMapping("/hello")  
    public String hello() {  
        return "hello world";  
    }  
    @GetMapping("/hello/dto")  
    public HelloResponseDto helloDto(@RequestParam("name") String name, @RequestParam("amount") int amount) {  
        return new HelloResponseDto(name, amount);  
    }  
}
```
### 코드 설명
#### 1. `@RequestParam`
- 외부에서 API로 넘긴 파라미커를 가져오는 어노테이션이다
- 여기서는 외부에서 name(`@RequestParam("name")`)이란 이름으로 넘긴 파라미터를 메소드 파라미터 name(`String name`)에 저장하게 된다.
추가된 API를 테스트하는 코드를 HelloControllerTest에 추가한다.
``` java
package com.jojoldu.book.springboot.web;  
  
import org.junit.jupiter.api.Test;  
import org.junit.jupiter.api.extension.ExtendWith;  
import org.springframework.beans.factory.annotation.Autowired;  
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;  
import org.springframework.test.context.junit.jupiter.SpringExtension;  
import org.springframework.test.web.servlet.MockMvc;  
  
import static org.hamcrest.Matchers.is;  
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;  
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;  
  
@ExtendWith(SpringExtension.class)  
@WebMvcTest(controllers = HelloController.class)  
public class HelloControllerTest {  
  
    @Autowired  
    private MockMvc mvc;  
  
    @Test  
    public void testHello() throws Exception {  
        String hello = "hello";  
  
        mvc.perform(get("/hello")).andExpect(status().isOk()).andExpect(content().string(hello));  
    }  
      
    @Test  
    public void testHelloDto() throws Exception {  
        String name = "hello";  
        int amount = 1000;  
          
        mvc.perform(  
                get("/hello/dto").param("name", name).param("amount", String.valueOf(amount)))  
                        .andExpect(status().isOk())  
                        .andExpect(jsonPath("$.name", is(name)))  
                        .andExpect(jsonPath("$.amount", is(amount)));  
    }  
}
```
### 코드 설명
#### 1. `param`
- API 테스트할 때 사용될 요청 파라미터를 설정한다
- 단, 값은 `String`만 허용된다.
- 그래서 숫자/날짜 등의 데이터도 등록할 때는 문자열로 변경해야만 가능하다.
#### 2. `jsonPath`
- JSON 응답값을 필드별로 검증할 수 있는 메소드이다.
- `$`를 기준으로 필드명을 명시한다.
- 여기서는 `name`과 `amount`를 검증하니 `$.name`, `$.amount`로 검증한다.
이제 추가된 API를 테스트하면 다음과 같다.
![[Pasted image 20250720192730.png]]