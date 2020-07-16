# Load_map

# 기초 좀 튼튼히 하자. 아 맨날 다 까먹고..
대답도 제대로 못하고 후회하고 기초를 튼튼히 하자
# 논문 리뷰를 기록하기  레포하나 만들기
# Lecture 
딥러닝 홀로서기 - 정주행 1번 완료. 코드구현은 하지 않았음.

테크보이 코딩 배우기

https://www.youtube.com/watch?v=M6kQTpIqpLs&list=PLa7Lj786Q-Gts3-LsBl5I56YQrQb4sHxI

# Paper review
인공지능 학회 (NIPS, ICML, ICLR, ICCV, CVPR 등)와 보안 학회 (S&P, NDSS, Secuirty, CSS 등) 논문
 
# 하버드 statistics
https://www.edwith.org/harvardprobability/joinLectures/17924

# sf cs231n
http://cs231n.stanford.edu/syllabus.html
argparse

모듈리스트 시퀀셜

파이토치 모델 하나

우분투

파이토치

다커

깃허브

파이썬

시각화 

웹크롤링

L1

L2 loss

선형대수, 미분방정식, 회귀분석

Value function 이용해서 Bellman equation 푸는 문제

그레인저 인과관계

게임이론

2nd price auction

계량경제학

R

딥러닝이라는게 Neural Network가 중층으로 연결된 구조라는 것, 그 구조 자체가 Regression을 반복적으로 실행하면서 데이터 전처리를 모델 내부에서 해결하는 작업

Autoencoder

Kalman filter

Bayesian 통계학

Logistic regression + Ensemble = Neural net 이라는 개념 이해, Activation function이 사실은 Kernel이라는 이해

Q-Q plot

데이터 안에 비선형의 강한 패턴이 반복될 때만 신경망 모델이 유의미하다는 것을 이해

Markov process와 Markov Decision process의 차이점은 나의 행동이 주변 환경을 변화시키는지 여부

딥러닝 모델을 웹에 올리기

5페이지부터 다 읽어야 돼

https://blog.pabii.co.kr/category/%eb%8d%b0%ec%9d%b4%ed%84%b0%ec%82%ac%ec%9d%b4%ec%96%b8%ec%8a%a4/page/5/

# Xavier initial values

kaggle

트윗 자연어처리 참가

https://www.kaggle.com/c/nlp-getting-started/overview


Correlation 구하는 공식
Network theory
Graph Theory
Newton method
Convex Optimization
Statistical Machine Learning
Tree model
Clustering
Ridge, Lasso
데이터에 알고리즘을 적용하기 전에 먼저 정규분포를 따르는지 아닌지 확인하기 위해 Histogram과 QQ-plot을 그려봤는데요,

1. 회귀모델의 진단
Q. 다중 회귀분석 (Multinomial regression)에서는 모델링 과정에서 정규성, 선형성, 등분산성, 이상치 등을 살펴보기 위해서 Plot을 그려보던데, 머신러닝에서 Gradient descent를 할 때는 Regression에서 하던것처럼 잔차에 대한 여러 가정들을 별도로 확인해보지 않아도 되나요?

A. 수업 중 Ensemble을 다루면서 저 질문에 대한 답변을 좀 더 깊이 있게 다룬다. 통계학에서 Regression은 모델링의 결과값이 Error 최소화, 즉 Variance 최소화에 맞춰져 있다. Variance가 크게 나오면 Regression 모델에서 나온 상관계수들이 0이 아니라고 주장하는 p-value 값을 믿을 수가 없기 때문이다. 그런데, Regression을 (특히 Linear regression을) 적용하는 데이터들은 Error의 패턴이 없는 경우, 즉 데이터의 Error에 해당하는 부분이 랜덤으로 생성된 경우다. 머신러닝을 쓰는 이유는 그런 랜덤이 아닌 부분, 패턴이 있는 부분, 특히 패턴이 단순한 선형 관계 (ax+b)가 아닌 부분을 찾아내기 위해서다.

그런 비선형의 복잡한 패턴을 다항식으로 (무식하게?) 찾아내는데는 한계가 있으니까, 머신러닝의 여러가지 방법론들 (SVM, Decision Tree, Neural network 등등)을 활용하는 것이고, Gradient descent는 그런 탐정놀이 게임 중 수학식으로 쉽게 떨어지지 않는 trial-and-error 계산을 하기 위해 디자인된 계산 방법이다. 비선형 함수를 찾아낸다는 말 자체에서 이미 데이터의 non-random을 가정하고 접근하는 것이고, Gradient descent는 계산 비용을 줄이면서 결과값이 최대한 근사치가 나오도록하는 계산법이다보니, 당연히 잔차의 구조에 대한 고민을 안 하게 될 수 밖에 없다.

 

2. 데이터의 정규성 확인
Q. 데이터에 알고리즘을 적용하기 전에 먼저 정규분포를 따르는지 아닌지 확인하기 위해 Histogram과 QQ-plot을 그려봤는데요, t-test 적용에서와 같이 정규성 검정까진 할 필요가 없는 건가요? 중심극한정리에서는 표본이 충분할 때 “표본분포”가 정규분포의 형태를 따른다고 했는데, 우리가 가진 데이터는 표본 하나인 것과 같으니 단순 데이터 셋의 정규성만 확인해도 충분한 것인지 궁금합니다.

A. 우선 몇 가지 사실 관계를 점검해야할 것 같다.

t-test하는데 정규성 검정이라는 것은, 원래 데이터가 (approximately, more precisely asymptotically) 정규분포를 따르고 있는지에 대한 확인이다. 원래 데이터가 정규분포가 아니라면 학부 수업 때 배운 t-stat말고 다른 계산법이 존재한다. (수학 & 통계학 시간에 Poisson일 경우 어떻게 해야된다고 했는지 노트 참조)

또 중심극한정리 (Central Limit Theorem, CLT)는 표본이 충분하면 분포가 정규분포를 따르는게 아니라, 샘플 추출을 여러번 했을 때, 그 평균값이 정규분포를 따른다는 정리다. 쉽게 예시를 들면, 모집단을 모르고 단순하게 1,000명의 표본집단만 여러번 뽑으면 (ex. 선거 직전 여론 조사), 그 표본집단의 평균값들 (처음엔 평균이 49%, 다음엔 51%, 그 다음엔 48.5% 등등등)이 정규 분포 형태로 고르게 퍼진다는 뜻이다. (수학적으로는 정규분포에 수렴한다고 표현한다.) 데이터가 많으면 무조건 정규분포에 수렴한다고 잘못 알고 있는 사람들이 많은데 (나도 예전엔….), 그러다보니 Underlying distribution이 포아송 분포인데도 t stat을 무조건 정규분포 기반으로 계산해서 모델 테스트를 엉망으로 하고 있다고 수학 & 통계학 시간에 강조한 바 있다.

일단 위의 1번에서 설명한대로, 데이터(의 에러)가 정규분포를 따른다면 Random 이라는 뜻이니까, 단순한 Linear regression으로 충분한 모델링이 되고, 머신러닝에서 배운 여러 테크닉들은 그런 randomness가 깨질 때만 의미가 있다. 따라서 현재 가진 데이터의 정규성을 확인해보는 것만으로 충분하다.

 

3. lm(), glm() 등의 함수와 caret 패키지 사용의 차이
Q. R 코딩 실습 부분에서 저는 평소 caret 패키지를 사용하지 않았었는데요, lm(), glm(), svm() 드으이 함수를 쓰는 것과, caret 패키지에서 method = ‘lm’, ‘glm’, ‘svm’ 등을 지정하는 것은 계산 방식에서 어떤 차이가 있나요? 앞의 함수들은 Normal equation으로 해를 구하고, caret은 gradient descent 알고리즘을 이용하는 건가요?

A. 우선 Normal equation이 작동하는 유일한 영역은 데이터가 정규분포를 따르고, 그 때 Best Unbiased Estimator인 MLE가 OLS estimator와 똑같을 때 밖에 없다. 이게 통계학 수업 시간에 배우는 내용이다.

위에 나온 glm(), svm() 같은 머신러닝 모델을 간략화해놓은 함수들은 Normal equation을 쓰는게 아니라, glm은 Newton’s method를 이용한 approximation, svm은 Duality를 이용한 Group error minimization 계산을 한다.

두 계산 함수 그룹의 차이는, caret은 R 플랫폼의 함수들을 결합해서 위의 계산을 하도록 만든 패키지고, svm, xgboost등의 외부 패키지들은 외부 플랫폼 (C++, Java 등등)의 계산 모듈을 빌려온다. 왜? 계산 속도 자체만 놓고보면 시스템 자원을 효율적으로 쓰지 못하는 R이 압도적으로 느리니까. 참고로 Python, R등의 언어는 C++, Java가 시스템 자원을 활용하는 방식과는 완전히 다른 설계로 짜여져 있기 때문에, 계산 속도가 현격하게 차이날 수 밖에 없다.

Python에서도 scikit-learn을 쓰면 R의 caret과 똑같은 상황이 벌어질 것이다. 진정한 개발용 언어인 Java, C++이 1초만에 계산하는걸 R이나 Python위에서 Java, C++의 계산 모듈을 빌리면 3-4초로 늘어나고, 그냥 caret, scikit-learn 위에서 돌리면 거의 20초 정도 걸린다는 벤치마크도 있다. 데이터를 처리하고 모델링을 고민할 수 있도록 만들어 놓은 프로그램이니만큼, 복잡한 데이터 구조 (ex. n x k x p x q 행렬)를 처리하는데 초점이 맞춰져 있다보니 정작 계산 속도의 효율성 측면에서는 굉장히 나쁜 플랫폼이 되어버리는 것이다.

 

4. Polynomial regression의 경우의 수
Q. 회귀 직선이 주어진 변수만으로 적합이 안 되어서 Polynomial 형태를 적용하게 될 때, 기존 변수가 n개라면 n(n+1)/2의 경우의 수가 생기게 되잖아요. 그럼 회귀 모델 적용할 때 2차 교호작용까지 고려해서 모델을 만드는게 이런 Polynomial 형태를 고려하는 형태의 예라고 생각해도 될까요? 만약 맞다면, 그래서 2차 교호작용 외 다른 경우의 수도 고려하고 싶다면, 실제로 n(n+1)/2 횟수만큼 변수를 바꿔가며 코딩을 해야하나요?

A. 아니다. (걱정하지 마시라ㅋㅋ)

그런 Polynomial regression을 이것저것 모든 조합으로 다 해보는게 무모한 작업이라는 걸 알고 있기 때문에 svm, decision tree, ensemble, neural network 같은 대체재들이 있는 것이다. Polynomial을 쓴다는 것 자체가 이미 데이터가 선형으로 설명되지 않는다는 이해를 깔고 있고, 데이터가 랜덤이라는 가정을 깨고 들어가는 것이다. Non-linear 패턴을 잡아낸다고 했을 때, 인간이 할 수 있는 가장 단순(무식)한 방법이 Polynomial이고, 이걸 좀 더 수학적으로 세련되게 (학문적으로는 Elegant라고 표현함) 만드는 모델이 바로 여러분들이 열심히 머신러닝 테크닉이라고 배우는 svm, ensemble, neural net 같은 모델들이다.

참고로, 지금도 사회학 계열로 연구하는 분들 중에 통계학을 거의 모르는 분들은 stepwise regression이라고 모든 변수를 다 넣어보고 그 중에 설명력 (R-squared)이 제일 좋은 모델을 “단순하게” 고르는 경우도 있다. (저 분도 박사학위가 있는데, 니가 생각하는거처럼 그렇게 절망적으로 통계학을 모르진 않을꺼야…라고? 연구자 취급 못 받을 사람들 진짜로 많다 ㅋㅋㅋ)

 

5. Variable Importance를 통한 변수의 중요도 판단
Q. Random Forest, Boosting 계열의 모델을 보면 Variable importance에 순위를 잡던데, 그 변수가 중요변수가 아닐수도 있다고 하셨던 부분이 어떤 경우였는지 잘 기억이 안 납니다. 

A. Random Forest를 비롯한 Tree 기반의 모델들에서 중요한 변수들 중에, 한 변수가 Tree 안에 여러번 반복이 되기 때문에 중요한 변수로 나타나는 경우가 있다. Tree 구조를 보면 이해가 되겠지만, 그 변수의 여러 구간 (ex. 3-5, 10-15, 30-37 등등, 연속되지 않은 여러 구간)이 y값에 영향을 주는 경우에 Linear regression에서는 연속된 구간 값을 사용하는 탓에 두 모델간 변수의 중요도가 다르게 나올 여지가 있다. 수업 중에서 Variable Importance plot을 그린 다음, 이걸 Regression 기반 모델들에도 무조건 적용하면 문제가 생길 수 있다고 지적하기도 한다.

 

6. 기타 추가 질문
Q. 우연히 ‘머신러닝 피드백을 이용한 게임 지표 분석’ 이라는 글을 보게 되었는데, 제가 실무적으로 활용하면 도움이 될듯한 내용들이라 꼭 적용해보고 싶은데요, 정확히 어떤 개념을 이용하는 것인지 잘 모르겠습니다.

링크의 글을 읽어보면, Vowpal-wabbit을 통해 손실함수를 계산해서 각 독립 변수들이 종속 변수에 어떻게 영향을 주는지 귀납적으로 추론한다는 방식이고, 여기서 독립 변수마다 계산된 연관성 점수 (Relation score)에 따라 positive, negative를 판단하여 독립 변수들의 영향도? 효과? 등을 파악한다는 것인데 여기서 나오는 연관성 점수가 무엇일까요?

A. 두 가지 경우를 생각해보자.

1번. 위의 5번에서 본 Variable importance를 구하는데, +인 경우와 -인 경우를 구분해서 순위를 다시 잡으면 어떻게 될까? 그 다음 단순하게 몇 %의 기여도만 따지는게 아니라, 기여도가 있는 변수들을 다 모아놓고 -100% ~ +100%로 재조정을 한다면?

2번. Regression에서도 가능할까? 물론이다. 모든 변수를 standardize 해놓고, 각각의 상관계수 값에 따라서 어떤 변수가 더 큰 기여도를 가지는지, 어떤 변수가 마이너스 기여도를 가지는지 구분할 수 있다. 그렇게 상관계수들을 모아놓고 등수를 잡거나, -100% ~ +100%로 재조정을 해도 된다.

실제로 Variable importance를 구하는 과정을 보면, 각각의 변수가 cost function 값이 얼마나 변하도록 만드는지에 따라서 변수의 중요도를 따진다. Regression에서 모든 변수를 같은 구간에서 움직이게 묶은 다음 (i.e. 표준화) 상관계수를 보는 것도 같은 컨셉이다.

저 링크의 글을 보면 Vowpal-wabbit이 무슨 엄청나게 기술(!)력 있는 모델인 것처럼 느껴지는데, 정작 위에서 설명한 단순한 중요도 계산 + 재조정 작업에 불과하다.

“우와~ 이거 완전 대박이네!”, “어캐 베끼지?”

보통 공대생, 특히 수학 (계산말고 수학) 공부 많이 안 하는 전공자들에게서 흔히 보는 모습인데, 뭔가 새로운 계산만 나오면 “우와~ 이거 완전 대박이네!”, “어캐 베끼지?” 이런 반응을 보이더라. (뭐, 스티브 잡스도 베끼는거 좋아하긴 했다.) 머신러닝에서 Neural network도 그래서 “떴다”고 생각한다. 그러나, 필자 같은 사람들 눈에는 “저거 XYZABCD로 계산했겠네”가 그냥 눈에 보인다. 평소에도 저런 모델링을 하는걸 머리속으로 고민하고 있으니까. 더더군다나 Vowpal-wabbit은 그렇게 수학적으로 Elegant 한 방법도 아니다. 저거 제대로 돌아가게 하려면 보정작업이 엄청나게 많이 필요할 것이다. (직접 적용해보시라.)


6. 게임이론

 

김영세 - 게임이론
- 게임이론의 권위자 김영세 교수님의 책. 입문서로는 좀 어렵다. 김영세 교수님은 이 외에도 게임이론에 관한 책을 많이 저술하셨으니 참고하시면 좋을듯.
 
왕규호, 조인구 - 게임이론

- 왕규호 교수님과 조인구 교수님 모두 게임이론의 권위자 이십니다. 조인구 교수님은 일리노이 대학에 계신데, 아주 촉망받는 한국인 교수님이시죠. 입문서로는 좀 어려운 경향이 있습니다. 그래도 설명이 꼼꼼하고 식도 자세히 설명되어 있기 때문에 손끝으로 공부하기 좋아하는 학생이라면 좋아할 책이라고 생각됩니다. 연습문제도 많습니다. <경제갤러 'ntme'>


Roger A. McCain - Game Theory [번역본 게임이론 - 이규억 역(아주대학교 교수)]

- 처음 공부하는 사람이 직관적으로 이해할 수 있도록 쉽고 친절하게 쓰여져 있습니다. 서술방식이나 논리전개가 평이하긴 하지만 내용의 넓이와 깊이는 상당하다고 생각됩니다. 앞에서 얘기한 왕규호 조인구 교수님의 게임이론이 3인공저 현대경제학원론 책 이라면 이 책은 맨큐의 경제학이라고 할 수 있습니다. 번역본의 완성도도 매우 높다고 생각되네요. 이규억 교수님은 게임이론의 창시자이신 모겐스턴 교수님의 조교로 계신적도 있습니다. <경제갤러 'ntme'>


프로그래머를 위한 선형대수
김창진 교수님 강의 노트
따라하며 배우는 데이터 사이언스
김우철 교수님 수리통계학, 회귀분석
박성현교수님의 회귀분석
김우철 - 현대통계학(빨간책)

홈페이지 구현 
https://tutorials.pytorch.kr/intermediate/flask_rest_api_tutorial.html
https://www.fun-coding.org/flask_basic-2.html

p-value

1. 수리통계학 모델링 지식
파비 블로그를 오랫동안 보신 분들은 이미 지겹도록 봤겠지만, 수리통계학 모델링 지식 수준이 일정 레벨 미만이면 데이터 사이언티스트 자격이 없다고 생각한다. 또, 경력이 쌓이면 연봉 수준이 오르겠지만, 수리통계학 지식 수준의 높낮이를 1번 기준으로 연봉 Range를 산정하고, 그 다음에 후술하는 다른 능력치가 경험에 따라 얼마나 축적되었느냐에 따라 연봉 수준이 결정된다고봐도 틀리지 않을 것이다.

개발자들이 자꾸 착각해서 자기들이 데이터 사이언티스트 후보인줄 알고 있으니까 그쪽 예시를 들면, 알고리즘 문제를 백번 풀어봐야 동접자수 백만명이 넘어가는 게임 서버를 운영하는데 직접적인 도움은 안 된다는 걸 경험으로 이해하고 있을 것이다. 여기서 알고리즘은 R, Python 써 본 경험을 말하고, 동접 백만명의 서버 운영을 위해 알아야하는 서버/DB쪽 지식이 바로 수리통계학 모델링 지식이라고 보면 된다.

예전에 데이터 사이언티스트 채용 글에도 썼던 것처럼, 대학원에서 배울법한 아래의 수리통계학 지식은 많이 알수록 좋다.

시계열 및 Sequence형 데이터 처리 (Autoregressive process, VAR, Kalman filtering, Fourier Transform 등)
패널 (Panel) 데이터 처리 (Bargava-Sargan, Arellano-Bond)
베이지안 통계학 (Thompson sampling 아이디어를 적용하는 모델들 ex. Bandit, Temporal-difference, Metropolis-Hasting 등)
Predictive modeling (Online learning, FTRL 계열 모델들)
Multi-task learning
네트워크 이론 (Eigen centrality를 활용한 Node별 ordering 경험 or 환상형 네트워크에서 propagation 효과를 풀어낸 경험)
게임이론 (Cho and Kreps, Trembling-Hand-Equilibrium 등 uncertainty case)
왜냐고? 머신러닝이라는게 사실은 계산통계학인데, 그 계산통계학 방법론들은 수리통계학을 쓰는 전공으로 길게 훈련을 받은 분들이 만들어낸 방법이니까. 그런 지식 중 일부를 회사 업무에 어떻게 활용하느냐에 따라 사업 내용도 바뀌고, 결과물도 바뀔 수 있을테니까.

On-the-job 데이터 사이언티스트가 그런 계산 테크닉을 새로 만들어내야 할 필요는 없지만, 최소한 그런 계산 테크닉들을 바로바로 이해할 수 있는 수준의 수학적 훈련은 받아놨어야 새로 나오는 “기술(?)”에 바로바로 대응하는 시니어가 될 수 있지 않을까?

 

2. 해당 업무에 대한 비지니스적 이해 + 적용할 수 있는 직관
위의 수리통계학 지식은 어떤 맥락에서 어떻게 쓸 수 있을지에 대한 센스가 없으면 그냥 책 속의 지식으로 끝난다.

대학원에서 열심히 공부하신 분들이 정작 직장에가서 배운 지식을 못 써먹고 있는 이유가 회사 업무에 별로 필요가 없어서일수도 있지만, 그런 직관을 키워본 적이 없기 때문에, 좀 심하게 말해서 머리가 굳었기 때문에 적용을 못하고 있는 것이다. 그런 분께 대학원 학위는 그냥 잉여 학위가 되어버리겠지.



아주 단순한 예시를 하나 들어보자. 경제학에서 쓰는 Cobb-Douglas 함수가 있다. (위의 방정식 참조) 노동과 자본이라는 요소 결합으로 생산량이 결정되는 단순 방정식을 이용해 공급곡선을 도출하는 굉장히 수학적인 논리 구축 작업 때 배운다. 그런데, 지식이 그 수준에 머물러 있으면 경제학은 실생활에 하나도 못 쓰는 이상한 학문이 되고 만다.

Senior Data Scientist로 있으면서 회사 내의 Confluence page들을 보던 중에, Multi factor가 영향을 주는 경우를 모델링 하려고 다른 Data Scientist가 위의 Cobb-Douglas 함수를 그대로 써 놨더라. 두 변수가 노동과 자본대신 CPC, 과거 반응율 등등의 광고시장 Metric 값으로 바뀌었던 점이 좀 달랐고, 그런 변수들이 광고지면 입찰가에 영향을 주는 작업이 Linear 관계가 아니니까 일단 이렇게 시작한다고 설명을 달아놨던데, 학교에서 배운 지식을 현업에 어떻게 응용할 수 있는지에 대한 아주 간단한 예시가 될 수 있을 것 같다.

각 변수가 얼마나 큰 영향을 미치는지를 a, b 값으로 표현하는 저 방정식에 Log를 취하면 우리가 알고 있는 Linear regression과 거의 동일한 방정식 형태가 된다. 그리고 t, t+1 기 사이의 log값 차이는 변화율 값이다. 이런 단순한 지식을 이용해서 광고지면 입찰가 모델과 결과값이 얼마나 크게 괴리가 나는지, 영향을 줬던 다른 요소들 (회사 기밀이다), 향후 모델 테스트 방식 같은 정보들을 상세하게 정리해놨던데, 그 분이 어떤 학문으로 훈련 받았건 상관없이, 광고지면 구매하는 과정을 잘 이해하고, 수학 모델링을 어떻게 활용할 수 있는지 뛰어난 직관을 갖고 있었기 때문에 할 수 있던 작업이 아니었나 싶다.

몇 달전 어느 물리학 박사과정 생이 데이터 사이언스 수업 끝에 “수학적인 부분은 일부러 깊게 안 들어가셔서 그런지 어렵지 않았는데, 이 분야에서는 그런걸 어떻게 적용한다는 직관이 정말 훨씬 더 중요한 것 같네요.” 라고 하더라.

 

3. 쪼개볼 수 있는 예리한 분석력
Data Analyst라는 직업군이 Data Scientist로 오해받는 경우를 자주보면서, “분석”이라는 단어를 쓰기가 참 걱정되는데, 달리 더 좋은 표현을 못 찾겠다.

어떤 연구를 한다는 건, 남들이 어떤 작업을 해 놨는지 살펴보고 이해한 다음에, 그걸 그대로 베끼는게 아니라 자기만의 새로운 “각도”를 추가하는 작업이라고 생각한다. 당연히 남들이 해 놓은 결과물을 이해할 수 있는 지적인 훈련도와 내공을 갖추고 있어야하고, 남들이 생각해보지 못한 관점에서 해당 연구 과제를 새롭게 볼 수 있는 능력이 아카데믹이 인정하는 “분석” 능력이다. 아카데믹들은 이런 능력을 “Analytic Mind”라고 표현하기도 한다.

쉬운 예시를 들면, 우리 회사 매출액이 올랐던 이유로 A, B, C, D가 나와있었는데, 혹시 Y도 포함되지 않을까, 그리고 그 Y가 A~D와는 전혀 관계없는 독립변수가 아닐까는 관점에서 출발해서, Y를 포함시켜야되는 이유를 합리적으로 정립하고 (연구자들이라면 그걸 수학 모델로 만들어야하겠지만, 굳이 직장에서는..), 실제 데이터를 이용해서 Y값이 매출액에 영향을 줬는지, Y값이 A~D와 전혀 관계없는 요소인지, 혹시 Y값은 다른 외부 요소에 어떤 영향을 받은건 아닌지를 살펴보는 작업이다.

좀 더 고난이도 예시를 들면, 벽돌깨기 게임을 학습하는 자동화 알고리즘 (타칭 “인공지능”)을 만든다고 할 때, 공이 움직이는 위치를 전부 스캔해서 입력하면 학습도 엉망으로 되고, 그 전에 아마 수학적인 관점에서 발산하는 결과를 얻을 것이다. 왜 발산하는지 (왜 딥러닝에 집어넣었는데 Learning이 안되는거지라고 생각하며) 단순히 값 바꿔보면서 백 번, 천 번의 실험으로 경험적으로 이해는 할 수 있을지 몰라도, 해결 방법을 찾기는 힘들 것이다. Experience Replay라는 걸 공부하고나면 이해되겠지만, 공 움직임같이 Sequence형 데이터에서 Autocorrelation이 생기는 부분(i.e. 연결된 움직임)을 제거해주지 않으면 선형대수적으로 Rank problem이 생긴다. 이 때, 시계열에 관련된 통계학 훈련이 잘 되어 있다면 Autocorrelation을 제거할 수 있는 여러 아이디어들을 Thought experiment로, 수식으로 정리해서 결과를 예측해볼 수 있기도 하고, 또 해당 게임의 작동 방식을 잘 이해하고 있으면 그에 맞춰서 데이터를 변형시키는 것도 가능한 옵션이다. 참고로, Experience Replay는 Sequence의 다른 부분에 있는 데이터를 도구변수 (Intrumental variable)처럼 활용해서 Autocorrelation을 제거해준다. 아마 당시에 가능한 옵션들이 Random Sampling을 해서 데이터의 Autocorrelation을 축소시키거나, 공의 움직임 스캔값을 선으로 바꿔 입력하는 것 정도가 있었을 것 같은데, 스캔 데이터 값을 보고 판단했으리라.

위에서 보듯이, 문제를 경험적으로가 아니라 체계적으로 인식하고, 그 해결을 위한 아이디어를 테스트해보는데, 그 테스트가 합리적인 테스트가 되기 위해서는 위에서 말한 수리통계학 모델링 지식, 그 지식을 활용할 수 있는 직관이 필수적이다.

참신한 생각이라는 표현을 좀 바꾸면, 남들이 생각해보지 못했던 각도에서 같은 결과를 바라보고, 그 분석 과정에서 기존의 생각을 더 단단하게 굳히거나, 새로운 관점을 추가할 수 있게되는 맥락이라고 생각하면 맞을 것 같다.

이런 훈련 없어도 “그냥 딥러닝에 다 집어넣으면 알아서 찾아주던데요?” 라고 생각하시는 분들께는 이 글을 바친다.

 

4. “My own” 포트폴리오
데이터 사이언티스트가 되고 싶다는 사람들, 학원에서 어떤 훈련을 받았다는 사람들, 혼자서 열심히 공부해봤다는 사람들 거의 대부분이 갖고 있는 포트폴리오를 보면 인터넷에서 쉽게 찾을 수 있는 타이타닉 데이터, MNIST 데이터를 이용한 작업 결과물들이다.

예외가 있을 수도 있겠지만, 그런 마르고 닳도록 남들이 가져다 쓴 데이터로 본인의 뛰어난 분석력을 보여주는데는 한계가 있을 수 밖에 없다. 되려 자기가 남의 결과물을 잘 베끼는 사람이라는 증거가 되기 십상이다.

지원하는 회사가 있는 산업의 관련 데이터를 뒤져서, 그 데이터를 그 회사에는 이렇게 쓰지 않을까는 생각을 담은 포트폴리오를 만드는 사람은 왜 그렇게 드물까?

그런 포트폴리오 만드는데 분명히 시간이 많이 걸릴 것이다. 그런데, 남들보다 더 뛰어나고 싶으면 계속 훈련해야하는거 아닌가? 내 이력서가 남들보다 덜 뛰어나다면 이런식으로 남들이 하지 않는 작업을 추가해놔야 조금이라도 더 면접관의 눈에 들지 않을까?

데이터 사이언티스트 인턴 채용 공고를 내고 받았던 지원서 중에, 네트워크 모델을 이용해서 병이 퍼져나가는 속도를 최소화할 수 있는 구조에 대한 고민을 담은 포트폴리오를 본 적이 있다. 학부 고학년 수업의 기말 레포트 정도에 해당하는 포트폴리오였는데, 우리 파비가 그런 인턴을 키울 수 있을만큼 여유있는 큰 조직이었다면 한번쯤 그 학생의 내공을 따져봤을 거라고 생각한다. 지금쯤 어딘가 다른 조직에서 성장하고 있으실텐데, 언젠가 꼭 한번 만나보고 싶은 분이다.

단순히 Regression 복잡하게 넣은 Neural net이니 SVM을 돌리는 수준의 공부를 한 분이 아니라, 수학을 이용한 모델링을 배우고 쓸려는 노력을 한 흔적이 보였기 때문이다. Recruiter의 시선을 사로잡는 포트폴리오에 대한 감이 좀 잡히셨으면 한다.

plus, 남들이 다 하니까 따라하는지는 모르겠는데, Kaggle에 굳이 얽매일 필요가 있는지 모르겠다. Kaggle처럼 몇 % 맞다는 결과값에 희열을 느끼는 집단이 주어진 데이터를 쥐어짜는 방법에만 집중하는게 되려 수리통계학 훈련도를 높이는데 장애물이 되지 않을까 싶다.

명심하시라. 데이터 사이언스 업무에서 포트폴리오는 내가 이것저것 많이 해 봤다고 자랑하는 용이 아니라, 나의 수리통계학적 내공을 상대가 가늠할 수 있는 도와주는 매개체일 뿐이다.

 

5. 데이터 베이스 관련 지식
일단 SQL을 할 줄 모르면 거의 업무를 할 수 없을 것이다. 회사에 들어가면 엔지니어들이 자료구조론에 대한 깊은 이해, 비지니스가 잘 돌아가도록 데이터 베이스를 짜 놨을텐데, 원하는 방식으로 DB에서 데이터를 추출해야 위에서 말한 내용을 적용할 수 있기 때문이다.

그래서 회사 들어가려면 SQL부터 먼저 공부해라고 말하는 분들도 많다. 당장 Select, from 한번 안 쳐 봤으면 아무리 금방 배우는 지식이라고 해도 회사 입장에서 선뜻 채용하기는 어려울 것이다.

면접 질문으로 Join을 Outer join으로 걸었을 때, Inner join으로 걸었을 때 어떤 결과가 나오는지 정도는 나올 수 있고, 이 정도 질문은 무사히 대답해야 면접관의 표정이 밝아지지 않을까?

더불어, 업무를 하다보면 자료 구조가 어떻게 바뀌었으면 좋겠다는 고민도 생길 것이고, 그런 이야기를 데이터 엔지니어들과 커뮤니케이션 해야할 일이 엄청나게 많아진다. 그들과 대화가 잘 풀려나가야 회사 업무를 원활하게 할 수 있을텐데, 어쩔 수 없이 자료구조론을 공부할 수 밖에 없을 것이다. 보통 데이터 무결성 (Integrity)에 관련된 질문들을 하고, 배열 (Array) vs. 리스트 (List) 저장 방식의 차이점에 대해서도 묻고, 좀 엉망인 DB Table들을 배열해 놓은 다음, 효율적으로 바꿔라는 과제를 던지기도 한다.

업무를 하다보니, 위의 내용이 데이터 엔지니어에게만 필수 지식이 아니라, 데이터 사이언티스트도 그 분들과 대화가 가능한 수준까지는 지식 수준을 올려놔야한다는 사실을 깨달았다.

혹시나 면접관 중에 데이터 엔지니어를 만날지 모를텐데, 미리 공부하고 가시는걸 추천한다. 수학, 통계학에 비해 그렇게 긴 시간을 투자해야할 만한 지식도 아니다.

 

(6. Python, R 코딩 능력)
안 쓰면 안 될 것 같아서 마지막에 추가하는 능력인데, 사실 그렇게 중요한지 잘 모르겠다. 위의 다른 능력들, 수학/통계학을 이용한 모델링 능력, 직관을 기르는 사고 훈련 등등이 잘 갖춰지기 위해서 어쩔 수 없이 손으로 뭔가 돌려봐야할텐데, 그런 코딩 언어는 세상에 참 많이도 있다.

연구실에 가보면 아직도 Matlab을 쓰시는 분들이 많을텐데, 버튼 하나로 작업 결과물을 Java, C로 변환해주고, 잘 모르는 사람들이 “딥러닝 개발자 = TensorFlow 경험자”라고 단순하게 생각하는 TensorFlow도 지원해준다. 연구실에서 더 희귀하게 보이지만 퀄리티 좋은 프로그램으로 Mathematica라는 것도 있다. 학교에서 그런 언어로 작업하셨던 분들이 직장가서 Python이나 R같은 비슷한 Script형 언어에 적응하는데 얼마나 긴 시간이 걸릴까? 1달 이상 걸리면 그 분의 연구 실적을 다시 확인해봐야 된다.

언어들이 다 비슷하기 때문에, 그냥 회사에서 많이 쓰는 언어에 맞추면 된다. Python을 많이 쓰면 거기에 맞추고, R 좋아하면 거기에 맞추고, 둘 다 쓰신다고들 하면 같이 맞춰주고, 속도 느려서 LLVM 언어들로 넘어가고 싶다면 Julia라는 대체재도 있다. Julia는 속도를 더 빠르게 해 주려고 언어 레벨에서 TensorFlow 컴파일링을 직접 지원한다. 아마 이렇게 돌려보면 Python 대비 10배 가까운 계산비용 절감을 경험할 수 있을 것이다. 사실 구글이 만든 TensorFlow는 C와 Java였다. Python이 아니라. 여담으로, 구글이 TensorFlow를 처음 내놓았을 때부터 지금까지 Python 버젼은 C 버젼이 Python에서 돌아갈 수 있도록 포팅해놓은 버젼이었던 탓에 가장 오류가 많이 생기는 버젼이었다. 그래픽 카드라는 시스템을 직접 제어해야하는데, 당연히 C로 프로그램을 짜야하지 않았을까?

다시 한번 강조하지만, 코딩을 잘하는 것과 개발을 잘하는 것은 1:1 동치 관계가 아니다. 학교 연구실에서도 코딩하고 있다는 사실, 그들의 아카데믹 코딩 중 일부가 일반에 알려진 머신러닝이라는 사실을 꼭 짚고 넘어가고 싶다.

정리하면, 코딩을 배울게 아니라, 기초가 되는 수학/통계학을 배우고, 그걸 내가 쓰는 코딩 언어로 어떻게 표현하는지를 익히면서 코딩 지식도 넓히고, 배우는 수학 모델에 대한 이해도를 조금씩 늘리는 방식이 맞는 접근법인 것 같다. (딱 대학원에서 이런 교육을 받지 않나?)

Gradient Boosting과 Random Forest의 결합이라는걸 바로 이해하게 

[패턴인식] KDE(Kernel density estimation, 커널밀도추정)
https://blog.naver.com/jamiet1/221392180461

feature importance : 

사이킷런의 feature importance는 각 feature별로 개별 노드의 Gini importance를 합산하여 계수화 한 것입니다.

feature별 Gini importance는 각 feature들이 사용된 트리 노드에서의 지니계수와 해당 노드의 가중치가 부여된 데이터 건수를 곱한 값을 구한 뒤 이 노드의 자식 노드인 왼쪽과 오른쪽 노드 각각의 지니계수와 가중치 부여한 데이터 건수를 곱한것을 마이너스 하여 계산합니다.  

아래는 이를 Pseudo 코드로 만든 것입니다.

해당 자료는 https://towardsdatascience.com/the-mathematics-of-decision-trees-random-forest-and-feature-importance-in-scikit-learn-and-spark-f2861df67e3 를 참조하였고, Pseudo 코드는 사이킷런의 tree 소스 코드를 참조하였습니다. 

Neural architecture transfer
채널 흐름에 따른 모델의 성능

# 최신 모델이다
HRNet, Res2Net, ResNest, efficientnet, ReXNet, SpineNet
