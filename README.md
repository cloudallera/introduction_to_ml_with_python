# 파이썬 라이브러리를 활용한 머신러닝
#### (직접 구현하면서 배우는 본격 머신러닝 입문서)

### **mystudy 폴더에 내 노트북에서 스터디 한 코드를 추가함**

이 레파지토리는 안드레아스 뮐러(Andreas Mueller)와 사라 귀도(Sarah Guido)의 책인 "Introduction to Machine
Learning with Python"의 번역서 "[파이썬 라이브러리를 활용한 머신러닝](https://tensorflow.blog/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%9D%BC%EC%9D%B4%EB%B8%8C%EB%9F%AC%EB%A6%AC%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/)" 책의 코드를 담고 있습니다.
이 레파지토리는 원서의 레파지토리를 포크한 것이며 최근 수정 사항을 반영하고 주석을 한글로 번역하였습니다.
이 책에 대한 자세한 사항은 옮긴이의 [블로그](https://tensorflow.blog/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%9D%BC%EC%9D%B4%EB%B8%8C%EB%9F%AC%EB%A6%AC%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/)나 한빛미디어 [웹사이트](http://www.hanbit.co.kr/store/books/look.php?p_code=B6119391002)에서 확인할 수 있습니다.

이 책의 내용은 scikit-learn 0.18, 0.19, 0.20, 0.20.1에서 테스트 되었습니다.

이 레파지토리는 책에 포함된 코드를 주피터 노트북 형태로 가지고 있으며 그래프와 데이터셋을 위한 ``mglearn`` 라이브러리를 함께 제공합니다.
aclImdb 데이터셋과 Naver sentiment movie corpus를 제외하고는 책에서 사용하는 데이터도 모두 포함하고 있습니다.
aclImdb 데이터셋은 앤드류 마스(Anrew Mass)의 [웹사이트](http://ai.stanford.edu/~amaas/data/sentiment/)에서 다운받을 수 있습니다. 자세한 내용은 책을 참고하세요.
Naver sentiment movie corpus는 루시님의 [깃허브](https://github.com/e9t/nsmc/)에서 다운받을 수 있습니다.

책 커버에 있는 도룡뇽은 [헬벤더](https://ko.wikipedia.org/wiki/%ED%97%AC%EB%B2%A4%EB%8D%94)입니다.

## 에러타(Errata)

"[파이썬 라이브러리를 활용한 머신러닝](https://tensorflow.blog/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%9D%BC%EC%9D%B4%EB%B8%8C%EB%9F%AC%EB%A6%AC%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/)"의 에러타는 옮긴이의 [블로그](https://tensorflow.blog/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%9D%BC%EC%9D%B4%EB%B8%8C%EB%9F%AC%EB%A6%AC%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/)나 한빛미디어 [웹사이트](http://www.hanbit.co.kr/store/books/look.php?p_code=B6119391002)에서 확인할 수 있습니다. 코드에 오류가 있다면 깃허브에 이슈를 남겨 주시거나 옮긴이의 [블로그](https://tensorflow.blog/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%9D%BC%EC%9D%B4%EB%B8%8C%EB%9F%AC%EB%A6%AC%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/)를 통해 연락 주세요.

## 설치

이 코드를 실행하려면 ``numpy``, ``scipy``, ``scikit-learn``, ``matplotlib``, ``pandas``와 ``pillow`` 패키지가 필요합니다.
결정 트리와 신경망 구조에 대한 그래프를 그리려면 ``graphviz``도 필요합니다.

개발 환경을 만들려면 [아나콘다](https://www.continuum.io/downloads)(Anaconda)를 설치하는 것이 가장 편리한 방법입니다.

### conda를 사용한 패키지 설치

설치된 파이썬이 있다면 ``conda`` 패키지 매니저를 사용하여 다음 명령을 실행하면 필요한 패키지를 모두 얻을 수 있습니다.

    conda install numpy scipy scikit-learn matplotlib pandas pillow graphviz python-graphviz

### pip를 사용한 패키지 설치

파이썬이 있고 pip를 사용하여 패키지를 설치하려면 다음 명령을 사용합니다.

    pip install numpy scipy scikit-learn matplotlib pandas pillow graphviz

또한 graphviz C 라이브러리를 설치해야 합니다. 패키지 매니저를 사용하여 쉽게 설치할 수 있으며 macOS는 homebrew를 사용하여 ``brew install graphviz`` 명령을 사용합니다. 우분투나 데비안이라면 ``apt-get install graphviz`` 명령을 사용합니다. 윈도우즈에서 graphviz를 설치하는 것은 쉽지 않습니다. 대신 conda나 아나콘다를 사용하세요.

![cover](cover.jpg)
