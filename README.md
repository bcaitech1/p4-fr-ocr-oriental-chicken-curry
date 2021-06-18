# Pstage 4 ] OCR

## 📋 Table of content

- [팀 소개](#Team)<br>
- [최종 결과](#Result)<br>
- [대회 개요](#Overview)<br>
- [문제 정의 해결 및 방법](#Solution)<br>
- [CODE 설명](#Code)<br>

<br></br>
## 👋 팀 소개 <a name = 'Team'></a>

- OCR 7조 **oriental-chicken-curry**
- 조원 : 김진현, 김홍엽, 김효진, 김희섭, 박성배

|                                                                                      김진현                                                                                      |                                                            김홍엽                                                             |                                                          김효진                                                           |                                                            김희섭                                                           |                                                            박성배                                                                                                                                 
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------: 
| <a href='https://github.com/openingsound'><img src='https://avatars.githubusercontent.com/u/65082579?v=4' width='200px'/></a> | <a href='https://github.com/MaiHon'><img src='https://avatars.githubusercontent.com/u/41847456?v=4' width='200px'/></a> | <a href='https://github.com/vim-hjk'><img src='https://avatars.githubusercontent.com/u/77153072?v=4' width='200px'/></a> | <a href='https://github.com/gan-ta'><img src='https://avatars.githubusercontent.com/u/51118441?v=4' width='200px'/></a>  | <a href='https://github.com/songbae'><img src='https://avatars.githubusercontent.com/u/65913073?v=4' width='200px'/></a>

<br></br>
## 🎖 최종 결과 <a name = 'Result'></a>
- Ranking : 2/12
- Score
    - Public : 0.8170
    - Private : 0.6065
    

<br></br>
##  대회 개요 <a name = 'Overview'></a>
<img src='https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F5602706%2F67bf0c680286baf2c979c8207a991bb2%2FScreen%20Shot%202020-08-19%20at%201.02.50%20PM.png?generation=1597868629120369&alt=media' width=800 height=300/>

<br></br>
수식 이미지를 latex 포멧의 text로 변환하는 문제입니다. 수식 인식의 경우는, 기존의 광학 문자 인식과는 달리 multi line recogintion을 필요로 합니다.<br><br>
기존 single line recognition 기반의 OCR이 아닌 multi line recognition을 이용하는 기존 OCR과는 차별화되는 Task입니다.



- 평가방법 
    - **`sentence accuracy`**, **`wer`**
    - **`score : sentence accuracy * 0.9  + wer * 0.1`**

<br></br>
## 📝 문제 정의 및 해결 방법 <a name = 'Solution'></a>

- 해당 대회에 대한 문제 정의, 해결 방법, 웹 서빙 등의 내용은 [여기](https://www.notion.so/OCR-07-d55776948a91481e9e5589a4956d163c)서 자세하게 확인 해 보실 수 있습니다.<br>

-  협업 관련  내용은  [여기](https://www.notion.so/4ff1baeb5d2e44f88a1e1c8dff158db5?v=ce487a62011b4171b4e54c6b591b2029)서 확인 할 수 있습니다
<br></br>
## 💻 CODE 설명<a name = 'Code'></a>
~~~
├── README.md
├── configs           # yaml -> 파라미터 수정 
├── data
├── data_tools
├── inference&practice
├── inference.py      # 모델 추론
├── log.py
├── networks          # SATRN , SRN 등 OCR 모델
├── requirements.txt  
├── train.py          # 학습 코드
├── unit_test.py      # test 코드
└── utils             # 그 외 유틸 코드
~~~

- Train & Test code
```
python code/train.py

python code/inference.py
```

