from fastapi import FastAPI

from checkService import checkresult, generatechk
from service import clean_textService, tokenizer,dataloder
from trainService import traindata

app = FastAPI()

# 해당 자료는 홍정모 연구소의 HongLabLLM을 참고하였으며,
# 개인 공부 및 pycharm으로 실행 가능하도록 메서드 및 환경을 재구성하였습니다.

# 자동회귀(autoregressive) LLM의 예제
# 디퓨전(Diffusion) LLM 기술도 나오기 시작함 , 한번에 한 단어씩이 아니라 전체를 생성함.

@app.get("/")
async def root():
    return {"message":"Hello World"}


@app.get("/clean_textService")
async def cleantext():
    clean_textService()  # 문자열 데이터 전처리
    return {"message": "clean_textService"}

@app.get("/tokenizer")
async def token():
    tokenizer() # 토큰이 인식하는 방식
    return {"message": "tokenizer"}

@app.get("/dataLoader")
async def loder():
    dataloder() # input_text와 output_text의 방식 체크
    return {"message": "dataLoader"}

@app.get("/traindata")
async def train():
    traindata() # Transformer 모델을 구성하는 핵심 모듈들의 구현 코드
    return {"message": "traindata"}


@app.get("/checkresult")
async def check():
    checkresult() # 결과체크
    return {"message": "checkresult"}

@app.get("/generatechk")
async def generatecheck():
    generatechk() # 결과체크
    return {"message": "generatechk"}

