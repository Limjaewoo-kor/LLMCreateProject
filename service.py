import re

def clean_textService():
    def clean_text(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            book_text = file.read()

        cleaned_text = re.sub(r'\n+', ' ', book_text)  # 줄바꿈을 빈칸으로 변경
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # 여러 빈칸을 하나의 빈칸으로

        print("cleaned_" + filename, len(cleaned_text), "characters")  # 글자 수 출력

        with open("cleaned_" + filename, 'w', encoding='utf-8') as file:
            file.write(cleaned_text)


    filenames_list = ["02 Harry Potter and the Chamber of Secrets.txt"]

    for filename in filenames_list:
        clean_text(filename)



import tiktoken  # pip install tiktoken

def tokenizer():
    tokenizer = tiktoken.get_encoding("gpt2")
    text = "Harry Potter was a wizard."

    tokens = tokenizer.encode(text)
    print("글자수:", len(text), "토큰수", len(tokens))
    print(tokens)
    print(tokenizer.decode(tokens))
    for t in tokens:
        print(f"{t}\t -> {tokenizer.decode([t])}")




import torch
from torch.utils.data import Dataset, DataLoader

def dataloder():

    class MyDataset(Dataset):
        def __init__(self, txt, max_length, stride):
            self.input_ids = []
            self.target_ids = []

            # token_ids = tokenizer.encode("<|endoftext|>" + txt, allowed_special={"<|endoftext|>"})
            token_ids = tokenizer.encode(txt)

            print("# of tokens in txt:", len(token_ids))

            for i in range(0, len(token_ids) - max_length, stride):
                input_chunk = token_ids[i:i + max_length]
                target_chunk = token_ids[i + 1: i + max_length + 1]
                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.target_ids[idx]

    # with open("cleaned_한글문서.txt", 'r', encoding='utf-8-sig') as file: # 선택: -sig를 붙여서 BOM 제거
    with open("cleaned_02 Harry Potter and the Chamber of Secrets.txt", 'r', encoding='utf-8-sig') as file: # 선택: -sig를 붙여서 BOM 제거
        txt = file.read()

    dataset = MyDataset(txt, max_length = 32, stride = 4)

    train_loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)

    dataiter = iter(train_loader)

    x, y = next(dataiter)

    print(tokenizer.decode(x[0].tolist()))
    print(tokenizer.decode(y[0].tolist()))

    # 주의: 여기서는 코드를 단순화하기 위해 test, valid는 생략하고 train_loader만 만들었습니다.
    #      관련된 ML 이론이 궁금하신 분들은 train vs test vs validation 등으로 검색해보세요.