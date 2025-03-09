from torch.utils.data import Dataset, DataLoader
import tiktoken


def checkresult():
    tokenizer = tiktoken.get_encoding("gpt2")

    # 모델을 정의할 때 사용하는 상수들

    VOCAB_SIZE = tokenizer.n_vocab  # 50257 Tiktoken
    # VOCAB_SIZE = len(tokenizer) # AutoTokenizer
    CONTEXT_LENGTH = 128  # Shortened context length (orig: 1024)
    EMB_DIM = 768  # Embedding dimension
    NUM_HEADS = 12  # Number of attention heads
    NUM_LAYERS = 12  # Number of layers
    DROP_RATE = 0.1  # Dropout rate
    QKV_BIAS = False  # Query-key-value bias

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

    import torch.nn as nn

    class MultiHeadAttention(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__()

            assert d_out % NUM_HEADS == 0, "d_out must be divisible by n_heads"

            self.d_out = d_out
            self.head_dim = d_out // NUM_HEADS

            self.W_query = nn.Linear(d_in, d_out, bias=QKV_BIAS)
            self.W_key = nn.Linear(d_in, d_out, bias=QKV_BIAS)
            self.W_value = nn.Linear(d_in, d_out, bias=QKV_BIAS)
            self.out_proj = nn.Linear(d_out, d_out)
            self.dropout = nn.Dropout(DROP_RATE)
            self.register_buffer('mask', torch.triu(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH), diagonal=1))

        def forward(self, x):
            b, num_tokens, d_in = x.shape

            keys = self.W_key(x)  # (b, num_tokens, d_out)
            queries = self.W_query(x)
            values = self.W_value(x)

            keys = keys.view(b, num_tokens, NUM_HEADS, self.head_dim)
            values = values.view(b, num_tokens, NUM_HEADS, self.head_dim)
            queries = queries.view(b, num_tokens, NUM_HEADS, self.head_dim)

            keys = keys.transpose(1, 2)
            queries = queries.transpose(1, 2)
            values = values.transpose(1, 2)

            attn_scores = queries @ keys.transpose(2, 3)

            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

            attn_scores.masked_fill_(mask_bool, -torch.inf)

            attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)

            context_vec = (attn_weights @ values).transpose(1, 2)

            context_vec = context_vec.reshape(b, num_tokens, self.d_out)
            context_vec = self.out_proj(context_vec)

            return context_vec

    class LayerNorm(nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.eps = 1e-5
            self.scale = nn.Parameter(torch.ones(emb_dim))
            self.shift = nn.Parameter(torch.zeros(emb_dim))

        def forward(self, x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            norm_x = (x - mean) / torch.sqrt(var + self.eps)
            return self.scale * norm_x + self.shift

    class GELU(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return 0.5 * x * (1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                (x + 0.044715 * torch.pow(x, 3))
            ))

    class FeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(EMB_DIM, 4 * EMB_DIM),
                GELU(),
                nn.Linear(4 * EMB_DIM, EMB_DIM),
            )

        def forward(self, x):
            return self.layers(x)

    class TransformerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.att = MultiHeadAttention(
                d_in=EMB_DIM,
                d_out=EMB_DIM)

            self.ff = FeedForward()
            self.norm1 = LayerNorm(EMB_DIM)
            self.norm2 = LayerNorm(EMB_DIM)
            self.drop_shortcut = nn.Dropout(DROP_RATE)

        def forward(self, x):
            shortcut = x
            x = self.norm1(x)
            x = self.att(x)
            x = self.drop_shortcut(x)
            x = x + shortcut

            shortcut = x
            x = self.norm2(x)
            x = self.ff(x)
            x = self.drop_shortcut(x)
            x = x + shortcut

            return x

    class GPTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(VOCAB_SIZE, EMB_DIM)
            self.pos_emb = nn.Embedding(CONTEXT_LENGTH, EMB_DIM)
            self.drop_emb = nn.Dropout(DROP_RATE)

            self.trf_blocks = nn.Sequential(
                *[TransformerBlock() for _ in range(NUM_LAYERS)])

            self.final_norm = LayerNorm(EMB_DIM)
            self.out_head = nn.Linear(EMB_DIM, VOCAB_SIZE, bias=False)

        def forward(self, in_idx):
            batch_size, seq_len = in_idx.shape
            tok_embeds = self.tok_emb(in_idx)
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
            x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
            x = self.drop_emb(x)
            x = self.trf_blocks(x)
            x = self.final_norm(x)
            logits = self.out_head(x)
            return logits

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(device)

    torch.manual_seed(123)
    model = GPTModel()
    model.to(device)


    # 파일로 저장했던 네트워크의 가중치들 읽어들이기
    model.load_state_dict(torch.load("model_010.pth", map_location=device, weights_only=True))
    model.eval() # dropout을 사용하지 않음

    idx = tokenizer.encode("harry potter is") # 토큰 id의 list
    idx = torch.tensor(idx).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(idx)

    logits = logits[:, -1, :]  #가장 마지막 토큰(harry potter is) 이후에 붙은 단어 추출

    # 가장 확률이 높은 단어 10개 출력
    top_logits, top_indices = torch.topk(logits, 10)
    for p, i in zip(top_logits.squeeze(0).tolist(), top_indices.squeeze(0).tolist()):
        print(f"{p:.2f}\t {i}\t {tokenizer.decode([i])}")

    # 가장 확률이 높은 단어 출력
    idx_next = torch.argmax(logits, dim=-1, keepdim=True)
    flat = idx_next.squeeze(0) # 배치 차원 제거 torch.Size([1])
    out = tokenizer.decode(flat.tolist()) # 텐서를 리스트로 바꿔서 디코드
    print(out)



import torch

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 파일로 저장했던 네트워크의 가중치들 읽어들이기
    model.load_state_dict(torch.load("model_010.pth", map_location=device, weights_only=True))
    model.eval() # dropout을 사용하지 않음


    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        if temperature > 0.0:
            # temperature 값을 키우면 예상되는 토큰에 난수값을 더해서 출력하기에 출력되는 값이 고정되지않고 변동 될 수 있다.
            # 장점 - 다양한 답변 기대 가능 / 단점 - 정확도가 떨어짐
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx



def generatechk():
    tokenizer = tiktoken.get_encoding("gpt2")

    # 모델을 정의할 때 사용하는 상수들

    VOCAB_SIZE = tokenizer.n_vocab  # 50257 Tiktoken
    # VOCAB_SIZE = len(tokenizer) # AutoTokenizer
    CONTEXT_LENGTH = 128  # Shortened context length (orig: 1024)
    EMB_DIM = 768  # Embedding dimension
    NUM_HEADS = 12  # Number of attention heads
    NUM_LAYERS = 12  # Number of layers
    DROP_RATE = 0.1  # Dropout rate
    QKV_BIAS = False  # Query-key-value bias

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

    import torch.nn as nn

    class MultiHeadAttention(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__()

            assert d_out % NUM_HEADS == 0, "d_out must be divisible by n_heads"

            self.d_out = d_out
            self.head_dim = d_out // NUM_HEADS

            self.W_query = nn.Linear(d_in, d_out, bias=QKV_BIAS)
            self.W_key = nn.Linear(d_in, d_out, bias=QKV_BIAS)
            self.W_value = nn.Linear(d_in, d_out, bias=QKV_BIAS)
            self.out_proj = nn.Linear(d_out, d_out)
            self.dropout = nn.Dropout(DROP_RATE)
            self.register_buffer('mask', torch.triu(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH), diagonal=1))

        def forward(self, x):
            b, num_tokens, d_in = x.shape

            keys = self.W_key(x)  # (b, num_tokens, d_out)
            queries = self.W_query(x)
            values = self.W_value(x)

            keys = keys.view(b, num_tokens, NUM_HEADS, self.head_dim)
            values = values.view(b, num_tokens, NUM_HEADS, self.head_dim)
            queries = queries.view(b, num_tokens, NUM_HEADS, self.head_dim)

            keys = keys.transpose(1, 2)
            queries = queries.transpose(1, 2)
            values = values.transpose(1, 2)

            attn_scores = queries @ keys.transpose(2, 3)

            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

            attn_scores.masked_fill_(mask_bool, -torch.inf)

            attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)

            context_vec = (attn_weights @ values).transpose(1, 2)

            context_vec = context_vec.reshape(b, num_tokens, self.d_out)
            context_vec = self.out_proj(context_vec)

            return context_vec

    class LayerNorm(nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.eps = 1e-5
            self.scale = nn.Parameter(torch.ones(emb_dim))
            self.shift = nn.Parameter(torch.zeros(emb_dim))

        def forward(self, x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            norm_x = (x - mean) / torch.sqrt(var + self.eps)
            return self.scale * norm_x + self.shift

    class GELU(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return 0.5 * x * (1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                (x + 0.044715 * torch.pow(x, 3))
            ))

    class FeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(EMB_DIM, 4 * EMB_DIM),
                GELU(),
                nn.Linear(4 * EMB_DIM, EMB_DIM),
            )

        def forward(self, x):
            return self.layers(x)

    class TransformerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.att = MultiHeadAttention(
                d_in=EMB_DIM,
                d_out=EMB_DIM)

            self.ff = FeedForward()
            self.norm1 = LayerNorm(EMB_DIM)
            self.norm2 = LayerNorm(EMB_DIM)
            self.drop_shortcut = nn.Dropout(DROP_RATE)

        def forward(self, x):
            shortcut = x
            x = self.norm1(x)
            x = self.att(x)
            x = self.drop_shortcut(x)
            x = x + shortcut

            shortcut = x
            x = self.norm2(x)
            x = self.ff(x)
            x = self.drop_shortcut(x)
            x = x + shortcut

            return x

    class GPTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(VOCAB_SIZE, EMB_DIM)
            self.pos_emb = nn.Embedding(CONTEXT_LENGTH, EMB_DIM)
            self.drop_emb = nn.Dropout(DROP_RATE)

            self.trf_blocks = nn.Sequential(
                *[TransformerBlock() for _ in range(NUM_LAYERS)])

            self.final_norm = LayerNorm(EMB_DIM)
            self.out_head = nn.Linear(EMB_DIM, VOCAB_SIZE, bias=False)

        def forward(self, in_idx):
            batch_size, seq_len = in_idx.shape
            tok_embeds = self.tok_emb(in_idx)
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
            x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
            x = self.drop_emb(x)
            x = self.trf_blocks(x)
            x = self.final_norm(x)
            logits = self.out_head(x)
            return logits

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(device)

    torch.manual_seed(123)
    model = GPTModel()
    model.to(device)

    # 파일로 저장했던 네트워크의 가중치들 읽어들이기
    model.load_state_dict(torch.load("model_010.pth", map_location=device, weights_only=True))
    model.eval() # dropout을 사용하지 않음

    # start_context = input("Start context: harry potter is ")

    idx = tokenizer.encode("harry potter is") # 토큰 id의 list
    # idx = tokenizer.encode(start_context)
    idx = torch.tensor(idx).unsqueeze(0)

    context_size = model.pos_emb.weight.shape[0]

    for i in range(10):

        token_ids = generate(
            model=model,
            idx=idx.to(device),
            max_new_tokens=10,
            context_size= context_size,
            top_k=10,
            temperature=0.5
        )

        flat = token_ids.squeeze(0) # remove batch dimension
        out = tokenizer.decode(flat.tolist()).replace("\n", " ")

        print(i, ":", out)