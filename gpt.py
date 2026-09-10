import re
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)


import json

examples = []

with open("qwen25_coder_7b_small.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        examples.append(json.loads(line))

input_text = ""

for example in examples[:1000]:
    input_text += example["task_description"]
    input_text += "\n"
    input_text += example["code"]
    input_text += "\n\n"

result = re.split(r'([,.:;?_!"()\']|--|\s)', input_text)
raw_text = [token.strip() for token in result if token.strip()]

vocab = sorted(list(set(raw_text)))
vocab_size = len(vocab)
print(vocab_size)
print(vocab)

str_to_int = {token: i for i, token in enumerate(vocab)}
print(str_to_int)
int_to_str = {i: token for token, i in str_to_int.items()}
print(int_to_str)

#Tokenization of all unique words in the given the given text and creating a dictionary of word to index and index to word mapping.
class Tokenizer:
    def __init__(self, str_to_int, int_to_str):
        self.str_to_int = str_to_int
        self.int_to_str = int_to_str

    def encode(self, txt):
        text = re.split(r'([,.:;?_!"()\']|--|\s)', txt)
        text = [item.strip() for item in text if item.strip()]
        ids = [self.str_to_int[token] for token in text]

        return ids

    def decode(self, ids):
        text = " ".join(
            [self.int_to_str[id] for id in ids]
        )  # Converts token IDs back into text
        text = re.sub(
            r'\s+([,.?!"()\'])', r"\1", text
        )  # Removes space before the specified punctuation
        return text

tokenizer = Tokenizer(str_to_int, int_to_str)
ids = tokenizer.encode(input_text)
data = torch.tensor(ids, dtype = torch.long)

# Prepare train and test data
n = int(0.9 * len(data))
train_data = data[:n]
print(len(train_data))
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    # print(x)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # print(y)

    return x, y

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size)
        self.query = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1)
        # Scale using head_size
        wei = wei * (q.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiheadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))

        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embd, n_head):

        super().__init__()
        head_size = n_embd // n_head
        self.attention = MultiheadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


class GPTModel(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size, n_layers):

        super().__init__()

        self.embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.embedding_table(idx)
        pos_emb = self.positional_embedding_table(torch.arange(T))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = GPTModel(vocab_size, n_embd, block_size, n_layer)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for i in range(max_iters):
    idx, targets = get_batch('train')
    logits, loss = model(idx, targets)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


model.eval()

prompt = "Count the number of strings in the input list  are entirely lowercase."

prompt_ids = tokenizer.encode(prompt)

idx = torch.tensor(
    [prompt_ids],
    dtype=torch.long
)

generated = model.generate(
    idx,
    max_new_tokens=20
)

result = tokenizer.decode(
    generated[0].tolist()
)

print(result)
