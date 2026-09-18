# mini_gpt — What This Is, Step by Step

> Read this like a note to your future self. It assumes you remember *nothing*
> about the code and re-derives the reasoning from scratch.

## 1. The one-sentence version

`gpt.py` builds a tiny **decoder-only Transformer language model** (the same
architecture family as GPT-2/GPT-3, just much smaller) completely from
scratch using raw PyTorch, trains it on ~1000 examples of Python
task-descriptions + code, and then asks it to continue a prompt one word at a
time. This follows the structure popularized by Andrej Karpathy's
"Let's build GPT" walkthrough, but swaps in a custom dataset and a
word-level tokenizer instead of the usual character-level Shakespeare demo.

The goal of this file is **not** to build a useful model — it's a learning
exercise to understand, line by line, how a GPT actually works internally
(embeddings → self-attention → feed-forward → stacked blocks → next-token
prediction → autoregressive sampling), by writing every piece by hand instead
of importing a library that hides it.

## 2. The data: `qwen25_coder_7b_small.jsonl`

Each line of this file is a JSON object with (at least) two fields:
- `task_description` — a natural-language description of a coding task
- `code` — a Python solution to that task

This looks like a distillation/fine-tuning dataset originally produced for
(or by) a code model (the filename references Qwen2.5-Coder-7B). It is
**not tracked in git** — it's ~130MB, far too large and not something you
want in version control, so it stays local-only (this is why an earlier
commit ("git") removed it after it had accidentally been committed once).

`gpt.py` only reads the **first 1000 lines** of this file (`examples[:1000]`)
and concatenates `task_description + "\n" + code + "\n\n"` for each one into
one giant string, `input_text`. That giant string is the entire "book" the
model is trained to read and predict, the same role the Shakespeare text
plays in the original tutorial. Limiting to 1000 examples keeps the
vocabulary and training loop small enough to run quickly on a laptop/CPU
while iterating on the code.

## 3. Tokenization — turning text into numbers

A neural net can't consume raw text; it needs integers. There are three
common ways to do this, in increasing order of sophistication:

| Approach | Granularity | Vocab size | Used here? |
|---|---|---|---|
| Character-level | single characters | tiny (~100) | No (this is what Karpathy's original demo uses) |
| Word-level | whole words/punctuation | large (thousands+) | **Yes — this is what `gpt.py` does** |
| Subword/BPE | learned chunks | tens of thousands | No (this is what real GPT models use) |

`gpt.py` implements a simple **word-level tokenizer** by hand:

```python
result = re.split(r'([,.:;?_!"()\']|--|\s)', input_text)
raw_text = [token.strip() for token in result if token.strip()]
```

This regex splits the text on whitespace and on punctuation characters,
*keeping the punctuation as its own tokens* (the parentheses around the
pattern in `re.split` mean "keep the delimiter in the output"). So
`"print(x)"` becomes `["print", "(", "x", ")"]` rather than one blob.

Then:
- `vocab = sorted(list(set(raw_text)))` — every unique word/symbol becomes
  one vocabulary entry, sorted for a deterministic ordering.
- `str_to_int` / `int_to_str` — a plain dictionary in each direction. This
  *is* the tokenizer: word → id → word.
- The `Tokenizer` class wraps `encode` (text → list of ids) and `decode`
  (list of ids → text, with a regex cleanup step that removes the stray
  space `decode` otherwise inserts before punctuation, e.g. turning
  `"code .` back into `"code."`).

**Why word-level instead of character-level or BPE?** It's a middle ground:
easier to implement than BPE, and produces much shorter token sequences than
character-level (so the model has to "see" fewer steps to span the same
amount of text). The trade-off, and the thing to remember if you revisit
this: the vocabulary is **fixed to whatever words appeared in the first 1000
training examples**. Any word not seen during training (including in your
own prompts at inference time) will raise a `KeyError` in `encode`, because
there's no "unknown token" fallback. This is a real limitation of this
implementation, not a bug to "fix later" — it's what you get from building
the simplest possible word-level tokenizer.

## 4. Turning text into training pairs

```python
data = torch.tensor(ids, dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
```

Standard 90/10 train/validation split on the token stream.

```python
def get_batch(split):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y
```

`block_size` (256) is the **context length** — how many tokens of history the
model looks at when predicting the next one. For each randomly chosen
starting position `i`, `x` is a chunk of 256 tokens and `y` is the *same*
chunk shifted one position to the right. This is the whole training signal:
**predict the next token, given all tokens before it.** `batch_size` (64)
just says how many such chunks are trained on simultaneously, for
efficiency.

(Note: `val_data` is computed but never actually used elsewhere in the
script — there's no validation-loss estimation loop, even though
`eval_interval`/`eval_iters` are defined at the top. If you come back to
extend this, that's the natural next thing to add, following the original
tutorial's `estimate_loss()` pattern.)

## 5. The model architecture

This is a from-scratch reimplementation of the Transformer decoder block
used in GPT-2. Built bottom-up:

### 5a. `Head` — one self-attention head

```python
k = self.key(x); q = self.query(x); v = self.value(x)
wei = q @ k.transpose(-2, -1)
wei = wei * (q.shape[-1] ** -0.5)          # scale
wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # causal mask
wei = F.softmax(wei, dim=-1)
out = wei @ v
```

Self-attention lets every token "look at" every other token in the sequence
and decide how much to weight each one when building its own updated
representation:
1. Every token produces a **query** (what am I looking for?), a **key**
   (what do I contain?), and a **value** (what do I offer, if picked?).
2. `q @ k.T` computes a similarity score between every pair of tokens.
3. Scaling by `1/sqrt(head_size)` keeps the softmax from saturating
   (standard "scaled dot-product attention" from the original Transformer
   paper — without it, large dot products push softmax into
   near-one-hot outputs and gradients vanish).
4. **Causal masking** (`tril`, a lower-triangular matrix of ones) is the
   part that makes this a *decoder* (GPT-style) rather than an *encoder*
   (BERT-style): token `t` is only allowed to attend to tokens `≤ t`, never
   to future tokens, by setting future positions' scores to `-inf` before
   softmax (so they become exactly 0 probability). This is what makes
   next-token prediction well-posed — without it the model could "cheat" by
   looking at the answer.
5. Softmax turns scores into a probability distribution ("attention
   weights"), and the output is a weighted average of every token's
   **value** vector.

`tril` is stored via `register_buffer` rather than as a parameter — it's a
constant, not something gradient descent should update, but it still needs
to move to the GPU with `.to(device)` and be saved/loaded with the model.

### 5b. `MultiheadAttention` — several heads in parallel

Instead of one attention computation, `n_head` (6) independent `Head`s run
in parallel, each with a smaller `head_size = n_embd // n_head` (384/6 = 64),
and their outputs are concatenated and linearly projected back to `n_embd`.
The intuition: one attention pattern might learn to track "which word does
this pronoun refer to" while another tracks "what's the matching closing
bracket" — multiple heads let the model learn several different relational
patterns simultaneously instead of averaging them all into one.

### 5c. `FeedForward` — per-token processing

```python
nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd)
```

After attention mixes *information between* tokens, the feed-forward
network processes *each token independently*, giving the model extra
non-linear capacity to transform what attention gathered. The 4x expansion
factor is the standard ratio from the original Transformer paper.

### 5d. `Block` — one Transformer layer

```python
x = x + self.attention(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

This is "pre-norm": LayerNorm is applied *before* each sub-layer, and the
sub-layer's output is *added* to its input (a residual/skip connection),
not used to replace it. Residual connections are what make it possible to
stack many layers (6 here) without training becoming unstable — gradients
have a direct path backward through the `+`, bypassing the transformation
entirely if needed.

### 5e. `GPTModel` — the full model

```python
tok_emb = self.embedding_table(idx)                       # what word is this?
pos_emb = self.positional_embedding_table(torch.arange(T)) # where is it in the sequence?
x = tok_emb + pos_emb
x = self.blocks(x)      # 6 stacked Transformer blocks
x = self.ln_f(x)        # final normalization
logits = self.lm_head(x)  # project to vocab-sized scores
```

Two embedding tables are summed together: a **token embedding** (a learned
vector per vocabulary word) and a **positional embedding** (a learned vector
per position 0..255). Attention alone has no notion of order — it treats
input as an unordered set — so position has to be injected explicitly. This
is why `block_size` is a hard limit: there's no positional embedding defined
beyond position 255, so the model *cannot* be run on a longer context
without changing `block_size` and retraining.

`logits` are unnormalized scores over the whole vocabulary for "what comes
next," for every position simultaneously. When `targets` are provided,
`F.cross_entropy` compares predicted logits against the actual next token
and produces the training loss — this single number is what
`loss.backward()` uses to update every weight in the network.

### 5f. `generate` — autoregressive sampling

```python
idx_cond = idx[:, -block_size:]     # only the last 256 tokens matter
logits, loss = self(idx_cond)
logits = logits[:, -1, :]           # only the prediction for the *next* token
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)  # sample, don't just argmax
idx = torch.cat((idx, idx_next), dim=1)
```

This is the inference-time loop: run the model, look only at its prediction
for the very next token, convert to a probability distribution, and
**sample** from it (rather than always taking the highest-probability word)
so that generations aren't perfectly deterministic/repetitive. The newly
generated token is appended and the whole thing repeats — each new token
becomes part of the input for predicting the one after it. This is why it's
called "autoregressive": the model conditions on its own past outputs.

## 6. Training loop and the actual run

```python
model = GPTModel(vocab_size, n_embd, block_size, n_layer)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for i in range(max_iters):          # 5000 iterations
    idx, targets = get_batch('train')
    logits, loss = model(idx, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

A bare-bones training loop: sample a random batch, compute loss, backprop,
update weights, repeat 5000 times. No loss printing/logging is currently
wired up during training (there's no `if i % eval_interval == 0: print(...)`
despite `eval_interval` being defined) — so as currently written, the script
runs silently for the whole training loop and you only find out how it did
from the final generated text.

After training, the model is put in `eval()` mode (disables dropout) and
given a hand-written prompt:

```python
prompt = "Count the number of strings in the input list  are entirely lowercase."
```

which is encoded, fed through `generate` for 20 new tokens, decoded back to
text, and printed.

**Expected outcome, honestly:** with only ~1000 short examples of training
text, a word-level vocabulary that size, and no logged validation loss to
check for overfitting, the generated continuation is very unlikely to read
as coherent code or English — the point of this script (as written) is to
watch the *machinery* work end-to-end (data → tokens → attention → loss →
generation), not to produce a usable coding assistant. Also note: `prompt`
must consist entirely of words that appeared in the training vocabulary, or
`tokenizer.encode(prompt)` will throw a `KeyError`.

## 7. Things to know if you come back to this in a year

These aren't bugs so much as "the script does exactly what's written, here's
what that implies":

- **`device` is defined but never used for anything.** `device = 'cuda' if
  torch.cuda.is_available() else 'cpu'` is computed, but the model and data
  tensors are never moved with `.to(device)`. This all currently runs on
  CPU (or whatever PyTorch's default device is) regardless of GPU
  availability.
- **No validation-loss tracking.** `eval_interval`/`eval_iters`/`val_data`
  are all defined, matching the structure of Karpathy's original tutorial,
  but the `estimate_loss()`-style function that would actually use them was
  never added. There's currently no way to tell from the script's output
  whether the model is over/under-fitting.
- **Fixed, closed vocabulary.** The tokenizer has no "unknown token"
  concept. It can only encode text made entirely of words it saw in the
  first 1000 training examples.
- **`qwen25_coder_7b_small.jsonl` is intentionally untracked** (too large
  for git). If you clone this repo fresh on another machine, `gpt.py` will
  fail at the `open(...)` call until that file is present locally again.
- **Only the first 1000 of the dataset's lines are used**, even though the
  full file is much larger — this was a deliberate speed/scope trade-off
  for a from-scratch learning exercise, not a data limitation.

## 8. The mental model to keep

If everything else here is forgotten, keep this: a GPT is a stack of blocks
that each (1) let every token gather relevant information from earlier
tokens via masked self-attention, then (2) transform that per token via a
small MLP, with residual connections holding it all together — trained on
nothing more than "predict the next token" and then run repeatedly,
feeding its own output back in, to generate new text one token at a time.
Every other detail in `gpt.py` (embeddings, layer norm, dropout, multiple
heads, the 4x feed-forward expansion) is a refinement layered on top of that
one idea.
