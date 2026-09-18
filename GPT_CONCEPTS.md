# GPT Internals — The Engineering Concepts Behind `gpt.py`

> This is a concept reference, not a code walkthrough (see `WALKTHROUGH.md`
> for that). Read this to re-learn *how a GPT actually works* — the ideas
> here are general to every GPT-style model, not just this toy one. Each
> section says what the concept is, why it exists (what problem it solves),
> and where to see it in `gpt.py` if you want to look at the real
> implementation.

---

## 1. The core idea: language modeling as next-token prediction

Everything in a GPT serves one objective: **given some tokens, predict the
probability distribution of the next token.** That's it — there is no other
task. A model that does this well, applied repeatedly (feed its own
output back in as new input), can generate arbitrarily long, coherent text.

This is called **autoregressive** modeling — "auto" (self) + "regressive"
(regressing on/predicting from) — because each prediction is conditioned on
the sequence generated so far, including the model's own previous outputs.

Mathematically, a GPT learns:

```
P(token_t | token_1, token_2, ..., token_{t-1})
```

for every position `t`, and training just means adjusting weights so this
predicted probability is high for whatever token *actually* came next in
real text. In `gpt.py`, this shows up as the loss computed in
`GPTModel.forward` (`F.cross_entropy(logits, targets)`) — nothing more
exotic than "how surprised was the model by the real next word?"

---

## 2. Tokenization — text is not numbers

Neural networks operate on numbers (tensors), not strings. **Tokenization**
is the process of converting text into a sequence of integers from a fixed
vocabulary (called **encoding**), and converting integers back into text
(**decoding**). It happens in two phases that are easy to conflate:

1. **Vocabulary construction** (done once, ahead of time): decide the fixed
   list of tokens the model will ever know about, and assign each one an
   integer id. In `gpt.py` this is `vocab = sorted(list(set(raw_text)))`
   plus the `str_to_int`/`int_to_str` dictionaries — built once from the
   training corpus, then frozen.
2. **Encoding/decoding** (done every time text goes in or out): apply that
   fixed vocabulary to turn new text into ids, and ids back into text. This
   is the `Tokenizer.encode`/`decode` methods.

Three common strategies for step 1, in order of how real-world GPTs evolved:

- **Character-level**: each character is a token. Tiny vocabulary (~100),
  but sequences get very long (a 10-word sentence might be 60 tokens).
- **Word-level**: each word/punctuation mark is a token (what `gpt.py`
  does). Shorter sequences, but the vocabulary is large and, critically,
  **fixed** — any word not seen during vocabulary construction has no valid
  token id (an "out-of-vocabulary" problem).
- **Subword tokenization (e.g. BPE — Byte-Pair Encoding)**: what real GPT
  models (GPT-2, GPT-3, GPT-4, etc.) actually use. It learns a vocabulary of
  frequently-occurring *chunks* of characters, so common words are one
  token, rare/unseen words fall back to smaller pieces (or even individual
  bytes), meaning there's **never** an out-of-vocabulary word. This is the
  single biggest practical difference between `gpt.py`'s tokenizer and a
  production GPT tokenizer.

### 2.1 How BPE actually builds its vocabulary (what `gpt.py` skips)

Byte-Pair Encoding starts from the smallest possible units (individual
bytes/characters) and *greedily merges* the most frequent adjacent pair,
repeating until a target vocabulary size is reached:

1. Start with every word split into individual characters/bytes, e.g.
   `"lower"` → `l o w e r`.
2. Count every adjacent pair across the whole corpus (`l`+`o`, `o`+`w`, …).
3. Merge the single most frequent pair into one new token everywhere it
   occurs (e.g. if `e`+`r` is most common, every `e r` becomes `er`).
4. Repeat steps 2–3 thousands of times (GPT-2's tokenizer does this until it
   has ~50,000 tokens).

The result is a vocabulary of variable-length chunks: extremely common
words end up as a single token, rarer words get split into a couple of
recognizable pieces (`"tokenization"` might become `token` + `ization`),
and anything truly novel can always fall back to individual bytes — which is
*why* there's no out-of-vocabulary problem: the vocabulary's smallest units
(bytes) can represent absolutely any string. `gpt.py`'s vocabulary
construction (unique whole words, sorted) is the simple case this
algorithm generalizes.

### 2.2 Special tokens (also skipped here)

Production tokenizers reserve a handful of ids for control purposes rather
than actual text, commonly:

- **`<BOS>`/`<EOS>`** (beginning/end of sequence) — lets the model learn
  where a document starts/stops, rather than every document blending into
  the next.
- **`<PAD>`** (padding) — when batching sequences of different lengths
  together, shorter ones are padded up to the longest one in the batch so
  they fit in one tensor; an accompanying **attention mask** tells the
  model to ignore pad positions entirely (they're not real content).
- **`<UNK>`** (unknown) — a fallback id for anything outside the vocabulary
  (mostly obsolete with byte-level BPE, since bytes can encode anything).

`gpt.py` has none of these: every one of its ~1000 training examples is
just concatenated into one continuous stream with no boundary markers, all
batches are fixed-length `block_size` slices (so no padding is ever
needed), and there's no `<UNK>` (an unseen word just crashes `encode` with
a `KeyError` instead of degrading gracefully).

Whatever the strategy, the result is always the same two lookup tables: a
mapping from token → integer id (used to *encode* input) and integer id →
token (used to *decode* output). This is the entire job of the `Tokenizer`
class in `gpt.py`.

---

## 3. Vectorization & embeddings — turning token ids into meaning

**Vectorization** is the general machine-learning term for converting raw
data (text, images, categories — anything) into numeric vectors a model can
compute with. Tokenization produces *ids* (single integers); vectorization
is the next step that turns each id into an actual vector of numbers. It
helps to see the naive version first, to appreciate why GPTs don't use it:

- **One-hot vectorization**: represent token id `47` (out of, say, 10,000)
  as a 10,000-length vector of all zeros except a single `1` at position
  47. This trivially turns ids into vectors, but every vector is equally
  "far" from every other (no notion of similarity), the vectors are huge
  and mostly wasted zeros, and none of it is learned.
- **Dense/learned embeddings** (what every real GPT — and `gpt.py` — uses
  instead): a much shorter vector (384 numbers in `gpt.py`, vs. a
  vocab-sized one-hot vector) *per token*, stored in a lookup table
  (`nn.Embedding`) whose values are ordinary trainable parameters, updated
  by backpropagation exactly like every other weight in the network.
  Because they're learned from data, tokens that behave similarly in
  context (e.g. `king`/`queen`, or `+`/`-` in code) naturally end up with
  similar vectors — nearby points in a 384-dimensional space — purely as a
  side effect of training on the next-token-prediction objective, with no
  explicit instruction to do so. This is the origin of the oft-repeated
  claim that embeddings "capture meaning": similarity in vector space
  (commonly measured via cosine similarity) ends up correlating with
  similarity in usage/meaning.

Concretely, an **embedding table** is just a matrix of shape
`(vocab_size, n_embd)` — one learned row per vocabulary entry — and
"embedding a token" is nothing more exotic than *reading out row `id`*.
`gpt.py` looks this up via `self.embedding_table(idx)`, which is exactly
equivalent to one-hot-encoding `idx` and multiplying by the embedding
matrix, just done as a fast lookup instead of a wasteful matrix multiply.

Two separate embedding tables are used in `gpt.py` and simply
**added together**:

- **Token embedding** — "what word is this?" (`embedding_table`)
- **Positional embedding** — "where does it sit in the sequence?"
  (`positional_embedding_table`)

The positional embedding exists because of a subtle fact covered next:
**attention itself has no concept of order.**

---

## 4. Self-attention — the mechanism that makes Transformers work

### 4.1 The problem it solves

Older architectures (RNNs) processed a sequence one token at a time, in
order, carrying a "hidden state" forward — which made order-awareness free
but made long-range dependencies hard (information has to survive many
sequential steps) and training slow (can't parallelize across time).

**Self-attention** instead lets every token directly look at every other
token in one step, and learn *how much* to weight each one — no
sequential bottleneck, and long-range relationships are just as easy to
learn as short-range ones.

### 4.2 Query, Key, Value

For each token, three vectors are computed via three separate learned
linear layers:

- **Query (Q)** — "what am I looking for from other tokens?"
- **Key (K)** — "what do I have to offer, as a label?"
- **Value (V)** — "what do I actually contribute, if picked?"

The analogy: think of a lookup in a dictionary/database. Your **query** is
what you search for; every entry has a **key** you compare your search
against; the entry's **value** is what you get back if it matches. Attention
does this "softly" — every entry contributes *some* amount, weighted by how
well its key matches the query, instead of returning just one exact match.

### 4.3 Scaled dot-product attention

```
attention_weights = softmax( (Q · Kᵀ) / sqrt(head_size) )
output = attention_weights · V
```

- `Q · Kᵀ` computes a raw similarity score between every pair of tokens
  (dot product = how aligned two vectors are).
- Dividing by `sqrt(head_size)` (the **scaling** in "scaled dot-product
  attention") keeps these scores from growing too large as the vector
  dimension grows. Without it, softmax on large values saturates into
  near one-hot outputs, which starves most tokens of gradient signal during
  training and makes learning unstable. This one line is why the paper that
  introduced this mechanism is literally titled *"Attention Is All You
  Need."*
- `softmax` turns the scores for each token into a probability distribution
  over "how much attention to pay to every other token" (all weights ≥ 0,
  summing to 1).
- Multiplying by `V` produces the output: a weighted blend of every token's
  value vector, weighted by relevance.

### 4.4 Causal masking — what makes it a *decoder*

For language modeling, token `t` must not be allowed to see tokens
`t+1, t+2, ...` — otherwise "predicting the next token" is trivially solved
by cheating (looking at the answer). This is enforced by **masking**: before
the softmax, every score for a "future" position is set to `-∞`, so after
softmax its weight becomes exactly `0`.

This is implemented with a **lower-triangular matrix** (`torch.tril`) —
position `(i, j)` is allowed (`1`) only if `j ≤ i`. This one change is the
entire difference between:
- a **decoder** (GPT-style: causal, left-to-right, good for generation), and
- an **encoder** (BERT-style: bidirectional, sees the whole input at once,
  good for understanding/classification but can't generate text token by
  token).

### 4.5 Multi-head attention

Rather than computing attention once with the full embedding dimension, it's
computed **several times in parallel**, each with a smaller slice of the
dimensionality (`head_size = n_embd / n_head`), then the results are
concatenated and linearly projected back to the full size.

Why: a single attention computation can only learn one "pattern" of
relationships. Splitting into multiple heads lets different heads
specialize — e.g. one head might learn to track subject–verb agreement,
another might track matching brackets/quotes, another might track
"what does this variable name refer to" — and the model combines all these
perspectives instead of averaging them into one.

---

## 5. Feed-forward network (the other half of a Transformer block)

After attention mixes information *between* tokens, a small
position-wise MLP processes *each token independently*:

```
Linear(n_embd → 4·n_embd) → ReLU → Linear(4·n_embd → n_embd)
```

Attention is fundamentally about *routing/mixing* information across
positions; it has limited capacity to *transform* that information
non-linearly. The feed-forward block is where most of the model's
per-token "reasoning" capacity actually lives. The 4x expansion-then-
contraction is the ratio used in the original Transformer paper and has
stuck as a convention since.

---

## 6. Residual connections — why deep networks don't fall apart

```
x = x + sublayer(x)
```

Rather than replacing `x` with the sublayer's output, the output is
**added** to the input. This "skip connection" (borrowed from ResNets)
means the gradient during backpropagation always has a direct, unobstructed
path back through the `+`, regardless of how the sublayer behaves. Without
this, stacking many layers (6, in `gpt.py`; 96+ in GPT-3) tends to make
training unstable or causes gradients to vanish — the network effectively
can't learn to use its own depth. Residual connections are what make
"just stack more layers" a viable way to scale a Transformer at all.

---

## 7. Layer normalization — keeping activations well-behaved

`nn.LayerNorm` rescales each token's vector to have zero mean and unit
variance (with learned scale/shift parameters afterward), independently for
every token. This keeps the numeric ranges flowing through the network
stable regardless of depth, which makes training more reliable and lets
higher learning rates be used.

`gpt.py` uses **pre-norm** (LayerNorm *before* attention/feed-forward,
inside the residual branch: `x + sublayer(norm(x))`), which is the modern
convention — it tends to train more stably than the original Transformer
paper's "post-norm" arrangement, especially as depth increases.

---

## 8. Dropout — the main regularizer here

`nn.Dropout(p)` randomly zeroes out a fraction `p` of values during training
(disabled automatically in `model.eval()`). It's applied after attention
weights, after projecting attention output, and inside the feed-forward
block. The effect: the network can't become overly reliant on any single
connection/neuron, which reduces overfitting — especially relevant here
given the training set is small (~1000 examples).

---

## 9. Stacking it all: one `Block`, then many

One Transformer **block** = self-attention (token-mixing) + feed-forward
(per-token transformation), each wrapped in a residual connection and
preceded by layer norm. `gpt.py` stacks 6 of these (`n_layer = 6`)
sequentially. Depth is what lets the model build up increasingly abstract
representations — early blocks might capture local syntax, later blocks
more global/semantic structure — the same way deeper layers in a CNN
capture increasingly abstract visual features.

---

## 10. Tensor shapes, end to end

It helps to keep three letters straight, since every module in `gpt.py`
manipulates tensors named this way:

- **B** — batch size (how many sequences processed in parallel; 64)
- **T** — time/sequence length (tokens per sequence; up to `block_size`=256)
- **C** — channels (the embedding dimension; `n_embd`=384, or `head_size`
  inside a single attention head)

The data flows as: `(B, T)` integer token ids → embeddings → `(B, T, C)` →
through all the blocks (shape unchanged: `(B, T, C)` in, `(B, T, C)` out,
which is exactly what lets blocks be stacked arbitrarily) → final linear
layer → `(B, T, vocab_size)` logits — a full probability distribution over
the vocabulary, for every position, in every sequence, simultaneously.

---

## 11. Training: loss, backpropagation, optimization

- **Cross-entropy loss** measures how far the model's predicted probability
  distribution is from the actual next token (a one-hot "correct answer").
  Lower loss = the model assigned higher probability to what really came
  next.
- **Backpropagation** (`loss.backward()`) computes the gradient of the loss
  with respect to every trainable parameter in the network — i.e., "which
  direction, and how much, would changing this weight reduce the loss?"
- **AdamW** (`torch.optim.AdamW`) is the optimizer that actually applies
  those gradients to update the weights. It's an adaptive-learning-rate
  variant of gradient descent (tracks per-parameter momentum and variance
  estimates) with decoupled weight decay — the de facto default optimizer
  for training Transformers.
- **`optimizer.zero_grad(set_to_none=True)`** clears out gradients from the
  previous step before computing new ones — gradients accumulate by
  default in PyTorch, so forgetting this would silently corrupt training.

One **training step** = sample a batch → forward pass (compute logits and
loss) → backward pass (compute gradients) → optimizer step (update
weights). `gpt.py` repeats this `max_iters` (5000) times.

---

## 12. Generation: sampling, not just predicting

At inference time there's no "target" to compare against — the model only
produces logits, which are converted to a probability distribution via
softmax. `gpt.py` then calls `torch.multinomial(probs, num_samples=1)`,
which **samples** a token according to those probabilities rather than
always taking the single most likely one (`argmax`, i.e. "greedy decoding").

Why sample instead of always taking the best guess? Greedy decoding tends
to produce repetitive, deterministic text and can get stuck in loops;
sampling introduces controlled randomness so the same prompt can produce
varied, more natural-feeling continuations. `gpt.py` does the simplest
possible version — plain multinomial sampling over the full distribution,
with no extra knobs. Production systems layer more control on top (see
§14.5 below for what those knobs actually do).

Each newly sampled token is appended to the running sequence and fed back
in for the next step — this loop is the entire mechanism behind "the model
writes one token, then reads its own output to write the next one."

## 13. The context window limit (`block_size`)

Because positional embeddings are a **fixed-size lookup table**
(`positional_embedding_table` has exactly `block_size` rows), the model
literally has no representation for position 256 or beyond. This is why
`generate` explicitly crops the input to the last `block_size` tokens
before every forward pass (`idx_cond = idx[:, -block_size:]`) — it's not an
optimization, it's a hard requirement of the architecture as built. This is
the same underlying reason real GPT products advertise a fixed
"context window" (e.g. 128k tokens) — it's determined at training time and
can't be exceeded without retraining (or using specialized
position-extension techniques, out of scope here).

---

## 14. Beyond this code — the rest of the real GPT stack

Everything above is implemented, in some form, in `gpt.py`. The concepts
below are **not** in this code at all, but they're the pieces that turn
"a working toy Transformer" into an actual product like ChatGPT/Claude.
Good to know these exist so you recognize them by name later, even without
needing to implement them yourself.

### 14.1 Activation functions: ReLU vs. GELU

`gpt.py`'s feed-forward block uses `nn.ReLU` (zero out anything negative,
pass everything else through unchanged). Real GPTs almost universally use
**GELU** (Gaussian Error Linear Unit) instead — a smoother curve that
doesn't have ReLU's hard corner at zero. It tends to train slightly better
in practice; the difference from ReLU is a refinement, not a different
idea.

### 14.2 Normalization variants: LayerNorm vs. RMSNorm

`gpt.py` uses `nn.LayerNorm` (recenters to zero mean *and* rescales to unit
variance). Many modern large models (e.g. LLaMA) use **RMSNorm** instead,
which skips the recentering step and only rescales by the root-mean-square
of the values — cheaper to compute, and empirically works about as well.

### 14.3 Positional encoding alternatives

`gpt.py` uses a **learned absolute** positional embedding table (one fixed
vector per position 0..255) — simple, but it hard-caps the context length
at `block_size` and doesn't generalize to positions never seen in training.
Alternatives used in production models:

- **Sinusoidal encoding** (the original Transformer paper): positions are
  encoded with fixed sine/cosine waves of different frequencies instead of
  learned vectors — not trainable, but can in principle extrapolate to
  longer sequences than were seen in training.
- **RoPE (Rotary Position Embedding)**: instead of adding a positional
  vector, it *rotates* the query/key vectors by an angle proportional to
  position before computing attention scores — used in LLaMA, GPT-NeoX, and
  most current open models because it generalizes better to long context
  and integrates naturally with relative-position reasoning.
- **ALiBi**: adds a fixed penalty to attention scores proportional to the
  distance between two positions, rather than encoding position in the
  vectors at all.

### 14.4 Weight tying

Many GPT implementations **share** (tie) the weights of the input token
embedding table and the final output projection (`lm_head`) — reusing the
same `(vocab_size, n_embd)` matrix for "word → vector" and "vector → word
scores," since both are conceptually inverses of each other. This roughly
halves the parameter count spent on vocabulary handling. `gpt.py` keeps
`embedding_table` and `lm_head` as two separate, independently-learned
matrices.

### 14.5 Decoding strategies, in full

`gpt.py` samples once from the raw softmax output. Production text
generation usually adds:

- **Temperature**: divide logits by a value `T` before softmax.
  `T < 1` sharpens the distribution (more confident/deterministic,
  approaching greedy as `T → 0`); `T > 1` flattens it (more random/diverse).
- **Top-k sampling**: zero out everything except the `k` highest-probability
  tokens before sampling, so the model can never sample something wildly
  implausible.
- **Top-p / nucleus sampling**: instead of a fixed count, keep the smallest
  set of top tokens whose probabilities sum to at least `p` (e.g. 0.9),
  which adapts `k` dynamically depending on how "confident" the
  distribution is at each step.
- **Repetition penalty**: down-weight tokens that already appeared recently,
  to reduce the model looping on the same phrase.
- **Beam search**: instead of sampling one path, track several candidate
  continuations ("beams") in parallel and keep the ones with the highest
  overall sequence probability — common in translation/summarization,
  less common for open-ended chat generation since it tends to produce
  bland, generic text.

### 14.6 Optimizer schedule and training stability tricks

`gpt.py` uses a single fixed learning rate (`1e-3`) for all 5000 steps.
Real training runs typically add:

- **Learning-rate warmup**: start the learning rate near zero and ramp it
  up over the first portion of training, since large updates on randomly
  initialized weights early on can destabilize training.
- **Cosine (or linear) decay**: gradually reduce the learning rate over the
  rest of training, so the model makes large exploratory updates early and
  small refining updates late.
- **Gradient clipping**: cap the norm of the gradient before the optimizer
  step, so a single unusually large batch/loss spike can't blow up the
  weights.
- **Weight decay** (already part of AdamW, and used here): pulls weights
  gently toward zero each step, a form of regularization.

### 14.7 Mixed precision and hardware efficiency

Real models train in **fp16/bf16** (16-bit floating point) instead of the
default 32-bit, roughly halving memory use and often significantly
speeding up training on GPUs with hardware support for it, with a "loss
scaling" trick to avoid numeric underflow. Very large models also use
**gradient checkpointing** (recompute activations during the backward pass
instead of storing all of them, trading compute for memory) and split
across multiple GPUs (**model/tensor/pipeline parallelism**) when a single
GPU can't hold the whole model. None of this is needed for `gpt.py`'s scale.

### 14.8 Inference-time optimization: the KV cache

Notice that `generate` in `gpt.py` **recomputes the entire forward pass**
for the whole context on every single new token — including re-deriving
keys/values for tokens it already processed in the previous step. Real
inference servers cache each token's key/value vectors the first time
they're computed (the **KV cache**) and, for every new token, only run the
expensive computation for that one new position — a large speedup that
matters enormously at production scale, but was left out here for
simplicity.

### 14.9 Quantization

For deployment, trained weights (normally 16 or 32-bit floats) are often
compressed to lower precision (e.g. 8-bit or 4-bit integers) —**quantization**
— trading a small amount of accuracy for large reductions in memory and
faster inference, especially relevant for running models on consumer
hardware.

### 14.10 From a base model to something like ChatGPT/Claude

`gpt.py`'s entire training loop is what's called **pretraining**: learning
to predict the next token over a big pile of raw text, with no notion of
"answering a question well" or "being helpful." Turning that into an
assistant involves further stages, applied *after* pretraining:

1. **Supervised fine-tuning (SFT)**: continue training on curated
   (prompt, ideal response) pairs, so the model learns the *format* of
   being a helpful assistant rather than just continuing arbitrary text.
2. **RLHF (Reinforcement Learning from Human Feedback)** or **DPO (Direct
   Preference Optimization)**: train the model further using human (or
   AI) preference judgments between candidate responses, pushing it toward
   answers people actually prefer — this is where behaviors like
   "refuse harmful requests" or "be concise" are shaped in, well beyond
   what raw next-token prediction on internet text would produce.

`gpt.py` never gets past step zero (pretraining) — the tiny generated
sample at the end reflects a base model's raw text-completion behavior,
not an instruction-following assistant's.

### 14.11 Evaluation: perplexity and benchmarks

The most direct metric for a language model is **perplexity** — roughly,
`e^(average cross-entropy loss)` — interpretable as "how many roughly-equally
-likely choices was the model uncertain between, on average, at each
step?" Lower is better; a perfect model has perplexity 1. Beyond raw
perplexity, real models are evaluated against benchmark suites (multiple
choice knowledge/reasoning tests, coding tasks, math problems, human
preference comparisons, etc.) — `gpt.py` never computes any evaluation
metric at all (see the note in §6 about the unused `eval_interval`).

### 14.12 Scaling up: efficient attention, MoE, and scaling laws

A few ideas that only matter once models/context get large, worth knowing
by name:

- **Multi-query / grouped-query attention**: share the same key/value
  projections across multiple (or all) heads instead of giving every head
  its own, cutting memory bandwidth during inference substantially with
  little quality loss — used in most current large models.
- **Sparse / sliding-window / linear attention**: standard self-attention
  costs `O(T²)` in sequence length `T` (every token attends to every other),
  which gets expensive at very long context. Various schemes restrict which
  positions each token can attend to (e.g. only the last N tokens, or a
  learned sparse pattern) to scale to much longer contexts more cheaply.
- **Mixture of Experts (MoE)**: replace one big feed-forward network per
  block with many smaller "expert" networks, and route each token through
  only a few of them — lets total parameter count grow much larger while
  the compute cost per token stays roughly fixed.
- **Scaling laws** (Kaplan et al., Chinchilla): empirical findings that
  model quality improves predictably as a power law with more parameters,
  more training data, and more compute — and that, for a fixed compute
  budget, there's an optimal *balance* between model size and data size
  (this is *why* modern models are trained on trillions of tokens rather
  than just being made bigger with the same amount of data).

---

## 15. How this toy model differs from a real production GPT

Useful to remember so you don't mentally overestimate what `gpt.py` proves:

| Aspect | `gpt.py` | Real GPT (e.g. GPT-3/4-class) |
|---|---|---|
| Tokenizer | hand-rolled word-level, closed vocabulary | BPE/byte-level, no OOV words |
| Parameters | ~10M-ish (n_embd=384, 6 layers, 6 heads) | billions to trillions |
| Training data | ~1000 examples, one local file | hundreds of billions of tokens |
| Training steps | 5000, no logged validation metric | huge distributed training runs with careful monitoring |
| Sampling | plain multinomial | temperature/top-k/top-p, plus RLHF/instruction-tuning on top |
| Positional encoding | fixed learned table, hard context limit | often rotary/relative encodings that generalize better to long context |

The *architecture* (embeddings → masked multi-head self-attention →
feed-forward, stacked with residuals and layer norm → next-token
cross-entropy loss → autoregressive sampling) is genuinely the same shape as
real GPTs, just scaled down by several orders of magnitude in every
dimension. That's the whole value of having built it by hand: every concept
above is a real, load-bearing piece of how large language models work
today, not a simplification that's later thrown away.
