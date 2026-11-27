# backend/main.py

import os
import json
import math
import pickle
import re
import string
import time
from pathlib import Path
from typing import List

import networkx as nx
import numpy as np
import uvicorn
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# MODULE 3 — BiLSTM + BiDAF MODEL DEFINITION
# ─────────────────────────────────────────────

class BidafAttention(nn.Module):
    """Bidirectional Attention Flow between context and query."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.w_ctx = nn.Linear(hidden_size * 2, 1, bias=False)
        self.w_q   = nn.Linear(hidden_size * 2, 1, bias=False)
        self.w_cq  = nn.Linear(hidden_size * 2, 1, bias=False)

    def forward(self, ctx, q, ctx_mask, q_mask):
        # ctx: (B, T, 2H)   q: (B, J, 2H)
        B, T, H2 = ctx.shape
        J = q.shape[1]

        ctx_exp = ctx.unsqueeze(2).expand(B, T, J, H2)
        q_exp   = q.unsqueeze(1).expand(B, T, J, H2)

        s = (self.w_ctx(ctx_exp) + self.w_q(q_exp) +
             self.w_cq(ctx_exp * q_exp)).squeeze(-1)   # (B, T, J)

        if q_mask is not None:
            s = s.masked_fill(~q_mask.unsqueeze(1), -1e9)

        # Context-to-Query
        a = F.softmax(s, dim=2)          # (B, T, J)
        A = torch.bmm(a, q)              # (B, T, 2H)

        # Query-to-Context
        q2c_logits = s.max(dim=2).values            # (B, T)
        if ctx_mask is not None:
            q2c_logits = q2c_logits.masked_fill(~ctx_mask, -1e9)
        b = F.softmax(q2c_logits, dim=1)            # (B, T)
        B_vec = torch.bmm(b.unsqueeze(1), ctx).expand(B, T, H2)  # (B, T, 2H)

        # Concatenate: [ctx, A, ctx*A, ctx*B]
        out = torch.cat([ctx, A, ctx * A, ctx * B_vec], dim=2)   # (B, T, 8H)
        return out, a   # also return attention weights for visualization


class BiLSTMQA(nn.Module):
    """
    BiLSTM + BiDAF extractive QA model.
    Exact architecture matching qa_best.pt weights.
    """

    def __init__(
        self,
        vocab_size:  int = 200002,
        emb_dim:     int = 100,
        hidden_size: int = 128,
    ):
        super().__init__()
        H = hidden_size

        # Embedding + projection
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.emb_proj  = nn.Linear(emb_dim, H)

        # Encoders
        self.ctx_encoder = nn.LSTM(H, H, num_layers=2, batch_first=True,
                                   bidirectional=True, dropout=0.0)
        self.q_encoder   = nn.LSTM(H, H, num_layers=2, batch_first=True,
                                   bidirectional=True, dropout=0.0)

        # BiDAF attention — input is 2H per side
        self.attention = BidafAttention(H)

        # Modeling LSTM — input is 8H (4 concat vectors of 2H each)
        self.model_lstm = nn.LSTM(H * 8, H, num_layers=2, batch_first=True,
                                  bidirectional=True, dropout=0.0)

        # Output heads
        # span_start/end input = 2H (model_lstm) + 8H (attention out) = 10H = 1280
        self.span_start  = nn.Linear(H * 10, 1)
        self.span_end    = nn.Linear(H * 10, 1)
        # answerable input = 2H = 256 (from q_encoder final hidden)
        self.answerable  = nn.Linear(H * 2, 1)

    def forward(self, ctx_ids, q_ids, ctx_lens, q_lens):
        # Embed + project
        ctx_emb = F.relu(self.emb_proj(self.embedding(ctx_ids)))   # (B, T, H)
        q_emb   = F.relu(self.emb_proj(self.embedding(q_ids)))     # (B, J, H)

        # Encode
        ctx_enc, _      = self.ctx_encoder(ctx_emb)   # (B, T, 2H)
        q_enc, (q_h, _) = self.q_encoder(q_emb)       # (B, J, 2H)

        # Masks
        B, T = ctx_ids.shape
        J    = q_ids.shape[1]
        ctx_mask = torch.arange(T, device=ctx_ids.device).unsqueeze(0) < ctx_lens.unsqueeze(1)
        q_mask   = torch.arange(J, device=q_ids.device).unsqueeze(0)   < q_lens.unsqueeze(1)

        # BiDAF attention
        att_out, att_weights = self.attention(ctx_enc, q_enc, ctx_mask, q_mask)  # (B, T, 8H)

        # Modeling LSTM
        mod_out, _ = self.model_lstm(att_out)   # (B, T, 2H)

        # Span prediction — concat attention + modeling output
        span_in = torch.cat([att_out, mod_out], dim=2)   # (B, T, 10H)
        s_logits = self.span_start(span_in).squeeze(-1)  # (B, T)
        e_logits = self.span_end(span_in).squeeze(-1)    # (B, T)
        s_logits = s_logits.masked_fill(~ctx_mask, -1e9)
        e_logits = e_logits.masked_fill(~ctx_mask, -1e9)

        # Answerability — mean-pool modeling LSTM over valid context tokens (matches training)
        m_pool = (mod_out * ctx_mask.unsqueeze(-1).float()).sum(1) / ctx_lens.clamp(min=1).unsqueeze(-1).float()
        a_logits = self.answerable(m_pool).squeeze(-1)   # (B,)

        return s_logits, e_logits, a_logits, att_weights

# ─────────────────────────────────────────────
# PATHS — adjust if your folder structure differs
# ─────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent
MODULE1_DIR = MODELS_DIR / "module 1 models"
SUMM_DIR    = MODELS_DIR / "summ_artifacts"
MODULE3_PATH = MODELS_DIR / "module 3 model" / "qa_best.pt"
MODULE4_DIR = MODELS_DIR / "module 4 distilbert"

def build_word2idx_from_glove(glove_path: str, max_words: int = 200_000) -> dict:
    """Rebuild the word2idx that was used during training."""
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    with open(glove_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_words:
                break
            word = line.split()[0]
            word2idx[word] = len(word2idx)
    return word2idx

# ─────────────────────────────────────────────
# LOAD MODULE 1 MODELS AT STARTUP
# ─────────────────────────────────────────────
print("Loading Module 1 models...")

with open(MODULE1_DIR / "tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

tfidf_matrix = load_npz(MODULE1_DIR / "tfidf_matrix.npz")

with open(MODULE1_DIR / "bm25.pkl", "rb") as f:
    bm25_model = pickle.load(f)

with open(MODULE1_DIR / "corpus_index.json", "r", encoding="utf-8") as f:
    corpus_index = json.load(f)

PASSAGES: List[str] = corpus_index["passages"]   # 20,233 passages
CORPUS_SIZE = len(PASSAGES)

# Tokenized corpus for BM25 (already stored inside bm25.pkl)
# BM25Okapi stores doc_freqs but needs tokenized queries at inference

print(f"Module 1 loaded: {CORPUS_SIZE} passages, TF-IDF shape {tfidf_matrix.shape}")

# ─────────────────────────────────────────────
# LOAD MODULE 2 CONFIG AT STARTUP
# ─────────────────────────────────────────────
with open(SUMM_DIR / "config.json", "r", encoding="utf-8") as f:
    summ_config = json.load(f)

SUMM_RATIO    = summ_config.get("ratio", 0.3)       # keep 30% of sentences
SUMM_MAX_SENTS = summ_config.get("max_sents", 5)    # never keep more than 5

print(f"Module 2 loaded: method={summ_config['method']}, ratio={SUMM_RATIO}, max={SUMM_MAX_SENTS}")

# ─────────────────────────────────────────────
# LOAD MODULE 4 — DISTILBERT QA MODEL
# ─────────────────────────────────────────────
print("Loading Module 4 — DistilBERT QA model...")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

distilbert_tokenizer = AutoTokenizer.from_pretrained(str(MODULE4_DIR))
distilbert_model     = AutoModelForQuestionAnswering.from_pretrained(str(MODULE4_DIR)).to(DEVICE)
distilbert_model.eval()

print(f"Module 4 loaded: DistilBertForQuestionAnswering on {DEVICE}")

# ─────────────────────────────────────────────
# LOAD MODULE 3 — BiLSTM + BiDAF QA MODEL
# ─────────────────────────────────────────────
print("Loading Module 3 — BiLSTM + BiDAF QA model...")

# The model embeds GloVe vocab — rebuild word2idx from the same GloVe file
# used during training: glove.6B.100d.txt (max_words=200_000)
# Adjust GLOVE_PATH to wherever glove.6B.100d.txt is on your machine
GLOVE_PATH = os.environ.get("GLOVE_PATH", str(MODELS_DIR / "glove.6B.100d.txt"))

BILSTM_WORD2IDX: dict = {}
if Path(GLOVE_PATH).exists():
    BILSTM_WORD2IDX = build_word2idx_from_glove(GLOVE_PATH, max_words=200_000)
    print(f"GloVe vocab loaded: {len(BILSTM_WORD2IDX)} words")
else:
    print(f"WARNING: GloVe file not found at {GLOVE_PATH}")
    print("BiLSTM model will load but all tokens will map to <UNK>.")
    print("Set GLOVE_PATH env var to point to glove.6B.100d.txt")

bilstm_model = BiLSTMQA(vocab_size=200002, emb_dim=100, hidden_size=128)
state_dict   = torch.load(str(MODULE3_PATH), map_location=DEVICE, weights_only=True)
bilstm_model.load_state_dict(state_dict)
bilstm_model = bilstm_model.to(DEVICE)
bilstm_model.eval()

print(f"Module 3 loaded: BiLSTM+BiDAF on {DEVICE}")

# Constants matching training
BILSTM_MAX_CTX = 400
BILSTM_MAX_Q   = 50


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
app = FastAPI(title="NLP Pipeline API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"status": "ok", "message": "NLP Pipeline API is running"}

# ─────────────────────────────────────────────
# HELPERS — MODULE 1: IR
# ─────────────────────────────────────────────

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def preprocess_ir(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace (matches training)."""
    text = text.lower().translate(_PUNCT_TABLE)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_simple(text: str) -> List[str]:
    """Whitespace tokenizer matching how BM25 was trained (preserves numbers)."""
    return preprocess_ir(text).split()


def retrieve_tfidf(query: str, top_k: int = 10):
    """TF-IDF cosine similarity retrieval."""
    q_vec = tfidf_vectorizer.transform([preprocess_ir(query)])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_indices]


def retrieve_bm25(query: str, top_k: int = 10):
    """BM25Okapi retrieval."""
    tokens = tokenize_simple(query)
    scores = bm25_model.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_indices]


def hybrid_retrieve_with_breakdown(query: str, top_k: int = 5, alpha: float = 0.5):
    """
    Vectorized hybrid TF-IDF + BM25 fusion.
    score = alpha * norm(tfidf) + (1 - alpha) * norm(bm25)
    Returns [(idx, hybrid_score, norm_tfidf, norm_bm25), ...].
    """
    q_clean = preprocess_ir(query)
    q_vec = tfidf_vectorizer.transform([q_clean])
    tfidf_scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    bm25_scores = np.asarray(bm25_model.get_scores(q_clean.split()), dtype=np.float64)

    tfidf_max = float(tfidf_scores.max()) if tfidf_scores.size and tfidf_scores.max() > 0 else 1.0
    bm25_max  = float(bm25_scores.max())  if bm25_scores.size  and bm25_scores.max()  > 0 else 1.0

    tfidf_norm = tfidf_scores / tfidf_max
    bm25_norm  = bm25_scores / bm25_max
    combined   = alpha * tfidf_norm + (1.0 - alpha) * bm25_norm

    top_indices = np.argsort(combined)[::-1][:top_k]
    return [
        (int(i), float(combined[i]), float(tfidf_norm[i]), float(bm25_norm[i]))
        for i in top_indices
    ]


def hybrid_retrieve(query: str, top_k: int = 5, alpha: float = 0.5):
    """Hybrid TF-IDF + BM25 fusion returning [(idx, score), ...]."""
    return [
        (idx, score)
        for idx, score, _, _ in hybrid_retrieve_with_breakdown(query, top_k=top_k, alpha=alpha)
    ]


# ─────────────────────────────────────────────
# HELPERS — MODULE 2: SUMMARIZATION
# ─────────────────────────────────────────────

def split_sentences(text: str) -> List[str]:
    """Split passage into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def tfidf_sentence_scores(sentences: List[str]) -> np.ndarray:
    """Score each sentence by its mean TF-IDF weight."""
    if not sentences:
        return np.array([])
    vecs = tfidf_vectorizer.transform(sentences)
    # Mean non-zero TF-IDF weight per sentence
    scores = np.array(vecs.mean(axis=1)).flatten()
    return scores


def textrank_scores(sentences: List[str], damping: float = 0.85) -> np.ndarray:
    """TextRank: build cosine similarity graph, run PageRank."""
    if len(sentences) < 2:
        return np.ones(len(sentences))

    vecs = tfidf_vectorizer.transform(sentences)
    sim_matrix = cosine_similarity(vecs)
    np.fill_diagonal(sim_matrix, 0)

    # Build graph
    G = nx.from_numpy_array(sim_matrix)
    try:
        pr = nx.pagerank(G, alpha=damping, max_iter=100)
        scores = np.array([pr[i] for i in range(len(sentences))])
    except nx.PowerIterationFailedConvergence:
        scores = np.ones(len(sentences)) / len(sentences)

    return scores


def position_scores(sentences: List[str]) -> np.ndarray:
    """Positional bias: first and last sentences get a bonus."""
    n = len(sentences)
    if n == 0:
        return np.array([])
    scores = np.zeros(n)
    # First sentence bonus
    scores[0] = 1.0
    # Last sentence bonus
    if n > 1:
        scores[-1] = 0.7
    # Small decay for middle sentences
    for i in range(1, n - 1):
        scores[i] = max(0, 1.0 - (i / n) * 0.5)
    return scores


def normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.ones_like(arr) / len(arr)
    return (arr - mn) / (mx - mn)


def summarize_passage(
    passage: str,
    alpha: float = 0.4,   # TF-IDF weight
    beta: float  = 0.4,   # TextRank weight
    gamma: float = 0.2,   # Position weight
) -> dict:
    """
    Hybrid extractive summarization.
    Returns dict with original sentences, selected indices, summary text,
    individual scores, and compression stats.
    """
    sentences = split_sentences(passage)
    n = len(sentences)

    if n == 0:
        return {"sentences": [], "selected_indices": [], "summary": passage,
                "scores": [], "total": 0, "kept": 0, "compression": 0}

    if n == 1:
        return {"sentences": sentences, "selected_indices": [0], "summary": sentences[0],
                "scores": [1.0], "total": 1, "kept": 1, "compression": 0}

    # Compute all three score types
    tf_scores  = normalize(tfidf_sentence_scores(sentences))
    tr_scores  = normalize(textrank_scores(sentences))
    pos_scores = normalize(position_scores(sentences))

    # Weighted combination (matching config: method=hybrid)
    combined = alpha * tf_scores + beta * tr_scores + gamma * pos_scores

    # Determine how many sentences to keep
    n_keep = max(1, min(SUMM_MAX_SENTS, math.ceil(n * SUMM_RATIO)))

    # Select top-scoring sentence indices (preserving original order)
    top_indices = sorted(np.argsort(combined)[::-1][:n_keep].tolist())

    summary_text = " ".join(sentences[i] for i in top_indices)
    compression_pct = round((1 - n_keep / n) * 100)

    return {
        "sentences": sentences,
        "selected_indices": top_indices,
        "summary": summary_text,
        "scores": [round(float(s), 4) for s in combined],
        "total": n,
        "kept": n_keep,
        "compression": compression_pct,
        "tfidf_scores":   [round(float(s), 4) for s in tf_scores],
        "textrank_scores": [round(float(s), 4) for s in tr_scores],
        "position_scores": [round(float(s), 4) for s in pos_scores],
    }


# ─────────────────────────────────────────────
# HELPERS — MODULE 4: DISTILBERT QA
# ─────────────────────────────────────────────

MAX_LENGTH     = 384
STRIDE         = 128
MAX_ANSWER_LEN = 50

def predict_distilbert(question: str, context: str) -> dict:
    """
    Run DistilBERT inference on a single (question, context) pair.
    Uses sliding window for contexts longer than MAX_LENGTH.

    Returns:
    {
      "answer": str,               # predicted answer text ("" if unanswerable)
      "answerable": bool,
      "start_logit": float,
      "end_logit": float,
      "cls_logit": float,          # logit for [CLS] as start token (answerability score)
      "start_token": int,          # token index of answer start in context
      "end_token": int,            # token index of answer end in context
      "tokens": [str, ...],        # all tokens in the input sequence
      "start_logits_top5": [...],  # top-5 (token_idx, logit) for start
      "end_logits_top5":   [...],  # top-5 (token_idx, logit) for end
      "token_type": [str, ...],    # "cls"/"sep"/"question"/"context"/"pad" per token
    }
    """
    distilbert_model.eval()

    # Tokenize with sliding window
    inputs = distilbert_tokenizer(
        question,
        context,
        max_length             = MAX_LENGTH,
        stride                 = STRIDE,
        truncation             = "only_second",
        return_overflowing_tokens = True,
        return_offsets_mapping    = True,
        padding                   = "max_length",
        return_tensors            = "pt",
    )

    offset_mapping   = inputs.pop("offset_mapping")
    overflow_mapping = inputs.pop("overflow_to_sample_mapping", None)
    inputs.pop("token_type_ids", None)  # DistilBertForQuestionAnswering does not accept token_type_ids in transformers 4.x

    # Run forward pass on all chunks
    best_answer    = ""
    best_score     = float("-inf")
    best_start_tok = 0
    best_end_tok   = 0
    best_start_log = 0.0
    best_end_log   = 0.0
    best_cls_log   = 0.0
    best_null_score = 0.0
    best_tokens    = []
    best_token_type = []
    best_start_top5 = []
    best_end_top5   = []

    num_chunks = inputs["input_ids"].shape[0]

    for chunk_idx in range(num_chunks):
        chunk_inputs = {k: v[chunk_idx].unsqueeze(0).to(DEVICE) for k, v in inputs.items()}
        offsets      = offset_mapping[chunk_idx]  # (seq_len, 2)

        with torch.no_grad():
            outputs = distilbert_model(**chunk_inputs)

        start_logits = outputs.start_logits[0].cpu()   # (seq_len,)
        end_logits   = outputs.end_logits[0].cpu()

        input_ids     = chunk_inputs["input_ids"][0].cpu()
        token_ids_list = input_ids.tolist()
        tokens_decoded = distilbert_tokenizer.convert_ids_to_tokens(token_ids_list)

        # Build token_type labels for visualization
        sep_positions = [i for i, tok in enumerate(tokens_decoded) if tok == "[SEP]"]
        token_type_labels = []
        for i, tok in enumerate(tokens_decoded):
            if tok == "[CLS]":
                token_type_labels.append("cls")
            elif tok == "[SEP]":
                token_type_labels.append("sep")
            elif tok == "[PAD]":
                token_type_labels.append("pad")
            elif len(sep_positions) >= 1 and i < sep_positions[0]:
                token_type_labels.append("question")
            else:
                token_type_labels.append("context")

        cls_logit  = float(start_logits[0])
        # BUG FIX: proper null score = CLS start + CLS end (not just CLS start)
        null_score = float(start_logits[0]) + float(end_logits[0])

        # Top-5 start and end positions for visualization
        start_top5_idx = torch.topk(start_logits, min(5, len(start_logits))).indices.tolist()
        end_top5_idx   = torch.topk(end_logits,   min(5, len(end_logits))).indices.tolist()
        start_top5 = [(int(i), round(float(start_logits[i]), 3)) for i in start_top5_idx]
        end_top5   = [(int(i), round(float(end_logits[i]),   3)) for i in end_top5_idx]

        # Find best valid span
        seq_len = len(token_ids_list)
        for start in range(seq_len):
            for end in range(start, min(start + MAX_ANSWER_LEN, seq_len)):
                # Skip if either token is not in the context portion
                if token_type_labels[start] != "context":
                    continue
                if token_type_labels[end] != "context":
                    continue
                score = float(start_logits[start]) + float(end_logits[end])
                if score > best_score:
                    best_score     = score
                    best_start_tok = start
                    best_end_tok   = end
                    best_start_log = round(float(start_logits[start]), 3)
                    best_end_log   = round(float(end_logits[end]),     3)
                    best_cls_log   = round(cls_logit, 3)
                    best_null_score = round(null_score, 3)
                    best_tokens    = tokens_decoded
                    best_token_type = token_type_labels
                    best_start_top5 = start_top5
                    best_end_top5   = end_top5

                    # Decode answer from offsets
                    start_char = int(offsets[start][0])
                    end_char   = int(offsets[end][1])
                    best_answer = context[start_char:end_char].strip()

    # ── Answerability (SQuAD 2.0 standard) ────────────────────────────────
    # Compare best span score against the NULL score (CLS as both start AND end).
    # Standard threshold = 0.0 — answerable only if span beats the null span.
    NULL_THRESHOLD = 0.0
    answerable = (
        best_score > float("-inf")
        and best_answer != ""
        and best_start_tok != 0
        and best_score > (best_null_score + NULL_THRESHOLD)
    )

    if not answerable:
        best_answer = ""

    # Trim trailing [PAD] tokens so the UI renders the full winning window cleanly
    non_pad_len = len(best_tokens)
    while non_pad_len > 0 and best_token_type[non_pad_len - 1] == "pad":
        non_pad_len -= 1

    return {
        "answer":           best_answer,
        "answerable":       answerable,
        "start_logit":      best_start_log,
        "end_logit":        best_end_log,
        "cls_logit":        best_cls_log,
        "null_score":       best_null_score,
        "start_token":      best_start_tok,
        "end_token":        best_end_tok,
        "tokens":           best_tokens[:non_pad_len],
        "token_type":       best_token_type[:non_pad_len],
        "start_logits_top5": best_start_top5,
        "end_logits_top5":   best_end_top5,
    }


# ─────────────────────────────────────────────
# HELPERS — MODULE 3: BiLSTM + BiDAF
# ─────────────────────────────────────────────

def bilstm_tokenize(text: str) -> list:
    """Exact tokenizer used during training."""
    return re.findall(r"\w+|[^\w\s]", text.lower())

def bilstm_encode(tokens: list, word2idx: dict, max_len: int):
    """Encode token list to padded integer ids."""
    ids = [word2idx.get(t, 1) for t in tokens[:max_len]]  # 1 = <UNK>
    length = len(ids)
    ids += [0] * (max_len - length)                        # 0 = <PAD>
    return ids, length

def predict_bilstm(question: str, context: str) -> dict:
    """
    Run BiLSTM + BiDAF inference on a (question, context) pair.

    Returns:
    {
      "answer": str,
      "answerable": bool,
      "start_token": int,
      "end_token": int,
      "start_logit": float,
      "end_logit": float,
      "answerable_score": float,
      "ctx_tokens": [str, ...],       # tokenized context words (max 80 for UI)
      "q_tokens": [str, ...],         # tokenized question words
      "attention_weights": [[float]], # (T_ctx x J_q) matrix, capped at 8x8 for UI
    }
    """
    bilstm_model.eval()

    ctx_tokens = bilstm_tokenize(context)
    q_tokens   = bilstm_tokenize(question)

    ctx_ids, ctx_len = bilstm_encode(ctx_tokens, BILSTM_WORD2IDX, BILSTM_MAX_CTX)
    q_ids,   q_len   = bilstm_encode(q_tokens,   BILSTM_WORD2IDX, BILSTM_MAX_Q)

    # Avoid zero-length tensors which break the model
    if ctx_len == 0:
        ctx_len = 1
    if q_len == 0:
        q_len = 1

    ctx_t = torch.tensor([ctx_ids], dtype=torch.long).to(DEVICE)
    q_t   = torch.tensor([q_ids],   dtype=torch.long).to(DEVICE)
    cl_t  = torch.tensor([ctx_len], dtype=torch.long).to(DEVICE)
    ql_t  = torch.tensor([q_len],   dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        s_logits, e_logits, a_logit, att_weights = bilstm_model(ctx_t, q_t, cl_t, ql_t)

    s_logits = s_logits[0].cpu()    # (T,)
    e_logits = e_logits[0].cpu()    # (T,)
    a_score  = float(torch.sigmoid(a_logit[0]).cpu())

    # Find best valid span (start ≤ end, both within ctx_len)
    best_score = float("-inf")
    best_start = 0
    best_end   = 0

    for s in range(min(ctx_len, BILSTM_MAX_CTX)):
        for e in range(s, min(s + 30, ctx_len, BILSTM_MAX_CTX)):
            score = float(s_logits[s]) + float(e_logits[e])
            if score > best_score:
                best_score = score
                best_start = s
                best_end   = e

    # Decode answer from tokens
    answer_tokens = ctx_tokens[best_start: best_end + 1]
    answer_text   = " ".join(answer_tokens)

    # Answerable threshold — sigmoid > 0.5
    answerable = a_score > 0.5
    if not answerable:
        answer_text = ""

    # Attention weights for heatmap — cap to 8 ctx tokens × 8 q tokens
    att_np = att_weights[0].cpu().numpy()  # (T, J)
    heat_ctx = min(8, ctx_len)
    heat_q   = min(8, q_len)
    heat     = att_np[:heat_ctx, :heat_q].tolist()

    # Cap token lists for frontend (no need to send 400 tokens)
    ui_ctx_tokens = ctx_tokens[:80]
    ui_q_tokens   = q_tokens[:BILSTM_MAX_Q]

    return {
        "answer":            answer_text,
        "answerable":        answerable,
        "start_token":       best_start,
        "end_token":         best_end,
        "start_logit":       round(float(s_logits[best_start]), 3),
        "end_logit":         round(float(e_logits[best_end]),   3),
        "answerable_score":  round(a_score, 3),
        "ctx_tokens":        ui_ctx_tokens,
        "q_tokens":          ui_q_tokens,
        "attention_weights": heat,   # list[list[float]], shape (≤8, ≤8)
    }


# ─────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)


class PipelineRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)


class SummarizeRequest(BaseModel):
    passage: str = Field(..., min_length=1)


class QARequest(BaseModel):
    question: str = Field(..., min_length=1)
    context: str = Field(..., min_length=1)


# ── Health check ──────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "corpus_size": CORPUS_SIZE,
        "summ_config": summ_config,
    }


# ── Module 1: Information Retrieval ──────────
@app.post("/api/retrieve")
def retrieve(req: QueryRequest):
    """Returns top-k passages with hybrid TF-IDF + BM25 scores."""
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    t0 = time.time()
    hybrid = hybrid_retrieve_with_breakdown(query, top_k=req.top_k, alpha=req.alpha)

    results = [
        {
            "rank": rank,
            "passage_id": idx,
            "passage": PASSAGES[idx],
            "score": round(score, 4),
            "tfidf_score": round(tfidf_sc, 4),
            "bm25_score":  round(bm25_sc, 4),
        }
        for rank, (idx, score, tfidf_sc, bm25_sc) in enumerate(hybrid, start=1)
    ]

    elapsed = round((time.time() - t0) * 1000, 1)
    return {"query": query, "results": results, "retrieval_time_ms": elapsed}


# ── Module 2: Summarization ───────────────────
@app.post("/api/summarize")
def summarize(req: SummarizeRequest):
    """Summarizes a single passage."""
    passage = req.passage.strip()
    if not passage:
        raise HTTPException(status_code=400, detail="passage is required")
    return summarize_passage(passage)


# ── Module 4: DistilBERT QA ──────────────────
@app.post("/api/qa/distilbert")
def qa_distilbert(req: QARequest):
    """Run DistilBERT QA on a question + context."""
    question = req.question.strip()
    context  = req.context.strip()
    if not question or not context:
        raise HTTPException(status_code=400, detail="question and context are required")

    t0     = time.time()
    result = predict_distilbert(question, context)
    elapsed = round((time.time() - t0) * 1000, 1)

    return {
        "question":          question,
        "context":           context,
        "inference_time_ms": elapsed,
        **result,
    }


# ── Module 3: BiLSTM + BiDAF QA ──────────────
@app.post("/api/qa/bilstm")
def qa_bilstm(req: QARequest):
    """Run BiLSTM + BiDAF QA on a question + context."""
    question = req.question.strip()
    context  = req.context.strip()
    if not question or not context:
        raise HTTPException(status_code=400, detail="question and context are required")

    t0     = time.time()
    result = predict_bilstm(question, context)
    elapsed = round((time.time() - t0) * 1000, 1)

    return {
        "question":          question,
        "context":           context,
        "inference_time_ms": elapsed,
        **result,
    }


# ─────────────────────────────────────────────
# PIPELINE ENDPOINT
# ─────────────────────────────────────────────
@app.post("/api/pipeline")
def pipeline(req: PipelineRequest):
    """Full Module 1 + 2 + 3/4 pipeline in one call."""
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    t0 = time.time()
    hybrid = hybrid_retrieve_with_breakdown(query, top_k=req.top_k, alpha=0.5)
    t1 = time.time()

    retrieved = []
    summ_ms = 0.0
    qa_ms   = 0.0
    for rank, (idx, score, tfidf_sc, bm25_sc) in enumerate(hybrid, start=1):
        passage = PASSAGES[idx]
        _ts = time.time()
        summ = summarize_passage(passage)
        summ_ms += (time.time() - _ts) * 1000

        _tq = time.time()
        # Run QA on FULL passage (not the summary) so no answer sentences are dropped
        qa_result = predict_distilbert(query, passage)
        qa_bilstm = predict_bilstm(query, summ["summary"])
        qa_ms += (time.time() - _tq) * 1000

        retrieved.append({
            "rank": rank,
            "passage_id": idx,
            "passage": passage,
            "score": round(score, 4),
            "tfidf_score": round(tfidf_sc, 4),
            "bm25_score":  round(bm25_sc, 4),
            "summary": summ,
            "qa_distilbert": qa_result,
            "qa_bilstm": qa_bilstm,
        })

    t2 = time.time()

    # ── Multi-passage QA selection ────────────────────────────────────────
    # Among all retrieved passages, pick the one where the model is most
    # confident it found an answer (highest span_score - null_score gap).
    def qa_confidence(qa: dict) -> float:
        """Higher = model more confident the answer is in this passage."""
        span_score = qa["start_logit"] + qa["end_logit"]
        null_score = qa.get("null_score", qa["cls_logit"])
        return span_score - null_score

    best_qa = max(
        (r["qa_distilbert"] for r in retrieved),
        key=qa_confidence,
    )

    return {
        "query": query,
        "retrieval_time_ms":     round((t1 - t0) * 1000, 1),
        "summarization_time_ms": round(summ_ms, 1),
        "qa_time_ms":            round(qa_ms, 1),
        "total_time_ms":         round((t2 - t0) * 1000, 1),
        "retrieved": retrieved,
        "best_qa": best_qa,   # best answer across all retrieved passages
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
