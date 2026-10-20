"""Extractive QA reader used to measure what compression does to answer quality.

BERT-base uncased fine-tuned on SQuAD v1.1 (ONNX Model Zoo, `bertsquad-12`,
Apache-2.0), run with onnxruntime on CPU. Long contexts are read with overlapping 256-token
windows and the best-scoring span wins. It is a real model whose answers change when
evidence is removed, which is exactly what the compression benchmark needs to measure.
"""
from __future__ import annotations

import collections
import os
import re
import string
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from third_party import bert_tokenization  # noqa: E402

MODEL_URL = "https://media.githubusercontent.com/media/onnx/models/main/validated/text/machine_comprehension/bert-squad/model/bertsquad-12.onnx"
VOCAB_URL = "https://raw.githubusercontent.com/microsoft/SDNet/master/bert_vocab_files/bert-base-uncased-vocab.txt"
SEQ = 256
STRIDE = 128
MAX_Q = 64
MAX_ANS = 30


def fetch(dirpath: str = "models") -> tuple[str, str]:
    os.makedirs(dirpath, exist_ok=True)
    m, v = f"{dirpath}/bertsquad-12.onnx", f"{dirpath}/vocab.txt"
    for url, p in ((MODEL_URL, m), (VOCAB_URL, v)):
        if not os.path.exists(p):
            print("downloading", url, file=sys.stderr)
            urllib.request.urlretrieve(url, p)
    return m, v


class Reader:
    def __init__(self, dirpath: str = "models", threads: int | None = None):
        import onnxruntime as ort

        m, v = fetch(dirpath)
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(m, so, providers=["CPUExecutionProvider"])
        self.tok = bert_tokenization.FullTokenizer(v, do_lower_case=True)
        self.windows_run = 0

    def answer(self, question: str, context: str) -> str:
        q = self.tok.tokenize(question)[:MAX_Q]
        words = context.split()
        sub, sub2word = [], []
        for wi, w in enumerate(words):
            for t in self.tok.tokenize(w):
                sub.append(t)
                sub2word.append(wi)
        room = SEQ - len(q) - 3
        starts = list(range(0, max(1, len(sub) - room + STRIDE), STRIDE)) or [0]
        feats = []
        for s in starts:
            span = sub[s : s + room]
            toks = ["[CLS]"] + q + ["[SEP]"] + span + ["[SEP]"]
            ids = self.tok.convert_tokens_to_ids(toks)
            seg = [0] * (len(q) + 2) + [1] * (len(span) + 1)
            mask = [1] * len(ids)
            pad = SEQ - len(ids)
            feats.append((ids + [0] * pad, seg + [0] * pad, mask + [0] * pad, s, len(span), len(q) + 2))
            if s + room >= len(sub):
                break
        best, best_span = -1e9, None
        for b in range(0, len(feats), 16):
            fb = feats[b : b + 16]
            out = self.sess.run(
                ["unstack:0", "unstack:1"],
                {
                    "unique_ids_raw_output___9:0": np.arange(len(fb), dtype=np.int64),
                    "input_ids:0": np.array([f[0] for f in fb], dtype=np.int64),
                    "segment_ids:0": np.array([f[1] for f in fb], dtype=np.int64),
                    "input_mask:0": np.array([f[2] for f in fb], dtype=np.int64),
                },
            )
            self.windows_run += len(fb)
            for k, f in enumerate(fb):
                st, en = out[0][k], out[1][k]
                off, n = f[5], f[4]
                si = np.argsort(-st[off : off + n])[:10]
                ei = np.argsort(-en[off : off + n])[:10]
                for a in si:
                    for e in ei:
                        if e < a or e - a + 1 > MAX_ANS:
                            continue
                        sc = st[off + a] + en[off + e]
                        if sc > best:
                            best, best_span = sc, (f[3] + a, f[3] + e)
        if best_span is None:
            return ""
        w0, w1 = sub2word[best_span[0]], sub2word[best_span[1]]
        return " ".join(words[w0 : w1 + 1]).strip(string.punctuation + " ")


def _norm(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, refs: list[str]) -> float:
    """SQuAD token F1, max over references."""
    best = 0.0
    p = _norm(pred).split()
    for r in refs:
        g = _norm(r).split()
        common = collections.Counter(p) & collections.Counter(g)
        same = sum(common.values())
        if same == 0:
            continue
        prec, rec = same / len(p), same / len(g)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best
