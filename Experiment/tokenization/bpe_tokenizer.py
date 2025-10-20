# bpe_tokenizer.py
# A from-scratch byte-level BPE tokenizer with training, encode/decode, save/load.
# Python 3.10+

from __future__ import annotations
import json
import math
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Iterable

# ----------------------------
# Utilities
# ----------------------------

def nfkc(text: str) -> str:
    """Optional normalization. You can disable or tweak if you want raw bytes."""
    return unicodedata.normalize("NFKC", text)

# ----------------------------
# Core BPE structures
# ----------------------------

END_OF_WORD = "</w>"     # sentinel so merges don't cross words
EOW_BYTE = 256           # internal code for </w> (outside byte range 0..255)

@dataclass
class BPESerial:
    vocab_size: int
    merges: List[Tuple[int, int]]           # list of pair merges in order
    special_tokens: List[str]
    token_to_id: Dict[str, int]             # human-readable (debug/help)
    id_to_token: Dict[int, str]             # for decoding special tokens
    version: str = "bpe.v1"

class BPETokenizer:
    """
    Byte-level BPE Tokenizer trained from scratch.
    - Base alphabet: bytes 0..255 + END_OF_WORD sentinel.
    - Trains on words (split by whitespace) with EOW appended.
    - Stores merges as pairs of ints (symbols). Symbols start as bytes or EOW, then grow via merges.
    - Special tokens are reserved at the front of the ID space.
    """

    def __init__(self):
        self.special_tokens: List[str] = []
        self.symbols: Dict[Tuple[int, ...], int] = {}   # not used directly; we map by ranks
        self.merges: List[Tuple[int, int]] = []
        self.ranks: Dict[Tuple[int, int], int] = {}     # pair -> rank (merge order)
        self.vocab_size: int = 0

        # ID mapping
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # Symbol ids used in BPE space (integers). Initially 0..256 for bytes + EOW
        self.base_symbols: List[int] = list(range(0, 257))  # 0..255 bytes plus 256=EOW

        # After training we’ll have additional merged symbol IDs:
        # next_sym_id grows from 257 upward (offset by len(special_tokens) when assigning final IDs)
        self.next_sym_id: int = EOW_BYTE + 1

        # Map merged symbol tuple -> new symbol id
        self.merge_sym_map: Dict[Tuple[int, int], int] = {}

    # ------------- Training -------------

    def train(self,
              corpus: Iterable[str],
              vocab_size: int,
              special_tokens: List[str] | None = None,
              normalize: bool = True) -> None:
        """
        Train BPE on an iterable of strings. Stops when vocab (including specials) reaches vocab_size.
        """
        self.special_tokens = list(special_tokens or [])
        target_size = max(vocab_size, len(self.special_tokens) + len(self.base_symbols))
        self.vocab_size = target_size

        # Build word list as sequences of ints (bytes) + EOW sentinel
        words: List[List[int]] = []
        for line in corpus:
            text = nfkc(line) if normalize else line
            # Split by whitespace into "words". Keep whitespace implicitly via EOW sentinel
            # (decoding will reinsert spaces from original bytes).
            for w in text.split():
                bs = list(w.encode("utf-8", errors="strict"))  # byte seq
                bs.append(EOW_BYTE)
                words.append(bs)

        if not words:
            raise ValueError("Empty corpus provided to train().")

        # Count initial symbol frequencies to bootstrap (not strictly needed).
        # Main driver: iteratively merge the most frequent adjacent pair.
        vocab_symbols_count = len(self.base_symbols)
        merges_needed = target_size - (len(self.special_tokens) + vocab_symbols_count)

        for merge_step in range(max(0, merges_needed)):
            pair_counts = self._count_pair_freqs(words)
            if not pair_counts:
                break
            best_pair, best_freq = max(pair_counts.items(), key=lambda kv: kv[1])
            # Create a new symbol id for this pair and apply it across words.
            new_id = self.next_sym_id
            self.next_sym_id += 1

            self.merges.append(best_pair)
            self.ranks[best_pair] = len(self.merges) - 1
            self.merge_sym_map[best_pair] = new_id

            # Replace occurrences of best_pair in all words
            words = [self._merge_word_symbols(w, best_pair, new_id) for w in words]

        # Finalize token <-> id mapping:
        # Reserve special token IDs at the front: 0..len(special)-1
        # Then assign base and merged symbol IDs after.
        cur_id = 0
        for tok in self.special_tokens:
            self.token_to_id[tok] = cur_id
            self.id_to_token[cur_id] = tok
            cur_id += 1

        # base symbols: bytes 0..255 and EOW
        # We won't expose human-readable names for base bytes; decoding handles raw bytes.
        # But we still need stable IDs for them. Assign them next.
        # We'll store them as an offset mapping:
        self.base_offset = cur_id
        for sym in self.base_symbols:
            self.id_to_token[self.base_offset + sym] = f"<b:{sym}>"
        cur_id += len(self.base_symbols)

        # merged symbols: created in training order
        self.merge_offset = cur_id
        for rank, pair in enumerate(self.merges):
            sym_id = self.merge_sym_map[pair]
            # Map merged sym id (training-space) to a stable absolute id:
            abs_id = self.merge_offset + (sym_id - (EOW_BYTE + 1))
            self.id_to_token[abs_id] = f"<m:{pair[0]}+{pair[1]}>"

        # Sanity: cap to requested vocab size (it’s okay if fewer merges were possible)
        self.vocab_size = len(self.id_to_token)

    def _count_pair_freqs(self, words: List[List[int]]) -> Dict[Tuple[int, int], int]:
        """Count adjacent pair frequencies across all word sequences."""
        counts: Dict[Tuple[int, int], int] = defaultdict(int)
        for w in words:
            if len(w) < 2:
                continue
            for a, b in zip(w, w[1:]):
                counts[(a, b)] += 1
        return counts

    def _merge_word_symbols(self, w: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """Replace all non-overlapping occurrences of 'pair' within the word by 'new_id'."""
        if len(w) < 2:
            return w
        out: List[int] = []
        i = 0
        while i < len(w):
            if i < len(w) - 1 and (w[i], w[i+1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(w[i])
                i += 1
        return out

    # ------------- Encoding -------------

    def encode(self, text: str, normalize: bool = True) -> List[int]:
        """
        Encode text to a list of token IDs.
        Algorithm:
          - split on whitespace into words
          - word -> bytes + EOW
          - greedily merge using learned ranks (classic BPE)
          - map base/merged symbols to absolute IDs, with special tokens recognized literally
        """
        # Handle exact special-token matches as standalone tokens.
        # If your UX needs inline specials, you can pre-scan and split them out.
        specials_set = set(self.special_tokens)

        def is_special(tok: str) -> bool:
            return tok in specials_set

        # Simple splitter that preserves specials as separate "words"
        chunks: List[str] = []
        i = 0
        while i < len(text):
            matched = False
            for sp in self.special_tokens:
                if text.startswith(sp, i):
                    chunks.append(sp)
                    i += len(sp)
                    matched = True
                    break
            if matched:
                continue
            # collect until next whitespace or special
            j = i
            while j < len(text):
                if text[j].isspace():
                    break
                if any(text.startswith(sp, j) for sp in self.special_tokens):
                    break
                j += 1
            if j > i:
                chunks.append(text[i:j])
            # now add whitespace as raw bytes to preserve spacing faithfully
            k = j
            while k < len(text) and text[k].isspace():
                chunks.append(text[k])  # keep as separate chunk (spaces/newlines)
                k += 1
            i = k

        ids: List[int] = []
        for ch in chunks:
            if is_special(ch):
                ids.append(self.token_to_id[ch])
                continue
            if ch.strip() == "":
                # whitespace chunk: encode raw bytes (no EOW)
                for b in ch.encode("utf-8"):
                    ids.append(self.base_offset + b)
                continue
            # regular word -> bytes + EOW -> apply merges
            w = list((nfkc(ch) if normalize else ch).encode("utf-8"))
            w.append(EOW_BYTE)
            sym_seq = self._apply_merges_greedy(w)
            # map symbols to absolute IDs
            for s in sym_seq:
                ids.append(self._sym_to_abs_id(s))
        return ids

    def _apply_merges_greedy(self, symbols: List[int]) -> List[int]:
        """Greedy BPE: repeatedly merge the lowest-rank pair until no pair is mergeable."""
        if not self.merges:
            return symbols
        # Build a quick lookup of pair ranks; smaller rank = earlier merge, more preferred
        ranks = self.ranks

        # Convert symbols to a list of ints; iteratively contract best pairs
        while True:
            # Find best (lowest-rank) pair present
            best_pair = None
            best_rank = None
            for a, b in zip(symbols, symbols[1:]):
                r = ranks.get((a, b))
                if r is not None and (best_rank is None or r < best_rank):
                    best_pair = (a, b)
                    best_rank = r
            if best_pair is None:
                break
            new_id = self.merge_sym_map[best_pair]
            # Replace non-overlapping occurrences
            symbols = self._merge_word_symbols(symbols, best_pair, new_id)
        return symbols

    def _sym_to_abs_id(self, s: int) -> int:
        """Map internal symbol id (byte/EOW/merged) to final absolute token ID."""
        if s <= EOW_BYTE:
            return self.base_offset + s
        # merged: map by offset order
        return self.merge_offset + (s - (EOW_BYTE + 1))

    # ------------- Decoding -------------

    def decode(self, ids: List[int]) -> str:
        """Map token IDs -> bytes -> UTF-8 string. Preserves all whitespace."""
        # reverse-map abs id -> symbol
        text_bytes: bytearray = bytearray()
        acc_word: List[int] = []

        def flush_word():
            # remove EOW marker and append bytes
            if acc_word:
                for x in acc_word:
                    if x == EOW_BYTE:
                        continue
                    if 0 <= x <= 255:
                        text_bytes.append(x)
                    else:
                        # merged symbol must be decomposed — but we encoded merged as single symbol.
                        # For decoding, we need a way back to base bytes. Simplest approach:
                        # during encoding we never emit merged abs IDs for whitespace-only chunks,
                        # and for words we only emit merged IDs; here we need to "unmerge".
                        # To avoid maintaining a full reverse-merge tree, we store merged symbols as their
                        # final composed value is unknown. A simpler trick:
                        # In encode(), we never converted merged symbol back to abs id without knowing bytes.
                        # So we must keep a mapping from merged symbol -> its expansion to bytes.
                        # For simplicity, we precompute merged->bytes decode table at save-time.
                        raise RuntimeError(
                            "Decoder needs merged-to-bytes mapping. Load from saved state (see save/load)."
                        )

        # We’ll implement decoding by using a precomputed map at runtime:
        if not hasattr(self, "_decode_sym_to_bytes"):
            raise RuntimeError(
                "Tokenizer missing decode table. Load a saved tokenizer or call build_decode_table() after training."
            )

        for tid in ids:
            # Specials produce no bytes; write their literal text or skip (here: skip)
            if tid in self.id_to_token and self.id_to_token[tid] in self.special_tokens:
                # You can choose to reinsert literal specials if desired.
                continue
            # Base byte or EOW or merged:
            sym = self._abs_id_to_sym(tid)
            if sym == EOW_BYTE:
                # End of a word: flush its bytes as utf-8
                text_bytes.extend(self._decode_sym_to_bytes(sym))  # typically empty
                flush_word_from_map = self._sym_list_to_bytes(acc_word)
                text_bytes.extend(flush_word_from_map)
                acc_word = []
            elif 0 <= sym <= 255:
                acc_word.append(sym)
            else:
                # merged symbol -> expand to bytes via table
                acc_word.extend(self._decode_merged_to_byte_seq(sym))

        # trailing word (if any)
        if acc_word:
            text_bytes.extend(self._sym_list_to_bytes(acc_word))
        return text_bytes.decode("utf-8", errors="strict")

    def _abs_id_to_sym(self, tid: int) -> int:
        """Map absolute ID to internal symbol id."""
        if tid < self.base_offset:
            # special token; no symbol
            return -1
        if tid < self.base_offset + len(self.base_symbols):
            return tid - self.base_offset
        # merged:
        return (tid - self.merge_offset) + (EOW_BYTE + 1)

    # ---------------- Save / Load & decode tables ----------------

    def _build_decode_tables(self):
        """
        Build reverse tables to expand merged symbols back to base bytes for decoding.
        We reconstruct merge expansions into bytes by replaying merges on unit words.
        """
        # Start with identity: each base symbol maps to [sym] bytes (EOW maps to [])
        sym_to_seq = {i: [i] for i in range(0, 257)}
        sym_to_seq[EOW_BYTE] = [EOW_BYTE]

        # Replay merges in order: new_sym = concat(seq[a], seq[b])
        for pair in self.merges:
            a, b = pair
            new_sym = self.merge_sym_map[pair]
            sym_to_seq[new_sym] = sym_to_seq[a] + sym_to_seq[b]

        # Two helpers:
        self._decode_merged_to_byte_seq = lambda s: [x for x in sym_to_seq[s] if 0 <= x <= 255]
        self._decode_sym_to_bytes = lambda s: [] if s == EOW_BYTE else ([s] if 0 <= s <= 255 else self._decode_merged_to_byte_seq(s))
        self._sym_list_to_bytes = lambda seq: bytearray([b for s in seq for b in self._decode_sym_to_bytes(s) if 0 <= b <= 255])

    def save(self, path: str):
        """Save a fully usable tokenizer (including decode tables) to JSON."""
        self._build_decode_tables()
        ser = {
            "meta": {
                "version": "bpe.v1",
                "vocab_size": self.vocab_size,
                "special_tokens": self.special_tokens,
                "base_symbols": len(self.base_symbols)
            },
            "merges": self.merges,
            "merge_sym_map": {f"{a},{b}": v for (a, b), v in self.merge_sym_map.items()},
            "token_ids": {
                "token_to_id": self.token_to_id,
                "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
                "base_offset": self.base_offset,
                "merge_offset": self.merge_offset
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ser, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            ser = json.load(f)
        obj = cls()
        meta = ser["meta"]
        obj.vocab_size = meta["vocab_size"]
        obj.special_tokens = meta["special_tokens"]

        obj.merges = [tuple(pair) for pair in ser["merges"]]
        obj.merge_sym_map = {tuple(map(int, k.split(","))): int(v) for k, v in ser["merge_sym_map"].items()}
        obj.ranks = {pair: i for i, pair in enumerate(obj.merges)}

        tid = ser["token_ids"]
        obj.token_to_id = {k: int(v) for k, v in tid["token_to_id"].items()}
        obj.id_to_token = {int(k): v for k, v in tid["id_to_token"].items()}
        obj.base_offset = int(tid["base_offset"])
        obj.merge_offset = int(tid["merge_offset"])

        # reconstruct numbering
        obj.next_sym_id = max([EOW_BYTE] + list(obj.merge_sym_map.values())) + 1
        obj._build_decode_tables()
        return obj

# ----------------------------
# Tiny demo / unit tests
# ----------------------------

if __name__ == "__main__":
    corpus = [
        "Hello world!",
        "Hello, hello — code-review bot.",
        "review this diff: - bug() + fix()",
        "emails: please write a crisp review request.",
        "路径/ファイル/파일 — filenames & code.",
        "Tabs\tand\nnewlines should survive ✅",
    ]

    tok = BPETokenizer()
    tok.train(
        corpus=corpus,
        vocab_size=500,                       # small demo size; raise for real training
        special_tokens=["<bos>", "<eos>", "<pad>", "<usr>", "<asst>"],
        normalize=True
    )
    tok._build_decode_tables()

    # Round-trip test
    sample = "<usr>Hello world!\nReview this diff:\n- bug()\n+ fix()</usr>"
    ids = tok.encode(sample)
    back = tok.decode(ids)

    print("Encoded IDs:", ids[:40], "… (total:", len(ids), ")")
    print("Round-trip OK:", back == sample)

    # Save & load test
    tok.save("bpe_tokenizer.json")
    tok2 = BPETokenizer.load("bpe_tokenizer.json")
    back2 = tok2.decode(tok2.encode(sample))
    print("Load round-trip OK:", back2 == sample)
