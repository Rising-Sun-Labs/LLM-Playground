import json
from collections import Counter

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>", "<img>"]

class SimpleBPE:
    def __init__(self, vocab=None, merges=None):
        self.vocab = vocab or {}
        self.merges = merges or []
        self.rev_vocab = {i:t for t,i in self.vocab.items()}

    @staticmethod
    def _word_to_symbols(word):
        return list(word) + ["</w>"]

    def train(self, text, vocab_size=1000, min_freq=2):
        corpus = text.split()
        words = Counter(corpus)
        vocab = set(SPECIALS)
        word_syms = {w: [self._word_to_symbols(w)]*f for w,f in words.items()}
        for w in words:
            for ch in self._word_to_symbols(w):
                vocab.add(ch)
        merges = []

        def get_stats():
            pairs = Counter()
            for reps in word_syms.values():
                for seq in reps:
                    for i in range(len(seq)-1):
                        pairs[(seq[i], seq[i+1])] += 1
            return pairs

        def merge_pair(a,b):
            new_map = {}
            for w, reps in word_syms.items():
                new_reps = []
                for seq in reps:
                    out=[]; i=0
                    while i < len(seq):
                        if i < len(seq)-1 and seq[i]==a and seq[i+1]==b:
                            out.append(a+b); i+=2
                        else:
                            out.append(seq[i]); i+=1
                    new_reps.append(out)
                new_map[w] = new_reps
            return new_map

        while len(vocab) < vocab_size:
            stats = get_stats()
            if not stats: break
            (a,b), freq = stats.most_common(1)[0]
            if freq < min_freq: break
            merges.append((a,b))
            word_syms = merge_pair(a,b)
            vocab.add(a+b)

        self.vocab = {tok:i for i,tok in enumerate(list(SPECIALS)+sorted(vocab-set(SPECIALS)))}
        self.rev_vocab = {i:t for t,i in self.vocab.items()}
        self.merges = merges

    def _encode_word(self, word):
        seq = self._word_to_symbols(word)
        for a,b in self.merges:
            i=0; out=[]
            while i < len(seq):
                if i < len(seq)-1 and seq[i]==a and seq[i+1]==b:
                    out.append(a+b); i+=2
                else:
                    out.append(seq[i]); i+=1
            seq = out
        return [self.vocab.get(s, self.vocab["<unk>"]) for s in seq]

    def encode(self, text, add_special_bos_eos=False, max_len=None):
        ids = []
        for w in text.strip().split():
            ids.extend(self._encode_word(w))
        if add_special_bos_eos:
            ids = [self.vocab["<bos>"]] + ids + [self.vocab["<eos>"]]
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids):
        toks = [self.rev_vocab.get(i, "<unk>") for i in ids]
        words=[]; cur=[]
        for t in toks:
            if t == "</w>":
                words.append("".join(cur)); cur=[]
            elif t in SPECIALS:
                continue
            else:
                cur.append(t)
        if cur: words.append("".join(cur))
        return " ".join(words)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"vocab": self.vocab, "merges": self.merges}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            obj = json.load(f)
        return cls(vocab=obj["vocab"], merges=[tuple(x) for x in obj["merges"]])
