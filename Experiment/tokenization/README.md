**_What we’re building (quick recap)_**

Start alphabet: 256 raw bytes + an end-of-word symbol </w>.

Learn BPE merges: repeatedly merge the most frequent adjacent pair in words until you hit your target vocab size.

Encode: split text into words, map to byte sequences + </w>, then greedily apply merges in the order they were learned.

Decode: map tokens back to bytes, drop </w> markers, and UTF-8 decode.

Normalization: basic NFKC (optional), but we’ll keep all bytes so round-trip works even without it.
