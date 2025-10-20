# Lesson 11: Long context & memory so it can handle big files, logs, and long chats (32k-128k+ tokens) without meling

### Goals

- extend context length safely (RoPE scaling, sliding/windowed attention)
- manage **KV cache** + chunking for long docs.
- add **retrieval memory** for multi-turn conversations.
- evaluate with **needle-in-a-haystack** and book/log QA
- wire it into your server with streaming + paging.





### micro-quiz (2 min)

- Questions
    - Why does sliding-window attention help at 64k?
    - What’s the risk of raising RoPE base too high?
    - How do you confirm long-context gains are real, not overfitting?

- Answers:
    - Reduces O(T²) to O(T·W), keeping compute/memory bounded while preserving locality.
    - You may distort short-range geometry → quality drops on normal prompts; fix with adaptation + piecewise scaling.
    - Run held-out NIAH + book/log QA across multiple lengths and report EM with confidence intervals.