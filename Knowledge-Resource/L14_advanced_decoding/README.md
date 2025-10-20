# Lesson 14: Advanced decoding (quality + speed)

0) goals
    - Implement **samplers: top-k/top-p (nucleus), typical, temperature, repetition/length penalities.**
    - Implement **beam search (+ diversity penality) for tasks that want exact matches.**
    - Implement **contrastive decoding**(keep fluent but grounded by model).
    - Implement **speculative decoding** to speed up generation using smaller draft model.
    - Add **stop sequences, bad word lists, and anti-loop** heuristics.
    - Expose all the request parameters in your API.