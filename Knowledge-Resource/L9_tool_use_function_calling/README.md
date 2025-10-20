# Lesson: 9 - Tool use & Function Calling


## 0) Goals

- Define a **tool schema** the model can emit(JSON)
- build a **tool server** (zip, git, diff, simple lint/tests)
- and a **planner loop**: model -> (tool call?) -> tool -> model -> answer
- make it safe (allowlist paths, timeouts) and **observable**
- add a tiny **eval** for tool success rate 

 