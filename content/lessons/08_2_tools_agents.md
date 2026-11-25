<<SVG:tools_agents_host_loop>>

*Figure: The model emits a tool call; host code validates and runs it, then returns an observation to the bounded agent loop.*

**Tool use lets a model emit a function call as text that your code executes; an agent wraps such calls in a plan, act, observe loop with memory.** The model only writes text; your code runs the tool.

## The Problem: Some Things Are Not Text

Retrieval returns text, but arithmetic, today's date, a database row or a booking are not text. A model cannot do exact arithmetic reliably, but it can **write the call**. And one call is not enough for "plan a trip, check prices, rebook if sold out": put the model in a loop. That is an **agent**.

## Tool Use and Function Calling

> [!compiler] The idea
> Tool use turns generation into: *decide a tool is needed, emit the call as text, run it outside the model, paste the result back, continue.* The history: verifiers for math word problems, **TALM** (Tool Augmented Language Models), **PAL** (write a *program* as the reasoning), **Toolformer**, **WebGPT**.

> [!compiler] How a model learns to call tools
> **TALM:** a text-to-text interface plus an **iterative self-play loop** that starts from a *small* set of tool demonstrations, generates more, keeps those that lead to the right answer, and fine-tunes again. **Toolformer:** the model proposes where to insert API calls in ordinary text; a call is **kept only if inserting its result lowers the loss on the following tokens**, then fine-tuned on the filtered text. Both are self-supervised.

> [!compiler] EndSem Q0(e), worked
> *"TALMs require extensive pre-training on tool-specific datasets to effectively utilize external tools like calculators and search engines."* **False.** TALMs bootstrap from a small number of demonstrations by self-play and fine-tuning. No answer key is in the repo; this is the compiler's reading.

:::analogy
A lawyer (the model) dictates a note "call the registry and ask for plot 42"; an assistant (your code) makes the call and brings back the answer. The lawyer never touches the phone.
:::

```python
import json
TOOLS = {"add": lambda a, b: a + b, "weather": lambda city: {"Delhi": 31}.get(city, None)}

def dispatch(reply):
    call = json.loads(reply)
    if call["call"] not in TOOLS:                 # guard: hallucinated function name
        return {"error": "unknown tool"}
    return {"result": TOOLS[call["call"]](**call["args"])}

print(dispatch('{"call": "add", "args": {"a": 17, "b": 25}}'))
print(dispatch('{"call": "weather", "args": {"city": "Delhi"}}'))
print(dispatch('{"call": "launch_rocket", "args": {}}'))
```

```output
{'result': 42}
{'result': 31}
{'error': 'unknown tool'}
```

> [!compiler] Failure modes to guard
> A hallucinated function name, a malformed argument, a wrong choice among many APIs (Gorilla trains on API documentation), and executing a call with side effects (payments, deletes) without confirmation. The tool *result* is untrusted text: a web page can contain instructions (prompt injection), so never let a retrieved string escalate its own permissions.

## Agents: Reasoning, Planning, Memory

> [!compiler] The loop
> An **agent** is an LLM inside a control loop with tools and memory. **ReAct** interleaves *Thought*, *Action* and *Observation*:
> ```text
> state = [task]
> for step in range(max_steps):                    # always bound the loop
>     out = LLM(state)                              # "Thought: ... Action: search[X]"  or  "Final: ..."
>     if out.is_final: return out.answer
>     obs = run_tool(out.action)                    # external, validated
>     state += [out.thought, out.action, ("Observation", obs)]
> ```

- **Reasoning:** chain-of-thought inside the loop. Errors compound: if each step is right with probability 0.9, ten steps succeed with $0.9^{10}\approx0.35$.
- **Planning:** *decompose* then act. **ReWOO** plans all tool calls up front (fewer LLM calls, less adaptive). **HuggingGPT** uses the LLM as a controller that picks specialist models.
- **Memory:** **short-term** is the context window; **long-term** is an external store, usually the same retrieval machinery as RAG. **Reflexion** writes a verbal critique of a failed attempt into memory and retries; **Self-Refine** has the model critique and rewrite its own output.

```python
for p in (0.99, 0.9, 0.8):
    print(f"per-step accuracy {p}: 10 steps succeed with probability {p ** 10:.2f}")
```

```output
per-step accuracy 0.99: 10 steps succeed with probability 0.90
per-step accuracy 0.9: 10 steps succeed with probability 0.35
per-step accuracy 0.8: 10 steps succeed with probability 0.11
```

The honest summary: agents trade reliability for generality. Bound the steps, log every action, and keep a human in the loop for anything irreversible.

:::def
An **LLM agent** is a language model run in a loop that repeatedly reasons, chooses an action (usually a tool call), observes the result and updates its memory until it can produce a final answer.
:::

:::key
- The LLM only **emits text**; your code validates and executes.
- TALM and Toolformer learn from few demonstrations, not extensive tool-specific pre-training.
- Agents: ReAct loop, planning (ReWOO), memory (context window plus retrieval store); errors compound.
:::

:::mistakes
- "The LLM runs the tool." Your code does.
- "Agents are just longer prompts." They add a loop with side effects, so they need step limits and logging.
:::

:::exam
- *(EndSem Q0(e), 2 marks)* True/False with justification on TALMs.
:::

## Quick Check

:::q Q1 · What is the rule Toolformer uses to keep a proposed API call?
:::ans
It keeps the call only if inserting its result reduces the loss on the following tokens.
:::
:::

:::q Q2 · Why must an agent loop have a step limit?
:::ans
Errors compound and the loop can run forever or take costly actions; bounding it and logging each step contains the risk.
:::
:::
