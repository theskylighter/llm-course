**Every runnable example is executed by the validator, and the output shown under it is compared with the real output.** That is how the pages stay trustworthy.

## The Problem: Examples That Quietly Lie

A snippet copied into a lesson without running it will, sooner or later, contain a typo, a missing include, or an output that was worked out by hand and is wrong. Students trust the page, not their own doubt.

## How the Validator Works

A fenced block in a supported language is a *program*. If the next fenced block is tagged `output`, the program's real stdout must match it.

```python
#@input: 3 4
a, b = map(int, input().split())
print(f"sum = {a + b}")
```

```output
sum = 7
```

Directives are comments inside the fence. They are hidden on the rendered page:

| Directive (Python) | Directive (C++) | Meaning |
|---|---|---|
| `#@input: text` | `//@input: text` | a line fed to standard input |
| `#@expect: error` | `//@expect: compile-error` | the block must fail on purpose |
| `#@norun` | `//@norun` | build or syntax-check only |
| `#@nocheck` | `//@nocheck` | skip the block |

## A Block That Must Fail

Teaching a common error is easier when the page proves it really is an error:

```python
#@expect: error
print(undefined_name)
```

## Run It

```text
python tools/sitegen.py check          # all lessons
python tools/sitegen.py check 01_2     # only lessons whose file name contains 01_2
```

:::mistakes
- Working out the expected output by hand instead of running the program and pasting the result.
- Leaving an intentional infinite loop without `norun`, which makes the checker time out.
:::

:::key
- One fenced block = one program; the `output` block that follows is checked.
- Use directives for input, deliberate failures and skipped blocks.
:::

## Quick Check

:::q Q1 · What happens if the output block under a program is out of date?
:::ans
`check` reports an **output mismatch** with the expected and actual text, and exits non-zero.
:::
:::
