# 📘 Machine Learning Mathematical Symbols Reference

This document lists the **most commonly used mathematical symbols** in **Machine Learning (ML)** and **Statistics**, along with their meanings, **LaTeX**, and **MathML** representations.

> 🧠 Use this as a quick reference or cheat sheet while writing papers, notebooks, or documentation involving math formulas.

---

## 🧩 1. Sets, Spaces, and Data

| Symbol | Meaning | LaTeX | MathML |
|:--|:--|:--|:--|
| 𝓧 | Input space (all possible inputs) | `\mathcal{X}` | `<mi mathvariant="script">X</mi>` |
| 𝓨 | Output space (labels) | `\mathcal{Y}` | `<mi mathvariant="script">Y</mi>` |
| 𝓓 | Dataset = { (xᵢ, yᵢ) } | `\mathcal{D}` | `<mi mathvariant="script">D</mi>` |
| ℝ | Real numbers | `\mathbb{R}` | `<mi mathvariant="double-struck">R</mi>` |
| ℕ | Natural numbers | `\mathbb{N}` | `<mi mathvariant="double-struck">N</mi>` |
| ℤ | Integers | `\mathbb{Z}` | `<mi mathvariant="double-struck">Z</mi>` |

---

## 🔢 2. Variables and Parameters

| Symbol | Meaning | LaTeX | MathML |
|:--|:--|:--|:--|
| **x** | Input (feature vector) | `\boldsymbol{x}` | `<mi mathvariant="bold-italic">x</mi>` |
| **θ**, **w** | Parameters / weights | `\boldsymbol{\theta}` | `<mi mathvariant="bold-italic">θ</mi>` |
| y | Output / target | `y` | `<mi>y</mi>` |
| ŷ | Predicted value | `\hat{y}` | `<mover><mi>y</mi><mo>^</mo></mover>` |
| ϵ | Noise term | `\varepsilon` | `<mi>ϵ</mi>` |

---

## 🧠 3. Probability and Statistics

| Symbol | Meaning | LaTeX | MathML |
|:--|:--|:--|:--|
| p(x) | Probability density | `p(x)` | `<mi>p</mi><mo>(</mo><mi>x</mi><mo>)</mo>` |
| p(y \| x) | Conditional probability | `p(y \mid x)` | `<mi>p</mi><mo>(</mo><mi>y</mi><mo>|</mo><mi>x</mi><mo>)</mo>` |
| 𝔼[x] | Expectation | `\mathbb{E}[x]` | `<mi mathvariant="double-struck">E</mi><mo>[</mo><mi>x</mi><mo>]</mo>` |
| Var[x] | Variance | `\mathrm{Var}[x]` | `<mi>Var</mi><mo>[</mo><mi>x</mi><mo>]</mo>` |
| Cov(x, y) | Covariance | `\mathrm{Cov}(x, y)` | `<mi>Cov</mi><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>)</mo>` |
| 𝒩(μ, σ²) | Normal distribution | `\mathcal{N}(\mu, \sigma^2)` | `<mi mathvariant="script">N</mi><mo>(</mo><mi>μ</mi><mo>,</mo><msup><mi>σ</mi><mn>2</mn></msup><mo>)</mo>` |

---

## ⚙️ 4. Linear Algebra

| Symbol | Meaning | LaTeX | MathML |
|:--|:--|:--|:--|
| Xᵀ | Transpose | `X^{\top}` | `<msup><mi>X</mi><mi>T</mi></msup>` |
| A⁻¹ | Matrix inverse | `A^{-1}` | `<msup><mi>A</mi><mo>-1</mo></msup>` |
| ∥x∥₂ | Euclidean norm | `\|x\|_2` | `<mfenced><mi>x</mi></mfenced><msub><mo>‖</mo><mn>2</mn>` |
| det(A) | Determinant | `\det(A)` | `<mi>det</mi><mo>(</mo><mi>A</mi><mo>)</mo>` |
| Tr(A) | Trace | `\mathrm{Tr}(A)` | `<mi>Tr</mi><mo>(</mo><mi>A</mi><mo>)</mo>` |

---

## 🔧 5. Optimization and Gradients

| Symbol | Meaning | LaTeX | MathML |
|:--|:--|:--|:--|
| min L(θ) | Minimize loss | `\min_\theta L(\theta)` | `<munder><mo>min</mo><mi>θ</mi></munder><mi>L</mi><mo>(</mo><mi>θ</mi><mo>)</mo>` |
| argmin | Value minimizing function | `\arg\min_\theta` | `<munder><mo>argmin</mo><mi>θ</mi></munder>` |
| ∇θL | Gradient | `\nabla_\theta L` | `<mi>∇</mi><msub><mi>L</mi><mi>θ</mi></msub>` |
| ∂L/∂θᵢ | Partial derivative | `\frac{\partial L}{\partial \theta_i}` | `<mfrac><mrow><mo>∂</mo><mi>L</mi></mrow><mrow><mo>∂</mo><msub><mi>θ</mi><mi>i</mi></msub></mrow></mfrac>` |

---

## 🧮 6. Common ML Functions

| Symbol | Meaning | LaTeX | MathML |
|:--|:--|:--|:--|
| f(x; θ) | Model (parametric function) | `f(x; \theta)` | `<mi>f</mi><mo>(</mo><mi>x</mi><mo>;</mo><mi>θ</mi><mo>)</mo>` |
| hθ(x) | Hypothesis function | `h_\theta(x)` | `<msub><mi>h</mi><mi>θ</mi></msub><mo>(</mo><mi>x</mi><mo>)</mo>` |
| σ(z) = 1 / (1 + e⁻ᶻ) | Sigmoid | `\sigma(z) = \frac{1}{1 + e^{-z}}` | `<mi>σ</mi><mo>(</mo><mi>z</mi><mo>)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo>-</mo><mi>z</mi></mrow></msup></mrow></mfrac>` |
| ReLU(x) = max(0, x) | ReLU activation | `\mathrm{ReLU}(x) = \max(0, x)` | `<mi>ReLU</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>=</mo><mi>max</mi><mo>(</mo><mn>0</mn><mo>,</mo><mi>x</mi><mo>)</mo>` |

---

## 📉 7. Loss Functions

| Symbol | Meaning | LaTeX | MathML |
|:--|:--|:--|:--|
| L(θ) | Loss function | `L(\theta)` | `<mi>L</mi><mo>(</mo><mi>θ</mi><mo>)</mo>` |
| ℓ(y, ŷ) | Per-sample loss | `\ell(y, \hat{y})` | `<mi>ℓ</mi><mo>(</mo><mi>y</mi><mo>,</mo><mover><mi>y</mi><mo>^</mo></mover><mo>)</mo>` |
| MSE = (1/N) Σ (yᵢ − ŷᵢ)² | Mean squared error | `\frac{1}{N}\sum_i (y_i - \hat{y}_i)^2` | `<mfrac><mn>1</mn><mi>N</mi></mfrac><mo>∑</mo><msup><mo>(</mo><mrow><msub><mi>y</mi><mi>i</mi></msub><mo>-</mo><mover><msub><mi>y</mi><mi>i</mi></msub><mo>^</mo></mover></mrow><msup><mo>)</mo><mn>2</mn></msup>` |
| CE = −Σ yᵢ log(ŷᵢ) | Cross entropy | `-\sum_i y_i \log(\hat{y}_i)` | `<mo>-</mo><mo>∑</mo><msub><mi>y</mi><mi>i</mi></msub><mi>log</mi><mo>(</mo><mover><msub><mi>y</mi><mi>i</mi></msub><mo>^</mo></mover><mo>)</mo>` |

---

## 🔗 8. Logical and Set Symbols

| Symbol | Meaning | LaTeX | MathML |
|:--|:--|:--|:--|
| ∈ | "is an element of" | `\in` | `<mo>∈</mo>` |
| ∀ | "for all" | `\forall` | `<mo>∀</mo>` |
| ∃ | "there exists" | `\exists` | `<mo>∃</mo>` |
| ∑ | Summation | `\sum` | `<mo>∑</mo>` |
| ∏ | Product | `\prod` | `<mo>∏</mo>` |
| ∝ | Proportional to | `\propto` | `<mo>∝</mo>` |
| ⇒ | Implies | `\Rightarrow` | `<mo>⇒</mo>` |
| ⇔ | If and only if | `\Leftrightarrow` | `<mo>⇔</mo>` |
| ∼ | Distributed as | `\sim` | `<mo>∼</mo>` |
| ∂ | Partial derivative | `\partial` | `<mo>∂</mo>` |
| ∇ | Gradient | `\nabla` | `<mo>∇</mo>` |
| ∫ | Integral | `\int` | `<mo>∫</mo>` |

---

## 🧾 Notes

- **MathML** is the web markup standard for representing math (used in browsers, Jupyter, and web apps).
- **LaTeX** is the preferred notation for scientific papers, notebooks, and Markdown.
- **Bold or calligraphic** fonts (𝓧, 𝓨, **x**, **θ**) often indicate **vectors, matrices, or sets**.

---

## 🧰 Recommended Tools

- [MathML Playground](https://mathml.playground.vercel.app/) — test and preview MathML
- [MathJax Docs](https://docs.mathjax.org/) — render LaTeX and MathML in browsers
- [Detexify](https://detexify.kirelabs.org/classify.html) — find LaTeX symbols by drawing

---

### 📚 Example Usage

Linear regression model in MathML and LaTeX:

```latex
y = \mathbf{x}^\top \boldsymbol{\theta} + \varepsilon, \quad
\varepsilon \sim \mathcal{N}(0, \sigma^2)
