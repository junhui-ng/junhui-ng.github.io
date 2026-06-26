---
layout: ../../layouts/BlogPost.astro
title: "Deconstructing the Transformer: Design Choices and Mathematical Foundations"
date: "2025-12-14"
desc: "A mathematical breakdown of Self-Attention and the breakdown of the seminal paper 'Attention Is All You Need'."
---

The Transformer [[1]](#transformer) architecture has been the backbone of recent language (and vision) models. A large number of technical reports and articles discussing the architecture are already available online. This blog does not aim to add yet another redundant overview. Instead, we focus on less obvious details of the architecture that are not discussed or clearly explained in the original paper, with the hope of addressing common theoretical and implementation-related questions.

---

## Scaled Dot-Product Attention

The core of the transformer architecutre lies in the scaled dot-product attention.
Given an input matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$, we
form three representations of $\mathbf{X}$ via learnable linear
projection. The three representations are named query
$\mathbf{Q} \in \mathbb{R}^{n \times d_k}$, key
$\mathbf{K} \in \mathbb{R}^{n \times d_k}$, and value
$\mathbf{V} \in \mathbb{R}^{n \times d_v}$. In the original transformer
paper and in most transformer-based LLMs, $d_k = d_v$. 
$$
\begin{align*}
    \mathbf{Q} &= \mathbf{X}\mathbf{W}_q \\
    \mathbf{K} &= \mathbf{X}\mathbf{W}_k \\
    \mathbf{V} &= \mathbf{X}\mathbf{W}_v
\end{align*}
$$ 
where
$\mathbf{W}_q,\mathbf{W}_k \in \mathbb{R}^{d \times d_k}$ and
$\mathbf{W}_v \in \mathbb{R}^{d \times d_v}$ are the learnable weights
for the query, key, and value, respectively.

The single-head scaled dot-product attention is then given by:
$$
\begin{equation}
    \text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{Softmax}\left( \frac{\mathbf{QK}^\top}{\sqrt{d_k}} \right) \mathbf{V}
\end{equation}
$$ 

The following subsections address questions that are commonly asked during lectures and interviews.

### Query-Key Dot-Product Relevance

The matrix multiplication $\mathbf{QK}^\top$ is essentially the dot 
product between all query and key pairs, $[\mathbf{QK}^\top]_{ij} = \mathbf{q}_i \mathbf{k}_j$.

**Why dot-product?**

Geometrically, the dot product of two vectors indicates how aligned the
vectors are. The larger the dot product between the query-key pair, the
stronger the relationship between them. This means that the magnitude of
$[\mathbf{QK}^\top]_{ij}$ indicates the relevance between the vectors at 
the $i$-th and $j$-th position.

### Scaling factor $\frac{1}{\sqrt{d_k}}$

The matrix multiplication $\mathbf{QK}^\top$ is scaled
by a factor $\frac{1}{\sqrt{d_k}}$, followed by a Softmax operator. For
each query $\mathbf{q}_i$, the Softmax function maps the scaled
dot-product to the probability distribution over all query-key pairs.
This gives us the amount of attention to pay to a particular token.

Two questions arise regarding the scaling factor $\frac{1}{\sqrt{d_k}}$:

1.  **Why do we scale the dot-product?**\
    When $d_k$ is large, the dot product $\mathbf{QK}^\top$ grows in
    magnitude, pushing Softmax into regions that have extremely small
    gradients. This is known as the vanishing-gradient problem.
    Therefore, the authors propose to scale the dot product by
    $\frac{1}{\sqrt{d_k}}$ to counteract the effect.

2.  **Why is the denominator $\sqrt{d_k}$ instead of other monotonically 
    increasing functions of $d_k$?**\
    The authors did not justify the selection of this particular scaling
    factor. One explanation is that it maintains the distribution of zero-mean 
    and unit variance (explained in [@zhang2024scaling]). This
    is based on the assumption that model weights are carefully
    initialized such that the activations have zero-mean and
    unit variance across layers. This assumption is
    usually valid in practice because the commonly used initialization
    methods were designed to have this property
    [@glorot-init; @kaiming-init]. Under this assumption, the matrices
    $\mathbf{Q}, \mathbf{K} \overset{\cdot}{\sim} \mathcal{N}(0,1)$
    (i.i.d.), which gives $\text{Var}[\mathbf{QK}^\top]=d_k$. Scaling it
    by factor $\frac{1}{\sqrt{d_k}}$ results in
    $\text{Var}[\frac{\mathbf{QK}^\top}{\sqrt{d_k}}]=1$.

---

## Multi-Head Scaled Dot-Product Attention

To allow the model to jointly attend to information from different
representation subspaces at different positions, the scaled dot-product
attention can be extended by adding multiple attention heads.

Recall that for a given input matrix $\mathbf{X}$ we linearly project it
into three representations $\mathbf{Q},\mathbf{K},\mathbf{V}$ and
compute the scaled dot-product attention. We refer to this as
single-head attention from this point onward. Instead of a single
attention head, we project the input matrix into $h$ different attention
heads, 
$$
\begin{equation}
    \text{head}_i = \text{Attention}(\mathbf{XW}^{(q)}_i,\mathbf{XW}^{(k)}_i,\mathbf{XW}^{(v)}_i), \quad\quad i=1,\dots,h
\end{equation}
$$ 
where $\mathbf{W}^{(q)}_i,\mathbf{W}^{(k)}_i \in \mathbb{R}^{d \times d_k}$
and $\mathbf{W}^{(v)}_i \in \mathbb{R}^{d \times d_v}$ are the learnable
weights for the query, key, and value in the $i$-th attention head,
respectively.

The attention heads are concatenated once again projected, resulting in
final values. 
$$
\begin{equation}
    \text{MultiHeadAttention}(\mathbf{X}) = 
    \begin{bmatrix}
        \text{head}_1 \\
        \text{head}_2 \\
        \dots \\
        \text{head}_h
    \end{bmatrix} \mathbf{W}^{(O)}
\end{equation}
$$ 
where $\mathbf{W}^{(O)} \in \mathbb{R}^{hd_v \times d}$
is a learnable weight.

Naively extending the attention to multiple heads increases the number
of parameters linearly. To reduce computational cost, the authors
propose to reduce dimensions $d_k$ and $d_v$ according to the number of
heads $h$, to be specific, $d_k = d_v = \frac{d}{h}$. Due to the reduced
dimension of each head, the computation cost of multi-head attention is
similar to that of single-head attention.

### Implementation Tips

In actual implementation, we do not initialize the multi-heads weight separately.
This is costly both in terms of computation and space. Instead, we treat them

### Why Multi-head?

Knowing how multi-head attention is performed in actual implementation, one might
ask: "Why multi-head representations instead of one large representation?"

Multi-head attention projects the vector into multiple representations, each head attends 
to different aspects. If we use a single large representation, the effects will be 
reduced by the softmax operation, because the Softmax normalization would force the 
model to average out these distinct relationships.

---

## Sinusoidal Encodes Relative Positions

Relative position is important in Transformers because language is
defined by the **relationships between words**, not their absolute
location in a sentence. Sinusoidal positional encoding naturally
captures relative position information.

Recall that in Section [2.2](#sec:pos-emb){reference-type="ref"
reference="sec:pos-emb"}, we provided the equation of sinusoidal
positional encoding: 
$$
\begin{equation*}
    [\mathbf{P}]_{ij} = 
    \begin{cases}
        \sin\left(\dfrac{i}{10000^{j/d}}\right) & \text{if } j \text{ is even} \\[10pt]
        \cos\left(\dfrac{i}{10000^{(j-1)/d}}\right) & \text{if } j \text{ is odd}
    \end{cases}
\end{equation*}
$$


**Theorem 1**. *The dot product of two sinusoidal positional encoding
vectors, $\mathbf{p}_k$ and $\mathbf{p}_j$, is a function that depends
only on their relative distance $(k-j)$.*

*Proof.* For a given frequency $\omega$, let 
$$
\begin{align*}
        \text{PE}(pos) &= 
        \begin{bmatrix}
            \sin(\omega \cdot pos) \\
            \cos(\omega \cdot pos)
        \end{bmatrix} \\
        \text{PE}(pos + k) &= 
        \begin{bmatrix}
            \sin(\omega \cdot (pos + k)) \\
            \cos(\omega \cdot (pos + k))
        \end{bmatrix} =
        \begin{bmatrix}
            \sin(\omega \cdot pos + \omega \cdot k) \\
            \cos(\omega \cdot pos + \omega \cdot k)
        \end{bmatrix}
\end{align*}
$$ 
be two positional encoding vectors with relative distance $k$.

Recall the trigonometric identity, 
$$
\begin{align*}
        \sin (a \pm b) &= \sin(a)\cos(b) \pm \cos(a)\sin(b) \quad \text{and} \\
        \cos (a \pm b) &= \cos(a)\cos(b) \mp \sin(a)\sin(b),
\end{align*}
$$

Then the dot product between the two vectors is 
$$
\begin{align*}
        \text{PE}(pos) \cdot \text{PE}(pos + k) &= \sin(\omega \cdot pos) \sin(\omega \cdot pos + \omega \cdot k) + \cos(\omega \cdot pos) \cos(\omega \cdot pos + \omega \cdot k) \\
        &= \cos(\omega \cdot pos - \omega \cdot pos + \omega \cdot k) \\
        &= \cos(\omega \cdot k).
\end{align*}
$$

The dot product between two sinusoidal position encoding vectors is a
function of their relative position $k$.

---
## References

1. <a id="transformer"></a> Vaswani A, et al. Attention Is All You Need. *NeurIPS*. 2017.

---

Reach out if you found the content to be inaccurate.
