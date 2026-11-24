# Week 7: Training Dynamics

**Mentor:** [James Ming Liang Ang](https://scholar.google.com/citations?user=hepV8W8AAAAJ&hl=en)\
**Date:** 5 Nov 2026

---

## How neural networks learn from data

Most of this course has looked at models *after* training: what they do, what personas they adopt, how they generalise. This week asks a different question: what happens *during* training? We study how a model's parameters, and therefore its predictions, change step by step as gradient descent runs.

The main reading works through a neural network small enough to solve by hand. Even this toy model shows three phenomena that appear in real networks: **conservation laws** that come from symmetries, **implicit bias** in which solution training picks, and **phase transitions** when a hyperparameter crosses a critical value.

### What you'll learn

By the end of the week, you will be able to:

* Say what training dynamics studies, and why some researchers argue that a science of AI must study it.
* Derive gradient flow as the small-step limit of gradient descent, and explain the "ball rolling downhill" analogy.
* Solve gradient flow for the two-weight linear network $$f(x) = abx$$, including its solution manifold and stationary points.
* Explain how a rescaling symmetry gives a conservation law, and how that law fixes which solution training reaches (implicit bias).
* Explain why the conservation law breaks for finite step sizes, and what can go wrong when the step size is too large.

### Prerequisites

Algebra up to quadratic equations, and basic calculus (derivatives, the chain rule, partial derivatives). The convergence proof in Section 3 of the reading uses some real analysis and can be skipped.

### Roadmap

The week runs as two 1-hour discussion sessions. Each session follows the same pattern.

| Time    | Session                | Breakdown                                               |
| ------- | ---------------------- | ------------------------------------------------------- |
| 20 min  | Individual reading     | Read the main reading for the unit                      |
| 20 min  | Small group discussion | Work through the derivations and exercises together     |
| 20 min  | Combined discussion    | Whole-group discussion led by the head TA               |

{% hint style="warning" %}
**Note for Odysea students.** The full tutorial that goes with the reading will be released publicly only after the programme ends. This gives you the chance to explore the extensions below yourselves, for example as a research project.
{% endhint %}

---

## Reading guide

This section summarises the main reading. It is a companion to the reading, not a replacement for it. Full citations are in the [References](#references) at the end of the chapter.

### 1. What is training dynamics about?

*Source: Ang (2026), Section 1.*

**Why study it.** There is a growing view that a science of AI should study how models *change* during learning, and not only the finished model. Biderman et al. (2026) argue this in an ICML 2026 position paper titled *Don't just "fix it in post"*. Simon et al. (2026) argue that an emerging "mechanics" of learning, with quantitative predictions, will become a scientific theory of deep learning.

**Definition.** Training dynamics is the study of how a model's parameters, internal representations and predictions evolve during training, and how this depends on the data, the architecture and the learning algorithm.

**The central question.** Following Ren and Sutherland (2025), which won an ICLR 2025 Outstanding Paper Award, the reading frames the field around one question: how does a change in parameters turn into a change in predictions?

$$\Delta\theta \;\longrightarrow\; \Delta f$$

**Gradient descent.** The *loss* $$L_{\mathcal{D}}(\theta)$$ measures how badly the model with weights $$\theta$$ fits the training data $$\mathcal{D}$$. For neural networks we cannot minimise it in closed form, so we take many small steps downhill:

$$\theta_{k+1} = \theta_k - \eta\,\nabla L_{\mathcal{D}}(\theta_k)$$

Here $$\eta > 0$$ is the learning rate. Picture the loss as a landscape and the weights as a ball: the negative gradient points in the direction of steepest descent, and the ball rolls downhill.

<figure><img src="https://jamesmldoesml.github.io/blogs/learning-dynamics/images/learning-dynamics.png" alt="Waddington&#x27;s epigenetic landscape, showing a ball above a branching landscape of ridges and valleys"><figcaption><p>C. H. Waddington's epigenetic landscape, from <em>The Strategy of the Genes</em> (1957). Originally a picture of cell development, used here as a visual analogy for a loss landscape. Image via the <a href="https://libraryblogs.is.ed.ac.uk/towardsdolly/2012/08/10/waddingtons-epigenetic-landscape/">University of Edinburgh Library</a>.</p></figcaption></figure>

**Gradient flow.** To move from discrete steps to continuous time, treat each step as taking time $$\eta$$, so that step $$k$$ happens at training time $$t = k\eta$$ and $$\theta_k = \theta(t)$$, $$\theta_{k+1} = \theta(t + \eta)$$. Rearranging the update rule gives the average rate of change of the weights per unit time:

$$\frac{\theta(t+\eta) - \theta(t)}{\eta} = \frac{\theta_{k+1} - \theta_k}{\eta} = -\nabla L_{\mathcal{D}}(\theta_k).$$

The left-hand side is a difference quotient. As $$\eta \to 0$$ it becomes a derivative, and the sequence of updates becomes a smooth trajectory:

$$\frac{d\theta}{dt} = \lim_{\eta \to 0} \frac{\theta(t+\eta) - \theta(t)}{\eta} = -\nabla L_{\mathcal{D}}(\theta)$$

This is **gradient flow**, the simplified ("Mickey Mouse") model of training used throughout the reading. It has exactly the same form as a ball moving through a potential $$V$$ under strong friction, $$\dot{\mathbf{r}} = -\nabla V(\mathbf{r})$$:

| Training                          | Physics                 |
| --------------------------------- | ----------------------- |
| weights $$\theta$$                | position $$\mathbf{r}$$ |
| training loss $$L_{\mathcal{D}}$$ | potential energy $$V$$  |
| training time $$t$$               | physical time $$t$$     |

<details>
<summary>Derivation: from Newton's law to gradient flow</summary>

A ball of mass $$m$$ in a potential $$V$$, with friction coefficient $$\gamma > 0$$ and physical time $$\tau$$, obeys

$$m\frac{d^2\mathbf{r}}{d\tau^2} = -\nabla V(\mathbf{r}) - \gamma\frac{d\mathbf{r}}{d\tau}.$$

When friction dominates, we can neglect the inertia term on the left:

$$\frac{d\mathbf{r}}{d\tau} \approx -\frac{1}{\gamma}\nabla V(\mathbf{r}).$$

Rescaling time with $$t = \tau/\gamma$$ gives $$\frac{d\mathbf{r}}{dt} = -\nabla V(\mathbf{r})$$.

</details>

### 2. A neural network you can solve by hand

*Source: Ang (2026), Section 2.*

**The model.** Take a network with two layers, one hidden unit and no activation function. The input is multiplied by $$b$$, then by $$a$$:

$$f(x) = a\,b\,x$$

We train it to learn the identity function $$y = x$$ from a dataset $$\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$$ with $$y_i = x_i$$, using mean squared error. Every term contains the factor $$(ab - 1)^2$$, so it comes out of the sum:

$$L_{\mathcal{D}}(a,b) = \frac{1}{2n}\sum_{i=1}^{n}(abx_i - x_i)^2 = \frac{1}{2}(ab-1)^2 \left(\frac{1}{n}\sum_{i=1}^{n}x_i^2\right).$$

The bracket is a fixed positive number that depends only on the inputs, not on the weights, so it does not change during training. We can set it to 1 (it can be absorbed into time or the learning rate), which gives the simplified loss

$$L_{\mathcal{D}}(a,b) = \tfrac{1}{2}(ab-1)^2.$$

**Gradient flow for this model.** Each weight changes at a rate set by the other weight:

$$\frac{da}{dt} = -\frac{\partial L_{\mathcal{D}}}{\partial a} = -(ab-1)\,b, \qquad \frac{db}{dt} = -\frac{\partial L_{\mathcal{D}}}{\partial b} = -(ab-1)\,a.$$

**The solution manifold.** The minimum loss is 0, reached at *every* point on the hyperbola $$ab = 1$$. This curve of equally good solutions is the **solution manifold**. The loss gives no reason to prefer $$(1, 1)$$ over $$(2, \tfrac{1}{2})$$. Whatever preference training shows among them comes from the initialisation and the algorithm. That preference is the **implicit bias** of training.

**Rescaling symmetry.** The map

$$(a,b) \longmapsto \left(\lambda a, \tfrac{b}{\lambda}\right), \qquad \lambda > 0$$

leaves $$ab$$, and therefore the output and the loss, unchanged. Under gradient flow, this symmetry gives a **conservation law**:

$$a(t)^2 - b(t)^2 = a(0)^2 - b(0)^2 =: c.$$

You can check this directly by plugging in the gradient flow equations:

$$\frac{d}{dt}(a^2 - b^2) = 2a\frac{da}{dt} - 2b\frac{db}{dt} = -2ab(ab-1) + 2ab(ab-1) = 0.$$

A quantity whose rate of change is always zero never changes. So, although there are infinitely many solutions, such as $$(1, 1)$$, $$(2, 0.5)$$ or $$(10, 0.1)$$, gradient flow from a given start is confined to one track, the curve $$a^2 - b^2 = c$$.

The direct check works for this model only. The symmetry argument explains *why* the law holds, and it carries over to other networks. Differentiating $$L(\lambda a, b/\lambda) = L(a,b)$$ with respect to $$\lambda$$ at $$\lambda = 1$$ gives $$a\,\partial_a L - b\,\partial_b L = 0$$. Under gradient flow, $$\frac{d}{dt}(a^2 - b^2) = 2a\dot a - 2b\dot b = -2(a\,\partial_a L - b\,\partial_b L) = 0$$. Differentiating a continuous symmetry at the identity like this is also the first step of Noether's theorem in physics.

{% hint style="info" %}
**Beyond linear networks.** The same symmetry appears in real architectures. ReLU is positively homogeneous, so you can scale the weights into a ReLU unit up and the weights out of it down. Gated feed-forward blocks such as SwiGLU and GeGLU, used in LLMs like Gemma and GPT-OSS, admit the same rescaling on their linear branch. (Some implementations clamp branch values for numerical stability, which breaks the exact symmetry.) In general, the conserved quantity is the difference between the squared norms of the two rescaled groups of parameters. Kunin et al. (2021) study these symmetries and the ways practical choices break them.
{% endhint %}

**Where does training end up?** Training moves along the curve $$a^2 - b^2 = c$$ set by the initialisation. The stationary points are the solution manifold $$ab = 1$$ (loss 0) and the origin $$(0,0)$$ (loss $$\tfrac{1}{2}$$).

* **Case $$c \neq 0$$.** The origin is not on the conserved curve, so training reaches a zero-loss solution. Solving $$ab = 1$$ and $$a^2 - b^2 = c$$ together gives

  $$a_\infty^2 = \frac{c + \sqrt{c^2+4}}{2}, \qquad b_\infty^2 = \frac{-c + \sqrt{c^2+4}}{2}.$$

  The initialisation alone picks out the endpoint: this is implicit bias made exact. (To check convergence, let $$p = ab$$. Then $$\dot p = \sqrt{c^2 + 4p^2}\,(1-p)$$, and the square root is strictly positive, so $$p \to 1$$.)
* **Case $$c = 0$$.** The curve splits into two lines. On $$a = b$$, every nonzero start reaches a zero-loss solution. On $$a = -b$$, the dynamics push both weights to the origin, a saddle point with loss $$\tfrac{1}{2}$$, and learning fails.

<figure><img src="https://jamesmldoesml.github.io/blogs/learning-dynamics/images/learning-dynamics-phase-portrait.png" alt="Phase portrait of gradient flow for y = abx: trajectories above a = -b reach positive solutions on ab = 1; those below reach negative solutions; trajectories on a = -b flow into the saddle at the origin."><figcaption><p>Phase portrait of gradient flow for <code>f(x) = abx</code>. Black curves: the zero-loss solutions <code>ab = 1</code>. Starts with <code>a₀ + b₀ &gt; 0</code> reach positive weights; starts with <code>a₀ + b₀ &lt; 0</code> reach negative weights. Both compute <code>f(x) = x</code>. Starts exactly on the red line <code>a = −b</code> flow into the origin. After Saxe, McClelland and Ganguli (2014), Figure 2.</p></figcaption></figure>

{% hint style="info" %}
**Terminology.** *Gradient flow* is the continuous-time limit of gradient descent. The *solution manifold* is the set of all parameters with minimum loss. *Implicit bias* is the preference among equally good solutions that comes from the initialisation and the training algorithm, not from the loss. A *conservation law* is a quantity that stays constant along every training trajectory.
{% endhint %}

### 3. Finite step sizes

*Source: Ang (2026), Section 3.*

Gradient flow is a model, and all models are wrong. Real training uses a finite learning rate $$\eta > 0$$:

$$a_{t+1} = a_t - \eta\, b_t(a_t b_t - 1), \qquad b_{t+1} = b_t - \eta\, a_t(a_t b_t - 1).$$

Write $$r_t = a_t b_t - 1$$ for the residual, so the updates become $$a_{t+1} = a_t - \eta b_t r_t$$ and $$b_{t+1} = b_t - \eta a_t r_t$$. One step then gives

$$a_{t+1}^2 - b_{t+1}^2 = \left(1 - \eta^2 r_t^2\right)(a_t^2 - b_t^2).$$

<details>
<summary>Derivation</summary>

Expand both squares:

$$a_{t+1}^2 = a_t^2 - 2\eta a_t b_t r_t + \eta^2 b_t^2 r_t^2, \qquad b_{t+1}^2 = b_t^2 - 2\eta a_t b_t r_t + \eta^2 a_t^2 r_t^2.$$

The cross terms cancel when you subtract, leaving $$(a_t^2 - b_t^2) - \eta^2 r_t^2 (a_t^2 - b_t^2)$$.

</details>

So the conservation law **breaks**: each step multiplies the imbalance $$a^2 - b^2$$ by $$1 - \eta^2 r_t^2$$, which is less than 1 whenever the model has not yet fitted the data ($$r_t \neq 0$$). The imbalance drifts over training instead of staying fixed. The factor is exactly 1 only in the limit $$\eta \to 0$$, which is gradient flow. When the step is small, the factor is close to 1 and the gradient flow picture is a good approximation.

**Too large a step diverges.** With $$\eta = 1$$ and $$a_0 = b_0 = 2$$, the weights go $$2 \mapsto -4 \mapsto 56 \mapsto -175504 \mapsto \cdots$$ and blow up.

**Small enough steps converge.** For any fixed initialisation, a small enough constant learning rate makes the weights converge, and unless $$a_0 = -b_0$$ they reach a global minimum. In this regime, the size of the imbalance $$|a^2 - b^2|$$ can only stay the same or shrink. This result holds for a **non-convex** loss with a **constant** learning rate. Many textbook convergence proofs need a convex loss or a decaying learning rate.

<details>
<summary>Proof sketch (advanced)</summary>

1. Choose $$0 < \eta < \dfrac{1}{4(a_0^2 + b_0^2 + 2)}$$. This keeps the weights bounded, keeps $$|\eta r_t| < \tfrac{1}{4}$$, and stops $$r_t$$ from changing sign.
2. The sum and difference of the weights update multiplicatively:

   $$a_{t+1} + b_{t+1} = (1 - \eta r_t)(a_t + b_t), \qquad a_{t+1} - b_{t+1} = (1 + \eta r_t)(a_t - b_t).$$

   Both factors are positive. Since $$r_t$$ keeps its sign, one of the two magnitudes increases monotonically and the other decreases. Both are bounded, so both converge (monotone convergence theorem), and therefore $$a_t$$ and $$b_t$$ converge to limits $$A$$ and $$B$$.
3. At the limit, $$A = A - \eta B(AB - 1)$$ and $$B = B - \eta A(AB-1)$$. So either $$AB = 1$$ (zero loss) or $$A = B = 0$$. A sign argument excludes the origin whenever $$a_0 + b_0 \neq 0$$.

See the reading for the full proof.

</details>

{% hint style="info" %}
**Convexity.** A function is *convex* if the chord between any two points on its graph lies on or above the graph, like $$x^2$$. For convex functions, every local minimum is a global minimum. The loss $$\tfrac{1}{2}(ab-1)^2$$ is not convex in $$(a, b)$$.
{% endhint %}

### 4. Weight decay and a phase transition

*Source: Ang (2026), Section 4.*

When weight decay with strength $$\gamma$$ is added, the toy model shows a **phase transition**. Below a critical value ($$\gamma = \tfrac{1}{2}$$ in the reading's setup), training reaches a stable learned solution. Above it, the only stable solution is the zero network, so the network stops fitting $$y = x$$. At larger step sizes, oscillation and divergence also appear.

<figure><img src="https://jamesmldoesml.github.io/blogs/learning-dynamics/additional-resources/figures/step-size-phase-diagram.png" alt="Phase diagram for learning y = x with f(x) = abx. Analytic stability boundaries separate stable learned solutions from the stable zero solution at weight decay gamma = one half. Oscillations and divergence beyond the stability boundaries are observed numerically."><figcaption><p>Phase diagram for learning <code>y = x</code> with <code>f(x) = abx</code>, in step size and weight decay. Analytic stability boundaries separate the learned phase from the zero-output phase at <code>γ = 1/2</code>. Oscillation and divergence are observed numerically.</p></figcaption></figure>

Near the critical point, training shows **critical slowing down**: the time to converge grows without bound as $$\gamma \to \tfrac{1}{2}$$, and at the critical point convergence becomes algebraic instead of exponential. Weight decay plays a role like temperature in a material that changes phase.

### Summary

Back to the opening question: how does a change in parameters $$\Delta\theta$$ turn into a change in predictions $$\Delta f$$? The toy model shows that parameter updates are not independent of each other.

* Under gradient flow, the weights are constrained by conservation laws, here $$a(t)^2 - b(t)^2 = a_0^2 - b_0^2$$. Each law comes from a symmetry of the network.
* These laws create an implicit bias. Out of infinitely many zero-loss solutions, the initialisation decides which one training reaches.
* With finite step sizes, the conservation law breaks and the imbalance between the weights drifts. If the step is too large, training diverges.

Studying these mechanics is one way to stop treating neural networks as black boxes. If we understand step by step how models learn, we have a better chance of predicting, and eventually controlling, what they learn and how they generalise.

---

## Exercises

Try these in your small group. They are the extensions suggested at the end of the reading, and good starting points for a research project.

1. **Check the algebra.** Without opening the derivation above, verify the finite-step identity $$a_{t+1}^2 - b_{t+1}^2 = (1 - \eta^2 r_t^2)(a_t^2 - b_t^2)$$, and reproduce the divergent sequence for $$\eta = 1$$, $$a_0 = b_0 = 2$$.
2. **Implicit bias in numbers.** Start gradient flow from $$(a_0, b_0) = (2, 0.1)$$. Which point on $$ab = 1$$ does training reach? Compare it with the start $$(0.1, 2)$$.
3. **Add a ReLU.** Replace the hidden unit with $$\mathrm{ReLU}(bx)$$. Which conservation law survives? What new failure modes appear?
4. **Add weight decay.** Add $$\gamma$$-weighted weight decay to the loss. Is $$a^2 - b^2$$ still conserved? Find the stationary points and reproduce the phase transition.
5. **Go deeper.** Use three or more layers, $$f(x) = a_1 a_2 \cdots a_L x$$. What are the conserved quantities now?
6. **Label noise.** Add noise to the targets. How does label noise interact with weight decay? (See Damian, Ma and Lee (2021) and Blanc et al. (2020) for related results.)

## Discussion questions

* The reading claims the toy model shows "small versions" of phenomena in large networks. Which of conservation laws, implicit bias and phase transitions do you expect to carry over to LLMs, and which are artefacts of the toy model?
* Implicit bias means the loss alone does not decide which solution a network learns. What does this mean for alignment, where we often specify what we want only through a loss or a reward?
* Biderman et al. (2026) argue we should not just "fix it in post". How could knowledge of training dynamics change how we do alignment training, compared with only evaluating finished models?
* Gradient flow ignores finite step sizes, stochastic minibatches and adaptive optimisers like Adam. When is a simplified model like this useful, and when is it misleading? (Cohen et al. (2021) on the "edge of stability" is a good example of where it breaks.)

---

## References

### Main reading

* Ang, J. M. L. (2026). *Training Dynamics*. Science of Deep Learning: Taster Series, October 2026. [https://jamesmldoesml.github.io/blogs/learning-dynamics/](https://jamesmldoesml.github.io/blogs/learning-dynamics/).

### Works cited in the reading

* Biderman, S., Khan, M. A., Mireshghallah, N., Arnett, C., Barez, F. and Saphra, N. (2026). *Position: Don't just "fix it in post": A science of AI must study training dynamics*. ICML 2026, PMLR 306, 169526–169542. [https://proceedings.mlr.press/v306/biderman26a.html](https://proceedings.mlr.press/v306/biderman26a.html).
* Kunin, D., Sagastuy-Brena, J., Ganguli, S., Yamins, D. L. K. and Tanaka, H. (2021). *Neural mechanics: Symmetry and broken conservation laws in deep learning dynamics*. ICLR. [https://openreview.net/forum?id=q8qLAbQBupm](https://openreview.net/forum?id=q8qLAbQBupm).
* Ren, Y. and Sutherland, D. J. (2025). *Learning dynamics of LLM finetuning*. ICLR. [https://proceedings.iclr.cc/paper_files/paper/2025/hash/afe1aa79e5eea7955f553c61a307273e-Abstract-Conference.html](https://proceedings.iclr.cc/paper_files/paper/2025/hash/afe1aa79e5eea7955f553c61a307273e-Abstract-Conference.html).
* Saxe, A. M., McClelland, J. L. and Ganguli, S. (2014). *Exact solutions to the nonlinear dynamics of learning in deep linear neural networks*. arXiv:1312.6120. [https://arxiv.org/abs/1312.6120](https://arxiv.org/abs/1312.6120).
* Simon, J., Kunin, D., Atanasov, A., Boix-Adserà, E., Bordelon, B., Cohen, J., Ghosh, N., Guth, F., Jacot, A., Kamb, M., Karkada, D., Michaud, E. J., Ottlik, B. and Turnbull, J. (2026). *There will be a scientific theory of deep learning*. arXiv:2604.21691. [https://arxiv.org/abs/2604.21691](https://arxiv.org/abs/2604.21691).

### Further reading

* Blanc, G., Gupta, N., Valiant, G. and Valiant, P. (2020). *Implicit regularization for deep neural networks driven by an Ornstein-Uhlenbeck like process*. COLT, PMLR 125, 483–513. [https://proceedings.mlr.press/v125/blanc20a.html](https://proceedings.mlr.press/v125/blanc20a.html).
* Cohen, J. M., Kaur, S., Li, Y., Kolter, J. Z. and Talwalkar, A. (2021). *Gradient descent on neural networks typically occurs at the edge of stability*. ICLR. [https://openreview.net/forum?id=jh-rTtvkGeM](https://openreview.net/forum?id=jh-rTtvkGeM).
* Damian, A., Ma, T. and Lee, J. D. (2021). *Label noise SGD provably prefers flat global minimizers*. NeurIPS 34. [https://papers.neurips.cc/paper_files/paper/2021/hash/e6af401c28c1790eaef7d55c92ab6ab6-Abstract.html](https://papers.neurips.cc/paper_files/paper/2021/hash/e6af401c28c1790eaef7d55c92ab6ab6-Abstract.html).
