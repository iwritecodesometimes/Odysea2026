# Week 8: Singular Learning Theory

**Mentor:** [James Ming Liang Ang](https://scholar.google.com/citations?user=hepV8W8AAAAJ&hl=en)\
**Date:** 12 Nov 2026

---

## Learning as inference, and why neural networks are singular

Last week we watched a model's parameters move during training. This week we ask what the *shape* of the loss landscape tells us about learning. We treat machine learning as **statistical inference**, and use Bayes' rule to ask how well a whole class of models fits the data. Classical statistics answers this with a neat formula, the Bayesian information criterion (BIC). But that formula assumes the model is *regular*, and neural networks are not. **Singular learning theory** (SLT) replaces the parameter count in the formula with a new measure of complexity, the **learning coefficient**. This measure reflects the geometry of the solutions, not just how many parameters there are.

### What you'll learn

By the end of the week, you will be able to:

* Derive Bayes' rule for events, and state it for model parameters.
* Explain what it means to treat machine learning as statistical inference, and contrast maximum likelihood with Bayesian inference.
* Explain what the marginal likelihood measures, and how it relates to the free energy in physics.
* Define the KL divergence and the Fisher information, and explain how they give a geometry on models.
* Explain what makes a model *singular*, using the model $$y = abx$$ from [Week 7](../07-training-dynamics/README.md).
* Use BIC to compare two models, and explain how SLT replaces the parameter count $$d$$ with the learning coefficient $$\lambda$$.
* State the main practical limits of SLT, and one way it is used to study neural networks.

### Prerequisites

Basic probability (conditional probability, probability densities), calculus including Taylor expansions, and some linear algebra (matrices, eigenvalues). Week 7 is helpful but not required.

### Roadmap

The week runs as two 1-hour sessions. Each session follows the same pattern.

| Time    | Session                | Breakdown                                               |
| ------- | ---------------------- | ------------------------------------------------------- |
| 20 min  | Lecture                | Introduction to the week's topics                       |
| 20 min  | Small group discussion | Work through the derivations and exercises together     |
| 20 min  | Combined discussion    | Whole-group discussion led by the head TA               |

Session 1 covers Sections 1–4: Bayes, inference and information geometry. Session 2 covers Sections 5–8: singular models, BIC and the learning coefficient.

### Optional readings

| Reading | What it adds | Used in |
| ------- | ------------ | ------- |
| Rasmussen and Ghahramani (2001), *Occam's razor* | Bayesian inference prefers simple *functions*, even in very large models | Section 3 |
| Murray and Ghahramani (2005), *A note on the evidence and Bayesian Occam's razor* | A worked example of how the marginal likelihood penalises unnecessary complexity | Section 3 |
| Lotfi et al. (2022), *Bayesian model selection, the marginal likelihood, and generalization* | Where the marginal likelihood fails as a guide to generalisation | Section 6 |
| Lau et al. (2023), *The local learning coefficient* | How to measure the learning coefficient of a real neural network | Section 7 |

Full citations and links are in the [References](#references).

---

## Summary

### 1. Bayes' rule

#### For events

Let $$A$$ and $$B$$ be events (sets of outcomes). Bayes' rule says:

$$
P(B \mid A) = \frac{P(A \mid B)\, P(B)}{P(A)}
$$

{% hint style="success" %}
**Activity (2 minutes).** Try to derive Bayes' rule yourself before reading on. *Hint:* write $$P(A \cap B)$$ in two different ways.
{% endhint %}

<details>

<summary>Show the derivation</summary>

The probability that both $$A$$ and $$B$$ happen can be written in two ways:

$$
P(A \cap B) = P(A \mid B)\, P(B)
\qquad\text{and}\qquad
P(A \cap B) = P(B \mid A)\, P(A).
$$

Setting the two right-hand sides equal gives

$$
P(A \mid B)\, P(B) = P(B \mid A)\, P(A),
$$

and dividing both sides by $$P(A)$$ gives Bayes' rule.

</details>

#### For random variables

The same rule holds for random variables, with probability *densities* $$p$$ in place of probabilities $$P$$. (Formally, a random variable is a *measurable function*, but we will not need the details.)

A **model** is a collection of probability distributions, each labelled by numbers called **parameters**. For example, $$q(x \mid \theta) = \mathcal{N}(\theta, 1)$$ is a model: one normal distribution with variance 1 for each value of the mean $$\theta$$. The problem of fitting a model to data is called **statistical inference**.

### 2. Machine learning is statistical inference

Training a neural network is statistical inference: we choose parameters so that the model fits the data. Seeing it this way helps with three things:

1. **Generalisation:** understanding why a model fitted to training data does well on test data.
2. **Information gain:** measuring how much a model learns from the data.
3. **Designing algorithms:** for example, Khan and Rue's *Bayesian learning rule* shows that many common optimisers can be derived as approximations to Bayesian inference.

There are two main approaches to inference.

* **Maximum likelihood** picks the single parameter $$\hat{\theta}$$ that makes the data most probable.
* **Bayesian inference** treats the **parameters as random variables**. It starts with a *prior* $$p(\theta)$$ and updates it on the data $$\mathcal{D}$$ to get a *posterior*:

$$
p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta)\, p(\theta)}{p(\mathcal{D})}
$$

Note that $$p(\theta)$$ is a *density*, not a probability. It must be non-negative and integrate to one, $$\int p(\theta)\, d\theta = 1$$, but it can be greater than 1 at a point.

### 3. The marginal likelihood and the free energy

The denominator in Bayes' rule is the **marginal likelihood**, also called the *evidence*:

$$
p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta)\, p(\theta)\, d\theta
$$

It averages the fit of every parameter in the model, weighted by the prior. So it measures **how well the model class as a whole fits the data**, not just its best parameter. This makes it the natural tool for comparing models.

The same quantity appears in statistical physics. There it is called the **partition function**, written $$Z_n = p(\mathcal{D}_n)$$ for a dataset of $$n$$ points. Its negative logarithm is the **free energy**:

$$
F_n = -\log Z_n
$$

This is the same idea as the free energy in physics (for example, the Gibbs free energy). A lower free energy means the model class explains the data better.

**Bayesian Occam's razor.** The marginal likelihood has a built-in preference for simpler models, with no extra penalty needed. A probability distribution must sum to one over all possible datasets. A flexible model can explain many datasets, so it must spread its probability thinly across them. A simple model explains fewer datasets, so it puts more probability on each. If the data are consistent with the simple model, the simple model wins. Murray and Ghahramani (2005) compute this effect explicitly for a toy problem, making concrete a famous schematic figure from MacKay's thesis.

But what counts as "simple"? Rasmussen and Ghahramani (2001) point out a puzzle: Bayesian inference sometimes prefers small models, yet very large models can also perform well. They resolve it by measuring the complexity of the *functions* a model implements, not of the *parameters* used to implement them. Measured that way, Occam's razor is always at work. Keep this distinction in mind: it is exactly the move singular learning theory makes in Section 7.

### 4. KL divergence and information geometry

Bayes' rule can also be written as an *optimisation problem*: the posterior is the distribution that best trades off fitting the data against staying close to the prior. "Close" here is measured by the **Kullback–Leibler (KL) divergence**, which measures how far apart two distributions are:

$$
\mathrm{KL}(p \,\|\, q) = \int p(\theta) \log \frac{p(\theta)}{q(\theta)}\, d\theta
$$

By *Jensen's inequality*, $$\mathrm{KL}(p \,\|\, q) \geq 0$$, with equality only when $$p = q$$. Note that it is not symmetric: $$\mathrm{KL}(p \,\|\, q)$$ is usually not equal to $$\mathrm{KL}(q \,\|\, p)$$.

Now compare a model at parameter $$\theta$$ with the model at a nearby parameter $$\theta + \delta$$. For small $$\delta$$, the KL divergence between them is approximately quadratic:

$$
\mathrm{KL}\big(p_\theta \,\|\, p_{\theta + \delta}\big) \approx \tfrac{1}{2}\, \delta^\top F(\theta)\, \delta
$$

Here $$F(\theta)$$ is the **Fisher information matrix**. This fact lets us treat the space of parameters as a curved space, with $$F(\theta)$$ measuring distances. The study of this geometry is called **information geometry**, developed by Shun-ichi Amari. Singular learning theory grew out of it.

{% hint style="info" %}
**Terminology.** The *marginal likelihood* (or *evidence*) $$p(\mathcal{D})$$ measures how well a model class fits the data. The *free energy* is its negative log. The *KL divergence* measures how far one distribution is from another. The *Fisher information matrix* $$F(\theta)$$ measures how sensitive the model's predictions are to small changes in each parameter.
{% endhint %}

### 5. Singular models

Information geometry, and much of classical statistics, assumes that $$F(\theta)$$ can be inverted. A model is **regular** if this holds at the true parameter, and **singular** if it does not.

$$F(\theta)$$ fails to be invertible when it has an eigenvalue of zero: $$F(\theta) v = 0 \cdot v$$ for some direction $$v$$. Moving the parameters along $$v$$ does not change the model's predictions at all, to first order.

**Example.** Take the two-weight network from Week 7, $$y = abx$$. For any $$c \neq 0$$, the parameters $$(a/c,\; c\,b)$$ give exactly the same function:

$$
y = abx = \Big(\frac{a}{c}\Big)(c\, b)\, x
$$

So there is a whole curve of parameters that compute the same thing, and moving along it changes nothing. That direction has zero Fisher information, so $$F(\theta)$$ cannot be inverted. Neural networks are full of symmetries like this one, so **neural networks are singular models**.

**Algebraic geometry.** The set of parameters that fit the data perfectly is the set where $$K(\theta) = 0$$, with $$K$$ as defined in Section 7. For models like $$y = abx$$, this set is defined by *polynomial* equations, such as $$ab = 1$$. Sets defined by polynomial equations are called **algebraic varieties**, and they are studied in *algebraic geometry*. If the true function is $$y = 0$$, the solution set is $$ab = 0$$: the two axes, which cross at the origin. That crossing point is a **singularity**. SLT uses algebraic geometry to study how the shape of these sets near singularities affects learning.

### 6. Case study: comparing models with BIC

Suppose we want to know whether a second predictor $$x_2$$ helps explain $$y$$. We compare two linear regression models:

$$
M_1:\; y = \beta_0 + \beta_1 x_1 + \varepsilon
\qquad\qquad
M_2:\; y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \varepsilon
$$

$$M_2$$ always fits the training data at least as well, because it has an extra parameter. The question is whether it fits *enough* better to justify the extra complexity. The **Bayesian information criterion** answers this:

$$
\mathrm{BIC} = -2 \log p(\mathcal{D}_n \mid \hat{\theta}_n) + d \log n
$$

The first term measures fit, at the maximum likelihood estimate $$\hat{\theta}_n$$. The second is a penalty for complexity: $$d$$ is the number of parameters and $$n$$ the number of data points. We prefer the model with the lower BIC.

BIC is an approximation to $$-2 \log Z_n$$, twice the free energy. So choosing the model with the lowest BIC approximately chooses the model with the highest marginal likelihood. The approximation comes from a Taylor expansion of the loss, explained in the next section. It works for regular models like linear regression.

{% hint style="warning" %}
**Marginal likelihood is not the same as generalisation.** It is tempting to read the marginal likelihood as "how well will this model predict new data". Lotfi et al. (2022) show that the two can come apart. The marginal likelihood asks how likely the data are under the *prior*, so it also penalises a model for its poor early guesses, before it has seen much data. As a result, it can be *negatively* correlated with generalisation, and can lead to both underfitting and overfitting when used to tune hyperparameters. They propose a partial fix, the *conditional* marginal likelihood, which scores a model only on later data after it has conditioned on earlier data.
{% endhint %}

### 7. Beyond BIC: the learning coefficient

**Where BIC comes from.** Let $$L(\theta) = \mathbb{E}\big[-\log p(X \mid \theta)\big]$$ be the expected loss, minimised at a true parameter $$\theta^*$$. Define the excess loss $$K(\theta) = L(\theta) - L(\theta^*) \geq 0$$. Expand $$L$$ around $$\theta^*$$ as a Taylor series:

$$
L(\theta) = L(\theta^*) + \nabla L(\theta^*)^\top (\theta - \theta^*) + \tfrac{1}{2} (\theta - \theta^*)^\top H\, (\theta - \theta^*) + \cdots
$$

At a minimum the gradient is zero, so near $$\theta^*$$:

$$
K(\theta) \approx \tfrac{1}{2} (\theta - \theta^*)^\top H\, (\theta - \theta^*)
$$

where $$H$$ is the Hessian (the matrix of second derivatives) of the loss.

**Counting good parameters.** Let $$V(\varepsilon)$$ be the volume of parameters, weighted by the prior, whose excess loss is at most $$\varepsilon$$:

$$
V(\varepsilon) = \int \mathbb{1}\big[K(\theta) \leq \varepsilon\big]\, \pi(\theta)\, d\theta
$$

If $$H$$ is invertible, the near-optimal set is an ellipsoid. Its volume grows like

$$
V(\varepsilon) \approx \frac{v_d\, 2^{d/2}}{\sqrt{\det H}}\, \varepsilon^{d/2}
$$

where $$v_d$$ is the volume of the unit ball in $$d$$ dimensions. (This assumes a roughly flat prior near $$\theta^*$$.)

**From volume to free energy.** The marginal likelihood (rescaled by the constant $$e^{-nL(\theta^*)}$$) can be written in terms of this volume:

$$
Z_n = \int e^{-n K(\theta)}\, \pi(\theta)\, d\theta = \int_0^\infty e^{-s}\, V\!\left(\frac{s}{n}\right) ds
$$

So what matters is how fast $$V(\varepsilon)$$ shrinks as $$\varepsilon \to 0$$. If $$V(\varepsilon) \propto \varepsilon^{d/2}$$, then $$Z_n \propto n^{-d/2}$$, and the free energy picks up the term $$\tfrac{d}{2} \log n$$. Doubled, that is the $$d \log n$$ penalty in BIC.

**When the Hessian is singular.** In a singular model, $$H$$ has zero eigenvalues, the "ellipsoid" is not an ellipsoid, and the formula above divides by $$\det H = 0$$. But the volume still scales as a power of $$\varepsilon$$, with a possible log correction:

$$
V(\varepsilon) \approx c\, \varepsilon^{\lambda} (-\log \varepsilon)^{m - 1}
$$

The exponent $$\lambda$$ is the **real log canonical threshold** (RLCT), also called the **learning coefficient**. The integer $$m \geq 1$$ is its *multiplicity*. In a regular model, $$\lambda = d/2$$ and $$m = 1$$. In a singular model, $$\lambda \leq d/2$$, often much smaller: the model behaves as if it had fewer effective parameters.

Algebraic geometry gives a way to find $$\lambda$$. Define the **zeta function**

$$
\zeta(s) = \int K(\theta)^{-s}\, \pi(\theta)\, d\theta
$$

This integral is finite for small $$s$$, and blows up once $$s$$ reaches $$\lambda$$. So $$\lambda$$ is the location of the first pole of $$\zeta(s)$$, and $$m$$ is the order of that pole.

**The singular BIC.** Replacing $$d/2$$ by $$\lambda$$ gives the version of BIC that holds for singular models:

$$
\mathrm{BIC}_{\mathrm{SLT}} = -2 \log p(\mathcal{D}_n \mid \hat{\theta}_n) + 2\lambda \log n - 2(m - 1) \log \log n
$$

For a regular model, $$2\lambda = d$$ and $$m = 1$$, so this is ordinary BIC. The complexity of a singular model is set not by how many parameters it has, but by the *geometry* of its near-optimal parameters.

**The local learning coefficient.** In practice, we often want the complexity of a *particular* trained network, not of the whole model class. The **local learning coefficient** (LLC) applies the same idea to a small neighbourhood of one parameter, such as the end point of training. It can be estimated numerically, and it gives a way to measure how complex a network's solution is. Lau et al. (2023) define the LLC and give a scalable estimator. They apply it to deep linear networks with up to 100 million parameters, to ResNet image models and to transformer language models, and use it to study how training choices affect a network's effective complexity.

{% hint style="info" %}
**Terminology.** A model is *regular* if its Fisher information matrix is invertible at the true parameter, and *singular* otherwise. The *learning coefficient* or *RLCT* $$\lambda$$ measures the effective complexity of a model: it controls how fast the volume of good parameters shrinks. The *local learning coefficient* (LLC) is the same measure around one particular parameter.
{% endhint %}

### 8. Practical caveats and uses

**Caveats.**

1. **ReLU networks.** The theory assumes the model is *analytic*: smooth enough to be written as a convergent power series. ReLU is not smooth at zero, so the classical results do not directly apply to ReLU networks.
2. **Normalisation is tricky** in practice.

**Uses.** The learning coefficient can be tracked over the course of training. Changes in the LLC can signal when a network forms new internal structure, such as circuits. This gives a way to **predict and detect how circuits change over time**. See *Stagewise development in neural networks* in the further reading.

### Recap

1. Machine learning is statistical inference.
2. Bayes' rule is one way of doing inference.
3. The marginal likelihood tells you how well your model *class* fits the data.
4. When the loss can be approximated by a quadratic around its minimum (a regular model), a Taylor expansion gives the marginal likelihood approximately, and hence BIC.
5. When it cannot (a singular model, such as a neural network), SLT replaces $$d/2$$ with the learning coefficient $$\lambda$$, giving $$\mathrm{BIC}_{\mathrm{SLT}}$$.

---

## Discussion questions

{% hint style="success" %}
**Discussion**

1. Maximum likelihood gives one parameter, and Bayesian inference gives a distribution over parameters. When would you want the whole distribution?
2. Why is the marginal likelihood a better tool for comparing models than the training loss alone? Lotfi et al. (2022) show it can still be a poor guide to generalisation. When would you trust it, and when not?
3. In $$y = abx$$, which direction in parameter space has zero Fisher information? Can you find a symmetry like this in a two-layer neural network?
4. In the BIC case study, $$M_2$$ has one more parameter than $$M_1$$. How much better must its fit be for BIC to prefer it, with $$n = 100$$ data points? With $$n = 10{,}000$$?
5. Rasmussen and Ghahramani (2001) measure complexity by the functions a model implements, not by its parameters. How does the learning coefficient do something similar? SLT says a singular model can behave as if it had fewer parameters than it does. How might this help explain why large neural networks generalise well (see [Week 5](../05-mysteries-of-deep-learning/README.md))?
6. Why might it matter for AI safety to detect when a network forms a new circuit during training?
{% endhint %}

---

## Further reading

* Watanabe, S. (2009). *Algebraic geometry and statistical learning theory*. Cambridge University Press. The founding text of SLT.
* Watanabe, S. (2018). *Mathematical theory of Bayesian statistics*. CRC Press. A more accessible treatment.
* Amari, S. (2016). *Information geometry and its applications*. Springer.
* Khan, M. E. and Rue, H. (2023). *The Bayesian learning rule*. Journal of Machine Learning Research, 24. [https://arxiv.org/abs/2107.04562](https://arxiv.org/abs/2107.04562).
* *Stagewise development in neural networks*. LessWrong. [https://www.lesswrong.com/posts/Zza9MNA7YtHkzAtit/stagewise-development-in-neural-networks](https://www.lesswrong.com/posts/Zza9MNA7YtHkzAtit/stagewise-development-in-neural-networks).
* [Iliad Intensive curriculum](https://iliad-intensive.org): the Singular Learning Theory module (B.3).

---

## References

### Optional readings

* Rasmussen, C. E. and Ghahramani, Z. (2001). *Occam's razor*. In *Advances in Neural Information Processing Systems 13*. [https://mlg.eng.cam.ac.uk/zoubin/papers/occam.pdf](https://mlg.eng.cam.ac.uk/zoubin/papers/occam.pdf).
* Murray, I. and Ghahramani, Z. (2005). *A note on the evidence and Bayesian Occam's razor*. Gatsby Unit Technical Report GCNU-TR 2005-003, University College London. [https://homepages.inf.ed.ac.uk/imurray2/pub/05occam/occam.pdf](https://homepages.inf.ed.ac.uk/imurray2/pub/05occam/occam.pdf).
* Lotfi, S., Izmailov, P., Benton, G., Goldblum, M. and Wilson, A. G. (2022). *Bayesian model selection, the marginal likelihood, and generalization*. In *Proceedings of the 39th International Conference on Machine Learning (ICML)*, PMLR 162. [https://proceedings.mlr.press/v162/lotfi22a/lotfi22a.pdf](https://proceedings.mlr.press/v162/lotfi22a/lotfi22a.pdf).
* Lau, E., Furman, Z., Wang, G., Murfet, D. and Wei, S. (2023). *The local learning coefficient: A singularity-aware complexity measure*. arXiv:2308.12108. [https://arxiv.org/abs/2308.12108](https://arxiv.org/abs/2308.12108).

### Other works cited

* Schwarz, G. (1978). *Estimating the dimension of a model*. The Annals of Statistics, 6(2), 461–464. The original paper on BIC.
