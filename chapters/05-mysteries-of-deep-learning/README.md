# Week 5: Mysteries of Deep Learning

**Mentor:** [Nicholas Chen](https://scholar.google.com/citations?user=mBM4R48AAAAJ) \
**Date:** 22 Oct 2026

---

## Why does deep learning work?

Last week we built a theory of learning around three barriers: approximation, generalisation and optimisation. We saw that an ideal learner, Solomonoff induction, gets past the first two but cannot be run. This week we turn to the learner we *can* run, deep learning, and ask how it gets past all three barriers at once. The short answer is that nobody fully knows. This week is about what those mysteries are, why classical theory fails to explain them, and some candidate explanations.

### What you'll learn

By the end of the week, you will be able to:

* Explain and tell apart the three classical mysteries of deep learning: approximation, generalisation and optimisation.
* Explain why each mystery can only be solved by using structure in the real world, since learning is not tractable for arbitrary tasks.
* Describe the main empirical mysteries: generalisation that depends on the data despite huge numbers of parameters, the success of gradient descent on non-convex problems, representational alignment across architectures, and in-context learning.
* State at least one candidate explanation for a mystery, and say what it does and does not explain.
* Explain the "deep learning as program synthesis" hypothesis, which links deep learning to Solomonoff induction, and evaluate it.
* Explain why these mysteries matter for AI safety.

### Roadmap

The week has two parts.

| Part            | Format                                                                                     |
| --------------- | ------------------------------------------------------------------------------------------ |
| Core            | Everyone reads the overview slides and *Deep Learning as Program Synthesis*, then discusses |
| Explorations    | Groups each pick one exploratory topic, read and discuss, then report back to the whole group |

{% hint style="info" %}
**Reading levels.** As in Week 4, **100** is introductory, **200** needs some maths or ML background, and **300** is advanced. Ranges such as 100–200 mean the reading works at either level.
{% endhint %}

---

## Preamble: the mysteries

Classical learning theory makes three predictions about deep learning, and all three seem to be wrong.

* **Approximation.** Approximating an arbitrary function in many dimensions needs exponentially many parameters (the *curse of dimensionality*). Yet networks of manageable size approximate the functions we care about.
* **Generalisation.** Modern networks have far more parameters than training examples. The bias–variance tradeoff from Week 4 says they should overfit badly. They do not.
* **Optimisation.** Training a neural network means minimising a non-convex loss. Local search on such a landscape could get stuck, and in the worst case, learning is provably hard. Yet simple gradient descent works reliably.

No Free Lunch (Week 4) tells us that no learner can succeed on *every* problem. So if deep learning succeeds, it must be because the problems we give it have structure that its inductive biases match. The question is: what structure, and what biases?

---

## Reading guide

This section summarises the main readings. It is a companion to the readings, not a replacement for them. Full citations are in the [References](#references) at the end of the chapter.

### Core: overview slides

*Source: Iliad B2 slides.*

The overview slides set out the three mysteries above and the main empirical puzzles. Read them first.

{% hint style="success" %}
**Discussion**

1. *(100–200, revisited from Week 4)* From an AI safety perspective, why is it worth trying to figure out scientifically how deep learning works? Why would we need scientific understanding for safety, if such understanding seems to have been unnecessary for capabilities?
{% endhint %}

### Core: Deep learning as program synthesis

*Source: Furman (2025). Level 200.*

The post makes a bold claim: deep learning works because it is doing something like **Solomonoff induction**. On this view, a trained network is not just a curve fitted to data. It implements a simple *program*, built out of reusable parts, and training is a search for the simplest program that explains the data. This would explain generalisation the way Solomonoff induction does: simple programs that fit the data tend to predict well.

The post separates this hypothesis from a trivial claim. Any model running on a computer is in some sense a program, including linear regression. The real claim is about *what kind* of solutions training finds. They are compositional and algorithmic, and the learner prefers simpler ones.

The post also insists on a distinction between **functions** (the input–output behaviour) and **programs** (the internal mechanism that produces it). Two networks can compute the same function by very different means, and the means matter. Mechanism determines how a network behaves on inputs it has not seen, how it changes under further training, and what interpretability can find inside it.

{% hint style="warning" %}
**A note from an Iliad contributor.** This post presents an *opinionated hypothesis* (that deep learning does something analogous to Solomonoff induction) alongside a *relatively consensus* discussion of the empirical mysteries. It is shared mainly for the second part, though you may find the hypothesis useful as a teaching tool. Read it critically.
{% endhint %}

{% hint style="success" %}
**Discussion**

1. *(100–200, revisited from Week 4)* Intuitively, what is Solomonoff induction, and why do we care about it?
2. *(100–200)* One could trivially say that a neural network "learns programs", because a neural network runs on a computer. In that sense, linear regression "learns programs" too. What distinguishes the author's hypothesis from this trivial fact? (The answer is in the article. Use this to check your understanding.)
3. *(100–200)* Where in the post do the three mysteries (approximation, generalisation and optimisation) appear? Why does the post put the section on optimisation in a separate place from the other two?
4. *(200)* The post insists on keeping "functions" and "programs" distinct. Why? Why would we care to distinguish two networks that implement the same function by different means?
{% endhint %}

---

## Explorations

Now break into groups. Each group picks **one** exploratory topic below, reads, and discusses. At the end of the session, everyone gathers back and each group shares what it learned.

{% hint style="warning" %}
**How to read these papers.** Several of these papers are dense and need a lot of background to understand fully. Read with the discussion questions in mind, and skim the details you do not need. This is a good chance to practise selective reading.
{% endhint %}

For **every** topic, also answer:

> Why is this a "mystery", compared to what we know in theory? Does theory seem to contradict it? If so, how: is there a mistake in the theory, or does the theory make assumptions that do not hold in our setting? Or does theory simply fail to explain it?

### Exploration A: Approximation

*Sources: Hoogland and Furman (2023), level 200; Poggio et al. (2017), level 300, semi-optional.*

**Approximation is expensive, but the lunch is cheap.** The **Universal Approximation Theorem** says a network with one hidden layer can approximate any continuous function as closely as you like. But it says nothing about *how many* neurons that takes, and for arbitrary functions the answer is exponential in the input dimension. So the theorem does not explain what makes neural networks special: many other model classes are universal approximators too.

The post argues that efficient approximation needs two things: target functions that are, in some sense, simple, and models whose inductive biases match that simplicity. Real-world targets tend to be *smooth* (low-frequency) and *hierarchical* (built from simpler parts). The "cheap lunch" is that the world has this structure, and deep networks exploit it. The post also argues that, in practice, approximation, generalisation and optimisation are hard to separate cleanly.

**Deep vs. shallow networks.** Poggio et al. review **depth separation** results. For *compositional* functions (functions built by composing many functions that each depend on only a few variables), deep networks need far fewer parameters than shallow ones. When the network's structure matches the function's composition, the curse of dimensionality goes away. Aim to get the general shape of the results: "under assumption X, the number of neurons needed to reach a given error scales like Y".

{% hint style="success" %}
**Discussion**

1. *(200)* The Universal Approximation Theorem says that a one-hidden-layer network can approximate any continuous function to arbitrary accuracy. Why is this not an explanation for the success of deep learning?
2. *(200–300)* If approximating arbitrary smooth functions provably needs exponentially many parameters (the curse of dimensionality), what must be true about the functions deep learning actually faces, for it to work at all? What is a "depth separation" result, and what does it suggest about the answer? (You may need a short web search to clarify some details.)
{% endhint %}

### Exploration B: Generalisation

*Sources: LawrenceC (2026), level 100–200; Wilson (2025), level 100–200, you can gloss over the maths.*

**The paper that killed deep learning theory.** The post tells the story of Zhang et al. (2017), *Understanding deep learning requires rethinking generalization*. Classical theory explains generalisation through **capacity**: if the hypothesis class is too rich (measured, for example, by VC dimension or Rademacher complexity), the model can fit anything, so fitting the training data proves nothing. Zhang et al. showed that standard image networks can perfectly fit CIFAR-10 and ImageNet with *random labels*, and even with random pixels. So the capacity of these networks is enormous, and capacity-based bounds on their error become vacuous. Yet the *same* networks, trained on *real* labels, generalise well. Explicit regularisation, such as weight decay or data augmentation, does not explain the gap. Whatever explains generalisation must depend on the *data* and the *training procedure*, not just on the hypothesis class.

**Deep Learning is Not So Mysterious or Different.** Wilson takes the opposite tone. He argues that phenomena such as benign overfitting, double descent and fitting random labels are not unique to neural networks, and that existing frameworks already describe them. Examples are PAC-Bayes and countable-hypothesis bounds, and the idea of **soft inductive biases**: a flexible hypothesis class combined with a *preference* for simpler solutions within it. On this view, a model can be able to fit noise and still prefer structure when structure is there.

{% hint style="info" %}
**Terminology.** *Capacity* measures how many different patterns a hypothesis class can fit. *VC dimension* and *Rademacher complexity* are two standard capacity measures. A bound is *vacuous* when it is so loose that it tells you nothing (for example, "test error is at most 100%").
{% endhint %}

{% hint style="success" %}
**Discussion**

1. *(100)* There are different notions of "generalisation", and they are not equivalent. What exactly do these resources mean by the word? How does it differ from *out-of-distribution* (OOD) generalisation?
2. *(200)* Why are the experimental results of Zhang et al. fatal to capacity-based generalisation bounds (VC dimension, Rademacher complexity and so on)? What does this imply about what any explanation of generalisation must depend on?
3. *(100–200)* The first post is pessimistic in tone: it declares that deep learning theory (or at least the theory of generalisation) has been "killed". *Deep Learning is Not So Mysterious or Different* takes almost the opposite attitude: the same empirical results are not very surprising under existing frameworks. Despite the difference in tone, how compatible are these views on the object level? What common picture do they paint?
{% endhint %}

**Optional readings:** Zhang et al. (2017), the original paper (level 200–300); Goldblum et al. (2023), *The No Free Lunch Theorem, Kolmogorov Complexity, and the Role of Inductive Biases in Machine Learning* (level 200–300), which ties this topic back to Week 4; and *The other paper that killed deep learning theory* (level 200), on a 2019 result by Nagarajan and Kolter.

### Exploration C: Optimisation

{% hint style="danger" %}
**Warning.** This topic needs a lot of background in learning theory and complexity theory. We recommend skipping it unless you already have that background.
{% endhint %}

*Sources: Shalev-Shwartz and Ben-David (2014), section 8.4 "Hardness of learning", level 300. Read sections 8.1–8.3 first for background. Optional: Barak et al. (2022); Abbe et al. (2023), both level 300.*

**Hardness of learning.** Some function classes are provably hard to learn efficiently, under standard cryptographic assumptions. The idea is that a *one-way function* (easy to compute, hard to invert) makes a target that no efficient algorithm can learn from examples. Neural networks can implement such functions. So there are targets that a network can *represent* (approximation is fine), and that could be identified from a modest amount of data (generalisation is fine, at least for an idealised Bayesian learner), but that no efficient algorithm, gradient descent included, can *find*. This is the cleanest case of optimisation coming apart from the other two barriers.

**Hidden progress and leap complexity.** Barak et al. study how gradient descent learns *sparse parities*, a standard hard problem. They find that it learns them close to the best runtime any algorithm of its kind could achieve, and that it makes steady "hidden" progress even while the loss looks flat. Abbe et al. describe how the difficulty of learning depends on the "leaps" a network must make between levels of a function's structure, and how training moves from one saddle point to the next.

{% hint style="success" %}
**Discussion**

1. *(200–300)* These cryptographic hardness arguments apply to neural networks, since neural networks can implement one-way functions. What does this imply about how long gradient descent will take to learn such functions? By contrast, how will Bayesian learning behave in this scenario, and why are hardness arguments vacuous for Bayes? What does this imply about how approximation and generalisation come apart from optimisation?
2. *(300)* Neural networks *can* represent worst-case targets, yet they train well in practice. Why care about these pathological examples, then? How is the takeaway different from the obvious point that "an algorithm's typical-case performance can be much better than its worst case"?
{% endhint %}

The Week 7–8 instructor recommends Bach's *Learning Theory from First Principles* as an alternative textbook.

### Exploration D: Representational alignment

*Source: Huh et al. (2024), level 100–200. You can gloss over the appendix. For the maths in the main paper, a basic understanding is enough.*

**The Platonic Representation Hypothesis.** It is well established that different neural networks often learn similar internal representations. Huh et al. go further. They argue that as models become larger and more capable, their representations **converge**, across different architectures, training objectives and even different *data modalities* (for example, vision and language). Their hypothesis is that the convergence is towards a shared statistical model of the underlying reality that generates the data. They call this a "platonic" representation, after Plato's idea of ideal forms behind appearances. Their evidence includes alignment between models that grows with scale and capability, and alignment between vision models and language models.

{% hint style="info" %}
**Iliad contributor's note.** One could argue that this hypothesis is not new either. See, for example, the natural abstraction hypothesis, which significantly predates it. But this paper was the first major academic paper to promote it.
{% endhint %}

{% hint style="success" %}
**Discussion**

1. *(100–200)* What is the *new* hypothesis that the paper promotes, and what observations were already established in earlier work? What evidence do the authors cite for their hypothesis? What distinguishes their hypothesis from the weaker claim that "models converge to shared representations"?
{% endhint %}

**Optional readings:** Zamir et al. (2018), *Taskonomy*, which hints at shared structure between computer-vision tasks; Gröger et al. (2026), which critiques the hypothesis, argues that similarity metrics are confounded by network scale, and proposes an "Aristotelian" alternative; and Sucholutsky et al. (2023), *Getting aligned on representational alignment* (level 100–200), a survey of the field.

### Exploration E: In-context learning

{% hint style="info" %}
**Note.** We prefer that groups do not pick this topic, because most AI safety and mechanistic interpretability courses already cover it.
{% endhint %}

*Source: Olsson et al. (2022), level 100–200. Optional: Stagewise Development in Neural Networks, level 300.*

**Induction heads.** Large language models can learn from examples in their prompt without any weight updates. This is called **in-context learning**. Olsson et al. study an **induction head**: a circuit of two attention heads that finds an earlier occurrence of the current token and predicts the token that came after it ("... A B ... A → B"). They find that induction heads form during a short, sharp phase of training. A visible bump in the loss curve appears at the same time, together with a sudden jump in in-context learning ability. They argue that induction heads may be the mechanism behind most in-context learning.

The optional reading on stagewise development needs singular learning theory (Week 8). You can come back to it after that week.

{% hint style="success" %}
**Discussion**

1. *(100–200)* What is an induction head, and what relationship does the paper draw between induction-head formation and in-context learning over the course of training? Why treat the fact that they happen at the same time as evidence of a mechanistic link, and not a coincidence?
{% endhint %}

---

## Further reading

The Iliad module lists more readings that are not covered here. A selection:

* Telgarsky, M. *Deep learning theory lecture notes*. [https://mjt.cs.illinois.edu/dlt/](https://mjt.cs.illinois.edu/dlt/).
* Gwern. *The scaling hypothesis*. [https://gwern.net/scaling-hypothesis](https://gwern.net/scaling-hypothesis).
* Saxe, A., McClelland, J. and Ganguli, S. (2013). *Exact solutions to the nonlinear dynamics of learning in deep linear neural networks*. [https://arxiv.org/abs/1312.6120](https://arxiv.org/abs/1312.6120).
* Betley, J. et al. (2025). *Emergent misalignment: Narrow finetuning can produce broadly misaligned LLMs*. [https://arxiv.org/abs/2502.17424](https://arxiv.org/abs/2502.17424). (See also Week 3.)

For the full list, see the [Iliad Mysteries of Deep Learning module](https://iliad-intensive.org/learning/mysteries-of-deep-learning/).

---

## Attribution

This chapter is adapted from the ILIAD Intensive module *Mysteries of Deep Learning* (B.2) by Zach Furman ([iliad-intensive.org/learning/mysteries-of-deep-learning](https://iliad-intensive.org/learning/mysteries-of-deep-learning/)), including its learning goals, reading list, discussion questions and contributor notes, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: the material was reorganised into a core session and group explorations, some discussion questions were reworded, reading summaries were written by the Odysea 2026 Technical Team, and level tags and reading guidance were added.

---

## References

### Core readings

* ILIAD. *B2: Mysteries of deep learning* (slides). [https://drive.google.com/drive/folders/1SV-VYOSTzEGcRxw5GuDBCLbwQWj3k0Ym](https://drive.google.com/drive/folders/1SV-VYOSTzEGcRxw5GuDBCLbwQWj3k0Ym).
* Furman, Z. (2025). *Deep learning as program synthesis*. LessWrong. [https://www.lesswrong.com/posts/Dw8mskAvBX37MxvXo/deep-learning-as-program-synthesis-1](https://www.lesswrong.com/posts/Dw8mskAvBX37MxvXo/deep-learning-as-program-synthesis-1).

### Exploration readings

**Approximation**

* Hoogland, J. and Furman, Z. (2023). *Approximation is expensive, but the lunch is cheap*. LessWrong, 19 April 2023. [https://www.lesswrong.com/posts/gq9GR6duzcuxyxZtD/approximation-is-expensive-but-the-lunch-is-cheap](https://www.lesswrong.com/posts/gq9GR6duzcuxyxZtD/approximation-is-expensive-but-the-lunch-is-cheap).
* Poggio, T., Mhaskar, H., Rosasco, L., Miranda, B. and Liao, Q. (2017). *Why and when can deep – but not shallow – networks avoid the curse of dimensionality: A review*. arXiv:1611.00740. [https://arxiv.org/abs/1611.00740](https://arxiv.org/abs/1611.00740).

**Generalisation**

* LawrenceC (2026). *The paper that killed deep learning theory*. LessWrong, 26 April 2026. [https://www.lesswrong.com/posts/ZvQfcLbcNHYqmvWyo/the-paper-that-killed-deep-learning-theory](https://www.lesswrong.com/posts/ZvQfcLbcNHYqmvWyo/the-paper-that-killed-deep-learning-theory).
* Wilson, A. G. (2025). *Deep learning is not so mysterious or different*. arXiv:2503.02113. [https://arxiv.org/abs/2503.02113](https://arxiv.org/abs/2503.02113).
* *Optional:* Zhang, C., Bengio, S., Hardt, M., Recht, B. and Vinyals, O. (2017). *Understanding deep learning requires rethinking generalization*. ICLR 2017. arXiv:1611.03530. [https://arxiv.org/abs/1611.03530](https://arxiv.org/abs/1611.03530).
* *Optional:* Goldblum, M., Finzi, M., Rowan, K. and Wilson, A. G. (2023). *The No Free Lunch theorem, Kolmogorov complexity, and the role of inductive biases in machine learning*. arXiv:2304.05366. [https://arxiv.org/abs/2304.05366](https://arxiv.org/abs/2304.05366).
* *Optional:* *The other paper that killed deep learning theory*. LessWrong. [https://www.lesswrong.com/posts/zcGmdQHX66NhC69v6/the-other-paper-that-killed-deep-learning-theory](https://www.lesswrong.com/posts/zcGmdQHX66NhC69v6/the-other-paper-that-killed-deep-learning-theory).

**Optimisation**

* Shalev-Shwartz, S. and Ben-David, S. (2014). *Understanding machine learning: From theory to algorithms*. Cambridge University Press. Section 8.4. [https://www.cambridge.org/core/books/understanding-machine-learning/3059695661405D25673058E43C8BE2A6](https://www.cambridge.org/core/books/understanding-machine-learning/3059695661405D25673058E43C8BE2A6). A copy for personal use is available from [the authors' page](http://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning).
* *Alternative:* Bach, F. (2024). *Learning theory from first principles*. MIT Press. [https://www.di.ens.fr/~fbach/ltfp/](https://www.di.ens.fr/~fbach/ltfp/).
* *Optional:* Barak, B., Edelman, B., Goel, S., Kakade, S., Malach, E. and Zhang, C. (2022). *Hidden progress in deep learning: SGD learns parities near the computational limit*. arXiv:2207.08799. [https://arxiv.org/abs/2207.08799](https://arxiv.org/abs/2207.08799).
* *Optional:* Abbe, E., Boix-Adserà, E. and Misiakiewicz, T. (2023). *SGD learning on neural networks: Leap complexity and saddle-to-saddle dynamics*. arXiv:2302.11055. [https://arxiv.org/abs/2302.11055](https://arxiv.org/abs/2302.11055).

**Representational alignment**

* Huh, M., Cheung, B., Wang, T. and Isola, P. (2024). *The platonic representation hypothesis*. ICML 2024. arXiv:2405.07987. [https://arxiv.org/abs/2405.07987](https://arxiv.org/abs/2405.07987).
* *Optional:* Zamir, A., Sax, A., Shen, W., Guibas, L., Malik, J. and Savarese, S. (2018). *Taskonomy: Disentangling task transfer learning*. arXiv:1804.08328. [https://arxiv.org/abs/1804.08328](https://arxiv.org/abs/1804.08328).
* *Optional:* Gröger, F., Wen, S. and Brbić, M. (2026). *Revisiting the platonic representation hypothesis: An Aristotelian view*. arXiv:2602.14486. [https://arxiv.org/abs/2602.14486](https://arxiv.org/abs/2602.14486).
* *Optional:* Sucholutsky, I. et al. (2023). *Getting aligned on representational alignment*. arXiv:2310.13018. [https://arxiv.org/abs/2310.13018](https://arxiv.org/abs/2310.13018).

**In-context learning**

* Olsson, C. et al. (2022). *In-context learning and induction heads*. Transformer Circuits Thread. [https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html).
* *Optional:* *Stagewise development in neural networks*. LessWrong. [https://www.lesswrong.com/posts/Zza9MNA7YtHkzAtit/stagewise-development-in-neural-networks](https://www.lesswrong.com/posts/Zza9MNA7YtHkzAtit/stagewise-development-in-neural-networks).
