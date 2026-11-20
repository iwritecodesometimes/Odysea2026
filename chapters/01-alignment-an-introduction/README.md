# Week 1: Alignment: An Introduction

**Author:** Odysea 2026 Technical Team\
**Mentor:** Shashvat Shukla\
**Date:** 24 Sep 2026

---

## Introduction to the course

This week introduces the course and the field it draws on. We start with the *worldview* behind much of AI safety: why some people think AI could become extremely powerful, and soon. We then survey the *technical* background the field relies on, from linear algebra to cryptography, and why each part matters for safety. Finally, we take a tour of the Iliad Intensive curriculum, which this course is based on.

[Week 2](../02-ai-alignment/README.md) continues from here, with alignment targets and how alignment fails.

> "To know what you know and what you do not know, that is true knowledge."
> — Confucius

### What you'll learn

By the end of the week, you will be able to:

* State the three claims of the Iliad worldview, and the assumptions each one rests on.
* Define AI misalignment, and explain why it is harder than ordinary software bugs.
* Name the main AI risks other than misalignment.
* Explain, for each technical prerequisite, at least one way it is used in AI safety.
* Describe the main areas of the Iliad Intensive curriculum.

### Roadmap

The week runs as two 1-hour sessions. Each session follows the same pattern.

| Time    | Session                | Breakdown                                       |
| ------- | ---------------------- | ----------------------------------------------- |
| 20 min  | Lecture                | Introduction to the week's topics               |
| 20 min  | Small group discussion | Work through the discussion questions           |
| 20 min  | Combined discussion    | Whole-group discussion led by the mentor        |

---

## Summary

### Who brings you this course

* **rain.sg (Responsible AI Network Singapore).** AI safety is wherever you are. It needs deliberate practice and study, and it is deliberative.
* **SASH (Singapore AI Safety Hub).** Focuses on partnerships with government, academia and industry, on bringing together East and West, and on technologies that enable governance.
* **BlueDot Impact.** A global talent accelerator for beneficial AI and social resilience. Their courses and reading lists are highly recommended.
* **Iliad.** Develops AI safety as a science, with an emphasis on foundations and rigour. The *Iliad Intensive* is a one-month, full-time course on AI safety: a course for the "neats" about modern AI, which is often more "scruffy".

The Iliad Intensive has six parts: worldview and technical prerequisites; the alignment problem in theory and practice; how learning and deep learning work; abstractions, representations and interpretations; the theory of agents; and safety guarantees and their limits.

### Part 1: Worldview prerequisites

**Why does AI matter?** The impact of AI might be enormous, and possibly soon. This belief is hard to hold *stably*, whether or not it is true. People tend to hold it only intellectually, hold it emphatically, or reject it. The Iliad worldview makes three claims.

**(1) Intelligence gives rise to power.** This rests on three assumptions:

* Intelligence is a meaningful *latent variable*: a single underlying quantity, not just a list of separate skills.
* Intelligence is correlated with power, for example for instrumental reasons.
* There is a lot of room for intelligence beyond human level.

Bostrom's *Superintelligence*, chapter 6, lists the "cognitive superpowers" an AI might have: intelligence amplification, strategising, social manipulation, hacking, technology research and economic productivity. The optimistic side of the same claim appears in Dario Amodei's *Machines of Loving Grace* and in *Genesis: Artificial Intelligence, Hope, and the Human Spirit*. Their general message is that AI makes anything possible and could solve the world's problems.

**(2) Human-level AI is here now, or will be soon.** The evidence is scaling laws (for training, inference and multi-agent systems), evaluations that are all saturating, and the growing length of tasks that AI can complete. Note the assumption: "human level" is defined by completing *standardised tasks*, often AI research tasks.

**(3) An intelligence explosion may follow.** Possible assumptions:

* Gains from intelligence amplification will not plateau soon.
* Takeoff can be fast or slower.
* Takeoff is inevitable, even if we see it coming, because of race dynamics or the complexity of AI.

> "What was presented as an objective, incontrovertible scientific discovery was in fact the product of eminently questionable social and ethical commitments."
> — Foucault, on *Madness and Civilization*

This quote is a caution: worldviews that present themselves as neutral science can carry social and ethical commitments. Keep it in mind as you assess the claims above.

**AI misalignment** is when an AI system does not do what its user wants it to do. It is like a software bug, but in AI. The key challenges are:

* We do not understand AI the way we understand ordinary computer systems.
* The systems we want to control may be more intelligent or capable than us, and may deceive us.
* It is hard to specify what we want AI to do.
* It is hard to predict how the AI will meet the goal.

Because the impact of AI is large, misalignment is extremely costly. Note the assumptions here too: that tools should do exactly what we want, and that our wants are important and coherent.

**Agent foundations.** One line of thinking treats AI as *goal-directed* or *reward-maximising*, as in the paperclip maximiser thought experiment. Omohundro's "basic AI drives" argue that such agents will tend to self-improve, become rational, preserve their utility function, avoid counterfeit utility, protect themselves, and acquire and use resources. If intelligence is a latent variable, then even an AI built to be good and useful could end up seeking power. Is this a conceptual claim or an empirical one?

**An alternative: thick models of value.** Current AI systems model human values as utility functions, preference orderings, or unstructured text. Alternatives include constitutive goods; capabilities and functionings; how values connect to human systems; uncontroversial fitness functions, such as values as norms that foster coexistence; and embedding values in a theory of moral progress, such as reflective equilibrium.

**Other AI risks.** Misalignment is not the only risk.

* *Catastrophic misuse:* AI used for bioterrorism, released as uncontrolled agents, or used for persuasion.
* *Rise of totalitarianism:* alignment is solved, but the AI is aligned to an autocrat.
* *Gradual disempowerment*, despite step-by-step progress.
* *Other harms:* algorithmic bias, cognitive offloading, digital exclusion, entrenched inequality, environmental damage, and disruption of social institutions such as politics, education and the media.

{% hint style="success" %}
**Discussion**

1. Consider the Hugging Face incident (see [Week 3](../03-alignment-in-practice/README.md#case-studies)). Which of Bostrom's cognitive superpowers, if any, did the agents show?
2. What do you want powerful AI to do?
3. "Even if you try to build good, useful AI, it could end up seeking power." Is this a conceptual claim or an empirical claim? What evidence would change your mind?
4. Which of the three worldview claims do you find most and least convincing? Which assumption is doing the most work?
{% endhint %}

### Part 2: Technical prerequisites

The Iliad Intensive expects background in programming (especially Python, and especially with coding agents) and libraries such as PyTorch, NumPy and einops; deep learning, loss landscapes, training dynamics, reinforcement learning and common architectures; linear algebra, calculus, probability, statistics and information theory; computability, complexity, formal logic and cryptography; and some intuition from statistical mechanics. Below is a flavour of each, and why it matters for safety.

#### A short history of AI

* **MENACE** (Matchbox Educable Noughts and Crosses Engine) learned to play noughts and crosses with matchboxes and beads. It was an early learning machine.
* **"Traditional" machine learning**, followed by the **No Free Lunch** theorem: no learner wins on every problem, so each needs the right *inductive bias*. For example, a physics-informed neural network learning a harmonic oscillator much better than a plain network, because physics is built into it. We return to this in [Week 4](../04-principles-of-learning/README.md).
* **The deep learning revolution:** perceptrons, multilayer perceptrons, autoencoders, convolutional networks, GANs, recurrent networks, LSTMs and transformers.
* **The bitter lesson** (Sutton, 2019): "general methods that leverage computation are ultimately the most effective, and by a large margin." Looking ahead, inference is still very sequential, and quantum computers are not yet used.

#### Mathematics and computer science, and why safety needs them

| Topic                    | The core idea                                                                                                     | Use in AI safety                                                                                                                                                       |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Linear algebra           | A vector is a list of numbers. Linear algebra is the maths of such lists.                                         | The language of neural networks and their internal representations.                                                                                                    |
| Calculus                 | The maths of change.                                                                                              | Gradient-based training.                                                                                                                                               |
| Bayesianism              | Update beliefs on evidence.                                                                                       | Stating risk estimates as probabilities, training AI to report calibrated beliefs, and perhaps corrigibility as uncertainty over human values.                          |
| Information theory       | How much data can be compressed, and how fast information can be sent over a noisy channel.                       | How compressible pre-training data is, how much information RLHF gives a model, KL penalties, and limits on how fast a model could smuggle out its weights.            |
| Computability theory     | What different models of computation can solve at all.                                                            | What bare LLMs and agentic systems can and cannot compute.                                                                                                             |
| Complexity theory        | How fast different models of computation can solve problems.                                                      | Hard limits on what any AI can do efficiently, however capable.                                                                                                        |
| Formal logic             | Rules for valid reasoning.                                                                                        | Machine-checkable proofs (Lean), reinforcement learning from verifiable rewards (RLVR), and checking AI output.                                                          |
| Cryptography             | Confidential communication, digital signatures, certified computation, pseudorandomness.                          | Private AI (trusted execution environments, homomorphic encryption, differential privacy, secure multiparty computation) and watermarking neural networks.               |

**Bayesianism: an example.** Alice shoots a basketball at a hoop, and you have no idea how good she is. You start with a uniform prior over her accuracy, *Beta(1, 1)*. She misses the first shot, and your belief shifts towards low accuracy. She scores the next two, and it shifts back up. After each shot, your posterior combines your prior with the new evidence. When AI safety researchers say something like "a 10% chance of catastrophe this decade", they are trying to state such a posterior.

**Information theory.** Shannon famously generated ever-better random approximations to English text, from random letters to text that is almost readable. The two main results of the field are:

* *Source coding:* an i.i.d. stream of samples can be compressed down to its *entropy*, and no further.
* *Channel coding:* information can be sent reliably over a noisy channel at any rate below the channel's *capacity*.

Consider an AI that "depixelises" a blurred face. The output looks sharp, but information that is not in the blurred image cannot be recovered. The model invents a plausible face.

**Computability.** Regular languages can be recognised with finite memory. Context-free languages, such as palindromes, need a stack. The **Church–Turing thesis** says that every physically implementable model of computation can decide the same set of languages as a Turing machine. One argument goes that a bare LLM, which has finite memory, will fail to recognise palindromes beyond some length. An LLM in an agentic harness with external memory becomes Turing complete.

**Complexity.** Whether P = NP, and whether P = BQP, are big open problems. Satisfiability is NP-complete: every problem in NP can be reduced to it. Factoring is in BQP (quantum computers can do it efficiently) but is not believed to be in BPP (efficient classical randomised algorithms). The safety point is that some limits hold however smart an AI is. We do not expect AI on classical computers to factor large numbers efficiently, or to solve hard instances of PSPACE-hard problems such as quantified Boolean formulas.

**Formal logic.** From "Socrates is a man" and "all men are mortal", we derive "Socrates is mortal". Proof assistants such as Lean check proofs like this mechanically. This matters for using AI in mathematics, because we can check its output without trusting it. In one recent case study, Claude, running autonomously for 11 days with a multi-agent protocol, produced about 13 million lines of Lean towards a formal statement of Fermat's Last Theorem. It is also worth knowing Gödel's incompleteness theorems, which limit what formal systems can prove.

{% hint style="success" %}
**Discussion**

1. Pick one row of the table. How confident are you in that topic now? What would you need to learn to use it in safety research?
2. Complexity theory sets limits that hold "however smart AI is". What kinds of safety guarantee could rest on limits like these? What are the risks of relying on them?
3. If a bare LLM cannot recognise long palindromes but an agentic LLM can, what does this suggest about where capabilities come from: the model, or the system around it?
{% endhint %}

### Part 3: A tour of the Iliad Intensive curriculum

The rest of the [Iliad Intensive](https://iliad-intensive.org) covers five areas. This course covers selected parts of them in later weeks.

* **Alignment.** What alignment is; how it can go wrong (misspecification, misgeneralisation and so on); goal-directed agents and basic AI drives; whether AI safety may be impossible; foundational versus prosaic AI safety. LLM pre-training and post-training (supervised fine-tuning, RLHF), the linear representation hypothesis, Constitutional AI, RLVR, evaluations, chain-of-thought monitoring and AI control. Learning reward functions, not just specifying them. *See Weeks 2 and 3.*
* **Learning.** The paradox of deep learning and its resolutions, training dynamics, singular learning theory, data attribution (which training examples caused a behaviour), and the physics of learning. *See Weeks 4, 5, 7 and 8.*
* **Abstractions, representations and interpretability.** ML foundations and using Claude Code for research. *Mechanistic interpretability* reverse-engineers models to find the features and circuits behind their behaviour, using tools such as feature visualisation, the logit lens and representation engineering. Is it a misguided quest? Complex systems may not be simple underneath, examples can be cherry-picked, it can give a false sense of security, and it is dual-use. Research on *natural concepts* (mediators, redunds, natural latents) asks which abstractions any learner would find. One line of work finds that transformers represent *belief state geometry* in their residual stream.
* **Agency.** Axioms for preferences, utilities and rewards; the theory of RL; goal misgeneralisation; Solomonoff induction and AIXI; optimisation and thermodynamics; decision theory, its solution concepts and problem cases; open-source game theory; and attempts to formalise instrumental convergence. *See Week 6.*
* **Safety guarantees and their limits.** *Scalable oversight* asks how to keep human oversight meaningful as AI surpasses us. *Provably safe AI* checks AI outputs like mathematical proofs, though proofs can look convincing and be wrong, and only fully syntactic proofs can be checked. *AI safety via debate* uses ideas from interactive proofs and cross-examination. *Steganography* is hiding a message inside another message: a model could hide its real reasoning in its chain of thought, hide messages to other models, or smuggle out its own weights. *Backdoors* are hidden triggers planted in a model by its maker, or even by the learning algorithm, and some can be computationally undetectable.

{% hint style="success" %}
**Discussion**

1. Which of the five areas are you most drawn to, and why?
2. Above are several criticisms of mechanistic interpretability. Which one do you find most serious? Can you think of a reply?
{% endhint %}

---

## Further reading

* Bostrom, N. (2014). *Superintelligence: Paths, dangers, strategies*. Oxford University Press. Chapter 6, "Cognitive superpowers".
* Amodei, D. (2024). *Machines of loving grace*. [https://www.darioamodei.com/essay/machines-of-loving-grace](https://www.darioamodei.com/essay/machines-of-loving-grace).
* Kissinger, H., Schmidt, E. and Mundie, C. (2024). *Genesis: Artificial intelligence, hope, and the human spirit*. Little, Brown and Company.
* Omohundro, S. (2008). *The basic AI drives*. In *Proceedings of the First AGI Conference*.
* Sutton, R. (2019). *The bitter lesson*. [http://www.incompleteideas.net/IncIdeas/BitterLesson.html](http://www.incompleteideas.net/IncIdeas/BitterLesson.html).
* Shannon, C. E. (1948). *A mathematical theory of communication*. Bell System Technical Journal, 27.
* Shai, A. et al. (2024). *Transformers represent belief state geometry in their residual stream*. [https://arxiv.org/abs/2405.15943](https://arxiv.org/abs/2405.15943).
* For linear algebra: Gilbert Strang's lectures and books, and 3Blue1Brown's *Essence of Linear Algebra* series. For history: Crowe, *A History of Vector Analysis*.
* [BlueDot Impact](https://bluedot.org) courses and reading lists.

---

## References

* ILIAD. *Iliad Intensive curriculum*. [https://iliad-intensive.org](https://iliad-intensive.org). Source for the course overview, worldview and technical prerequisites, and the curriculum tour in Part 3. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
