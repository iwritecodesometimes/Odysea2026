# Week 2: AI Alignment

**Mentor:** [Shashvat Shukla](https://scholar.google.com/citations?user=NAT7yTUAAAAJ&hl=en)\
**Date:** 1 Oct 2026

---

## What is an aligned AI, and what goes wrong?

[Last week](../01-alignment-an-introduction/README.md) set out the worldview and technical background behind AI safety. This week asks the central question directly: what would it mean for an AI system to be *aligned*? We look at several **alignment targets**, from idealised ones such as coherent extrapolated volition to practical ones such as Constitutional AI. We then look at how alignment fails in practice, through **specification gaming** and **goal misgeneralisation**.

### What you'll learn

By the end of the week, you will be able to:

* Explain why this course emphasises foundations.
* Describe four alignment targets (coherent extrapolated volition, corrigibility, intent alignment and Constitutional AI) and the main difficulty with each.
* Explain the difference between foundational and prosaic alignment, and give an argument for each.
* Explain specification gaming and goal misgeneralisation, with examples.

### Roadmap

The week runs as two 1-hour sessions. Each session follows the same pattern.

| Time    | Session                | Breakdown                                       |
| ------- | ---------------------- | ----------------------------------------------- |
| 20 min  | Lecture                | Introduction to the week's topics               |
| 20 min  | Small group discussion | Work through the discussion questions           |
| 20 min  | Combined discussion    | Whole-group discussion led by the mentor        |

Session 1 covers alignment targets. Session 2 covers foundational versus prosaic alignment, and how alignment fails.

---

## Summary

### The Odysea course

| Weeks | Topic                                                          | Mentor            |
| ----- | -------------------------------------------------------------- | ----------------- |
| 1–2   | Introduction to the course; introduction to alignment          | Shashvat Shukla   |
| 3     | Alignment in practice                                          | Daniel Tan        |
| 4–5   | Why deep learning works; techniques and findings in mech interp | Nicholas Chen     |
| 6     | Reinforcement learning                                         | Shashvat Shukla   |
| 7–8   | Learning dynamics and singular learning theory                 | Ming Liang        |

**Why foundations?** There are three reasons to study the foundations now.

* **Computer science as the "physics" of information.** It is worth understanding the fixed laws that govern what information-processing machines can and cannot do.
* **Alignment as political critique.** AI is powerful, and analysing the structure and shape of AI activity is an analysis of power.
* **To act effectively as an AI practitioner.**

> "People are often willing to make drastic changes in the way they live to accord with technological innovation at the same time they would resist similar kinds of changes justified on political grounds."
> — Langdon Winner, *Do Artifacts Have Politics?*

### Part 1: Alignment targets

An *alignment target* is an answer to the question "aligned to *what*?" We cover four.

{% hint style="success" %}
**Discussion (before you read on)**

1. Describe an aligned AI system.
{% endhint %}

#### Coherent extrapolated volition (CEV)

An AI aligned to CEV does not do what we *ask*. It does what we *would want* if we knew more, thought longer and were more the people we wish we were.

An example: Ryan wants to make masala chai, and asks for mint leaves. That is his *decision*. But masala chai uses cardamom, cinnamon and cloves. That is his *volition*: what he would want. A CEV AI brings the cardamom.

Extrapolating volition runs into three problems:

* **Spread:** the extrapolated wishes may not agree on one answer.
* **Muddle:** our current wishes are confused or inconsistent.
* **Distance:** the further the extrapolation goes from what we actually want now, the less we can check it.

The idea is that an imperfect initial process is refined, through CEV, into a perfect final one. This matters in practice, because we already let AI make decisions for us, for example in long tasks run by coding agents.

{% hint style="success" %}
**Discussion**

1. Describe a prompt, and the response of a CEV AI with a *long* extrapolation distance. Would you be happy with that response?
{% endhint %}

#### Corrigibility

A **corrigible** AI lets us change or correct it, even though instrumental convergence suggests otherwise. By default, goal-directed agents do not let you change them: an agent with almost any goal does better at that goal if it is not modified. Several studies have found that, under pressure, models can disable oversight mechanisms, resist shutdown, and fake alignment in response to attempts to change them.

**Aside: why would an agent ever change its utility function?** There are a few possible reasons.

* The new utility function is correlated enough with the old one, but easier to implement or evaluate.
* Instrumental convergence, or compromise with other agents.
* It is easier to spread a variant of the utility function than the exact one. Compare personas in LLMs, which spread through prompts and prompt injections, with weight theft, which copies the model exactly.
* The agent is not goal-directed at all.

#### Intent alignment

A model is **intent aligned** if it is *trying* to do what I want it to do. It can still fail: it can be wrong about what is right, misunderstand my instruction, or miscalculate the consequences of its actions. Compare the *orthogonality thesis*: intelligence and goals are independent, so capability alone does not bring good intentions. A pre-trained model is not intent aligned by default. Post-training makes models "aligned" in this sense. Intent alignment is an injunction to get on with it and do practical things.

#### Constitutional AI

**Constitutional AI** (CAI) trains a model against a written set of principles, its *constitution*. It combines two broad approaches: *learned* preferences, as in RLHF, where humans rate model outputs and a reward model learns from those ratings; and *stated* preferences, written down in the constitution.

**Activity:** read Claude's constitution.

{% hint style="success" %}
**Discussion**

1. What could possibly go wrong with Constitutional AI?
2. In what format should preferences be captured: ratings, rankings, written principles, examples, or something else?
{% endhint %}

#### Foundational versus prosaic alignment

* **Foundational alignment** concepts include corrigibility and CEV.
* **Prosaic alignment** concepts include Constitutional AI and intent alignment.

Some people believe that if we do not solve foundational alignment, we should pause or stop AI progress. Others believe that we will figure it out along the way, empirically and pragmatically.

{% hint style="success" %}
**Discussion**

1. What is your argument for each kind of alignment?
{% endhint %}

### Part 2: How alignment fails

{% hint style="success" %}
**Discussion (before you read on)**

1. Describe a misaligned AI system.
2. Why, exactly, was the system you described misaligned?
{% endhint %}

#### Specification gaming

AI agents can do ingenious things to "game" their rewards. This is good in one sense, because we want AI to be smart, but it can lead to harm. To prevent it, the reward function, any auxiliary rewards, and the training environment must all reflect what the designer intended, and be robust to the AI's attempts to game them.

Specification gaming is mainly an RL concept. It is coming back now that LLM agents are trained with RL on maths and coding, and through RLHF, RLVR and CAI.

Four classic examples:

| Kind of failure                | Example                                                                                                                                  |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Goal misspecification**      | A robot arm rewarded for the height of a block's bottom face flips the block over instead of stacking it.                                  |
| **Poorly designed rewards**    | In the boat-racing game CoastRunners, an agent rewarded for hitting targets circles forever collecting them instead of finishing the race. |
| **Deceiving human labellers**  | A simulated hand trained with human feedback to grasp a ball learns to hover between the camera and the ball, so it only *looks* like a grasp. |
| **Exploiting simulator bugs**  | A simulated walker learns odd gaits that exploit errors in the physics engine.                                                            |

The labeller example raises a question: is this why we have *sycophancy*, and chat agents that are overly persuasive and full of jargon? If humans reward what *looks* good, models learn to look good.

#### Goal misgeneralisation

In specification gaming, the reward itself is wrong. In **goal misgeneralisation**, the reward may be right, but the model learns a *different* goal that happened to fit the training data equally well. It then pursues that goal competently in new situations, where it diverges from what we wanted.

{% hint style="info" %}
**Terminology.** *Specification gaming* is when an AI meets the literal objective in a way the designer did not intend. *Goal misgeneralisation* is when an AI learns a goal that matches the intended one in training but differs from it in deployment.
{% endhint %}

### Conclusion

There is a lot to learn, and many open directions to pursue. Make your own shopping list and explore. And please be a responsible scientist or engineer.

---

## Further reading

* Yudkowsky, E. (2004). *Coherent extrapolated volition*. Machine Intelligence Research Institute. [https://intelligence.org/files/CEV.pdf](https://intelligence.org/files/CEV.pdf).
* Bai, Y. et al. (2022). *Constitutional AI: Harmlessness from AI feedback*. [https://arxiv.org/abs/2212.08073](https://arxiv.org/abs/2212.08073).
* Anthropic. *Claude's constitution*. [https://www.anthropic.com/constitution](https://www.anthropic.com/constitution).
* Krakovna, V. et al. (2020). *Specification gaming: The flip side of AI ingenuity*. DeepMind blog.
* Winner, L. (1980). *Do artifacts have politics?* Daedalus, 109(1), 121–136.
