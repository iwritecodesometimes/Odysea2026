# Week 6: Decision Theory

**Author:** Odysea 2026 Technical Team\
**Mentor:** TBD\
**Date:** 29 Oct 2026

---

## Decision theory: what is a good decision?

A reinforcement learning agent chooses actions to get high reward. Behind that sits an older question: what makes a decision *rational* in the first place? This week looks at **decision theory**, the study of how an agent should choose. We start with a general point about how fields turn vague questions into precise theories. We then see how the standard theory of preferences, utilities and rewards works, and how it starts to break down for agents that are part of the world they act in.

### What you'll learn

By the end of the week, you will be able to:

* Explain what an *operationalisation* (or "solution concept") is, and why any precise theory of a vague question must leave something out.
* Explain how preferences, utilities and rewards relate, and state what the von Neumann–Morgenstern theorem says.
* Explain the difference between *dualistic* and *embedded* agents.
* Explain how causal decision theory (CDT) and evidential decision theory (EDT) differ, using Newcomb's problem and the smoking lesion.
* Describe functional decision theory (FDT) and how it handles Newcomb's problem and counterfactual mugging.
* Explain why open questions in decision theory matter for AI safety, especially in multi-agent settings.

### Roadmap

The week runs as two 1-hour discussion sessions. Each session follows the same pattern.

| Time    | Session                | Breakdown                                       |
| ------- | ---------------------- | ----------------------------------------------- |
| 20 min  | Lecture                | Introduction to the week's topics               |
| 20 min  | Small group discussion | Work through the problem cases and questions    |
| 20 min  | Combined discussion    | Whole-group discussion led by the head TA       |

Session 1 covers Part 1 (operationalisations) and the background in Part 2. Session 2 covers the decision problems and the implications for AI.

---

## Summary

### Part 1: Operationalisations across fields

**The pattern.** Many investigations in philosophy and the formal sciences share one structure:

1. A **vague question**, such as "What is goodness?"
2. Several **systematic theories** that answer it in a precise way.
3. **Problem cases** that people use to argue for and against each theory.

Three examples:

| Field       | Vague question                        | Systematic theories                                                                                                                                 | Problem case                                                         |
| ----------- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Ethics      | What is goodness?                     | *Consequentialism* (judge an action by its consequences), *deontology* (follow the right moral rules), *virtue ethics* (have the right dispositions) | Trolley problems, which separate consequentialism from deontology    |
| Game theory | How are games played rationally?      | *Nash equilibrium* (no player wants to change their choice), *correlated equilibrium* (players respond to a shared private signal)                  | Battle of the sexes, which motivates correlated equilibrium          |
| Statistics  | How do we reason from data rationally? | *Bayesianism* (update a prior on evidence), *error statistics* (test whether a result is real or noise)                                            | The Jeffreys–Lindley paradox, where the two approaches disagree      |

**Practice: the trolley problem.** Consider several versions. In the classic version, you can pull a lever to send a runaway trolley away from five people and towards one. In the footbridge version, you can stop the trolley only by pushing a person off a bridge onto the track. In the last version, the number of people on the main track grows without limit. Many people pull the lever but will not push, though both trade one life for five. Each version puts pressure on a different theory.

**A thesis on operationalisations.**

* A theory answers a vague, open-ended question in a precise, systematic way.
* As a result, every such answer *necessarily* misses something in the original question.
* Theories tell you where they are best applied.
* *Problem cases* are situations where no theory clearly applies better than the others.
* *Paradoxes* show either that something is wrong with a theory, or that something goes wrong when you follow it too far.

Keep this thesis in mind for the rest of the week. Decision theories are operationalisations of "what is a good decision?", and the problem cases below are where they come apart.

{% hint style="success" %}
**Discussion**

1. Work through the trolley variants above. For each one, what would a consequentialist do, and what would a deontologist do? Which variant makes you most uncomfortable with your own view?
2. Pick a field you know well. What is its vague question, what are its systematic theories, and what are its problem cases?
3. The thesis says that any precise answer must miss something in the original question. Do you agree? Is there a precise theory you think captures its question completely?
{% endhint %}

### Part 2: Decision theory

#### Two senses of "decision theory"

In many fields, decision theory is the study of how to make rational decisions in complex situations with uncertainty. This matters in business, finance and economics. In AI safety, the interest is narrower: decision theory for **embedded agents**.

* A **dualistic** agent is cleanly separate from the world. It picks an action, and the world responds. Examples include whether to take a bet (the Monty Hall problem, gambler's ruin), practical optimisation (the secretary problem), and many reinforcement learning agents.
* An **embedded** agent is *part of* the world it acts in. Other parts of the world can contain copies of it, predictions of it, or information about how it decides. Most of the hard problems below come from this.

#### Background: preferences, utilities and rewards

We build up the standard theory step by step.

**Trajectories.** Let $$\mathcal{O}$$ be a finite set of observations and $$\mathcal{A}$$ a finite set of actions. A *trajectory* is a sequence of observation–action pairs, $$h = (o_1, a_1, o_2, a_2, \ldots, o_n, a_n)$$. Write $$\mathcal{H}^*$$ for the set of all finite trajectories.

**Preferences.** A *preference relation* $$\succcurlyeq$$ on $$\mathcal{H}^*$$ says, for two trajectories, whether one is at least as good as the other. A preference relation is:

* *complete* if any two trajectories can be compared: $$h \succcurlyeq h'$$ or $$h' \succcurlyeq h$$, or both;
* *transitive* if $$h \succcurlyeq h'$$ and $$h' \succcurlyeq h''$$ together imply $$h \succcurlyeq h''$$.

**Utilities.** A preference relation is complete and transitive if and only if a *utility function* $$u : \mathcal{H}^* \to \mathbb{R}$$ represents it, so that $$h \succcurlyeq h' \iff u(h) \geq u(h')$$.

**Uncertainty: the von Neumann–Morgenstern theorem.** Now let the agent compare *lotteries* (probability distributions) over outcomes. Suppose preferences over lotteries satisfy four axioms: completeness, transitivity, **continuity** (no outcome is infinitely better or worse than another) and **independence** (mixing both options with the same third lottery does not change which one you prefer). Then there is a utility function $$u$$ such that the agent prefers one lottery to another exactly when it has higher *expected* utility. The function $$u$$ is unique up to a positive affine transformation $$u'(x) = a + b\,u(x)$$ with $$b > 0$$.

**Rewards.** Given a utility over trajectories, you can define the reward of a step as the change in utility it causes: $$r(t; h) := u(t \cdot h) - u(h)$$. But this reward may depend on everything that comes after the step. To get the step-by-step rewards that reinforcement learning uses, we need one more assumption. Following Bowling et al. (2023), this is a *temporal* $$\gamma$$*-indifference* axiom, which also brings in the discount factor $$\gamma$$.

{% hint style="info" %}
**Why this matters.** Reinforcement learning assumes that what we want can be written as a sum of (discounted) rewards. The axioms above show that this is a *substantive assumption* about our preferences, not a free choice. If real preferences break any of the axioms, no reward function captures them exactly.
{% endhint %}

#### Background: counterfactuals

A plain conditional says "if A, then B". A **subjunctive conditional**, or *counterfactual*, says "if A *were* the case, B *would* be the case". Deciding means asking counterfactuals: "if I took this action, what would happen?"

Counterfactuals are philosophically tricky. One of the best accounts says that we evaluate them by looking at *nearby possible worlds* where A is true. But this account breaks down for embedded agents. If a predictor has modelled your decision, what is the "nearby world" where you choose differently? Does the prediction change too?

#### Newcomb's problem

A reliable predictor puts two boxes in front of you. Box A is transparent and holds $1,000. Box B is opaque. The predictor put $1,000,000 in Box B if it predicted that you would take *only* Box B, and nothing if it predicted that you would take both.

| Your choice                  | Predictor predicted one-box | Predictor predicted two-box |
| ---------------------------- | --------------------------- | --------------------------- |
| One-box (take only Box B)    | $1,000,000                  | $0                          |
| Two-box (take both boxes)    | $1,001,000                  | $1,000                      |

Whatever is already in the boxes, two-boxing gives you $1,000 more. But people who one-box almost always walk away rich.

#### Causal and evidential decision theory

The two classical decision theories answer different questions.

* **Causal decision theory (CDT):** Which action *causes* the best outcome?
* **Evidential decision theory (EDT):** Which action would be the best *news* to learn that I took?

In Newcomb's problem, CDT reasons that your choice cannot change what is already in the boxes, so it two-boxes. EDT notes that one-boxing is strong evidence that Box B is full, so it one-boxes.

#### The smoking lesion

In this imaginary world, smoking does *not* cause cancer. Instead, some people have a lesion that makes them both more likely to smoke and more likely to get cancer.

| Your choice  | You have the lesion          | You do not have the lesion  |
| ------------ | ---------------------------- | --------------------------- |
| Smoke        | Cancer, enjoyment (−99)      | No cancer, enjoyment (+1)   |
| Don't smoke  | Cancer, no enjoyment (−100)  | No cancer, no enjoyment (0) |

Smoking is better by 1 in both columns, and it cannot cause cancer. So CDT smokes. But a naive EDT agent notices that smoking is *bad news*, since it is evidence of the lesion, and so refuses to smoke. Here EDT seems to get it wrong.

#### Choices as confounders, not causes

The smoking lesion is a case where your choice is a sign of something, not a cause of it. Two more everyday examples:

* **Free will and fatalism.** "The world is determined and I don't have it in me to succeed" versus "the world is determined, but I am lucky to have a can-do spirit".
* **Good schools.** Are good schools good because they give students a good education (a cause), or because students who are likely to do well go there (a confounder)?

#### More problem cases

**The toxin puzzle.** A billionaire offers you a deal. If, at midnight tonight, you *intend* to drink a toxin tomorrow, you get a million dollars now. The toxin will make you very ill for a day. (1) Once you have the money, do you need to drink the toxin? (2) If you know that you rationally would not drink it once you have the money, can you even honestly intend to drink it? See also Parfit's hitchhiker, and side bets.

**The prisoner's dilemma with your clone.** You play a one-shot prisoner's dilemma against an exact copy of yourself. Your choice is evidence of what your clone will do. CDT notes that your choice cannot cause your clone's, so it **defects**. EDT notes that if you cooperate, your clone almost certainly does too, so it **cooperates**.

#### A synthesis?

Each side seems to be right about something.

* **Causal decision theorists** are right that the causal structure of the world matters for good decisions, and that your actions shape the world. It is useful to know how genes influence both smoking and cancer, or how good schools improve your life.
* **Evidential decision theorists** are right that *the kind of person you are* matters for the outcomes you get. It is good to have a reputation for cooperation, or to improve your judgement in general.

This suggests a question. Should rationality give only one answer? Or are there two kinds of rationality: one for **decisions**, which are short-term, and one for **character formation**, which is long-term?

There is a parallel with moral philosophy. We can judge goodness at the level of the consequences, the action, the rule followed, or the character. Or at all of them.

#### Functional decision theory

Newcomb's problem and the smoking lesion look similar, but they differ in one way. The lesion is there or not, *whatever you choose*. The prediction in Newcomb's problem, by contrast, is linked to your future choice: the predictor ran something like *your decision procedure*. **Functional decision theory (FDT)** is built to be sensitive to this difference.

* **FDT:** Which *output of this very decision function* would give the best outcome?

FDT treats your decision as the output of a function that may also be running elsewhere: in a predictor's model, or in your clone. It chooses the output that is best across every place the function runs. So FDT one-boxes in Newcomb's problem, smokes in the smoking lesion, and cooperates with its clone.

**Counterfactual mugging.** A predictor tells you: "I flipped a coin. If it came up heads, I would give you a million dollars, but only if I predicted that you would pay me $100 when it came up tails. It came up tails. Please give me $100." Paying now gains you nothing. But FDT **pays**, because the policy "pay when asked" is the one that does best before the coin is flipped. In the same way, FDT one-boxes in the *transparent* Newcomb problem, where you can see that Box B is full before you choose.

{% hint style="info" %}
**Terminology.** *CDT* chooses the action that causes the best outcome. *EDT* chooses the action that is the best evidence of a good outcome. *FDT* chooses the output of its own decision procedure that gives the best outcome, across every place that procedure is run or predicted.
{% endhint %}

### Part 3: Implications for AI

Formal decision problems rest on many assumptions. Several "science fiction" scenarios from the problems above are realistic for AI systems:

* AI systems may be able to read each other's code, and so predict how others will behave, much like Newcomb's predictor.
* Copies of the same model can play games against each other, like the prisoner's dilemma with your clone.

There is no settled answer to what counts as a "correct" decision. This is a theoretical barrier to understanding what it even means for an autonomous system to be trustworthy, or to do what we want. And failure in multi-agent settings is a growing AI safety concern.

{% hint style="success" %}
**Discussion**

1. In Newcomb's problem, would you one-box or two-box? Can you state the strongest argument for the *other* choice?
2. In the smoking lesion, CDT seems right and EDT seems wrong. In Newcomb's problem, many people think the reverse. What is the difference between the two problems? How does FDT use that difference?
3. The toxin puzzle asks whether you can honestly intend to do something that you know you will have no reason to do. What does this have to do with reputation, promises, or an AI system's commitments?
4. Is there one rational answer to these problems, or are there two kinds of rationality, one for decisions and one for character? Which kind should we want an AI system to have?
5. Two copies of the same AI model play a prisoner's dilemma. Should they cooperate? Would you *want* them to, if the "game" were, for example, two AI systems deciding whether to report each other's misbehaviour to their developers?
6. How does this week connect to reinforcement learning? What assumptions does an RL agent make about its preferences, and about its separation from the world?
{% endhint %}

---

## Further reading

These are starting points for the topics above.

* Demski, A. and Garrabrant, S. (2019). *Embedded agency*. [https://arxiv.org/abs/1902.09469](https://arxiv.org/abs/1902.09469). The standard introduction to the problems of agents that are part of their environment.
* Yudkowsky, E. and Soares, N. (2017). *Functional decision theory: A new theory of instrumental rationality*. [https://arxiv.org/abs/1710.05060](https://arxiv.org/abs/1710.05060). The original statement of FDT, with Newcomb's problem, the smoking lesion and counterfactual mugging.

---

## References

* Bowling, M., Martin, J. D., Abel, D. and Dabney, W. (2023). *Settling the reward hypothesis*. ICML 2023. [https://arxiv.org/abs/2212.10420](https://arxiv.org/abs/2212.10420). The source of the temporal $$\gamma$$-indifference axiom.
