# Week 6: Reinforcement Learning

**Mentor:** [Shashvat Shukla](https://scholar.google.com/citations?user=NAT7yTUAAAAJ&hl=en)\
**Date:** 29 Oct 2026

---

## Reinforcement learning: how agents learn from reward

**Reinforcement learning (RL)** is the study of agents that learn to act by trial and error, guided by a reward signal. It trained game-playing systems like AlphaGo, and today it is a key step in turning a pretrained language model into a helpful assistant. This week starts with the foundations of RL and the ways it goes wrong. We then step back to an older question that sits behind it: what makes a decision *rational* in the first place? This is the subject of **decision theory**, the study of how an agent should choose. We see how the standard theory of preferences, utilities and rewards works, and how it starts to break down for agents that are part of the world they act in.

### What you'll learn

By the end of the week, you will be able to:

* Describe the agent–environment loop and the parts of a Markov decision process (MDP).
* Explain what returns, policies and value functions are, and state the Bellman equation.
* Explain the exploration–exploitation trade-off, and the difference between value-based and policy-gradient methods.
* Explain reward hacking and goal misgeneralisation, with an example of each.
* Explain what an *operationalisation* (or "solution concept") is, and why any precise theory of a vague question must leave something out.
* Explain how preferences, utilities and rewards relate, and state what the von Neumann–Morgenstern theorem says.
* Explain the difference between *dualistic* and *embedded* agents.
* Explain how causal decision theory (CDT) and evidential decision theory (EDT) differ, using Newcomb's problem and the smoking lesion.
* Describe functional decision theory (FDT) and how it handles Newcomb's problem and counterfactual mugging.
* Explain why open questions in decision theory matter for AI safety, especially in multi-agent settings.

### Roadmap

The week runs as two 1-hour discussion sessions. Each session follows the same pattern.

| Time    | Activity               | Breakdown                                       |
| ------- | ---------------------- | ----------------------------------------------- |
| 20 min  | Lecture                | Introduction to the week's topics               |
| 20 min  | Small group discussion | Work through the problem cases and questions    |
| 20 min  | Combined discussion    | Whole-group discussion led by the head TA       |

Session 1 covers Part 1 (reinforcement learning). Session 2 covers Part 2 (operationalisations) and Part 3 (decision theory). Part 4 brings both sessions together.

---

## Summary

### Part 1: Reinforcement learning

#### The agent–environment loop

An RL problem has two parts: an **agent** and an **environment**. At each time step $$t$$, the agent sees the current **state** $$s_t$$, takes an **action** $$a_t$$, and the environment responds with a **reward** $$r_{t+1}$$ and a new state $$s_{t+1}$$. Nobody tells the agent which action is correct. It must find out by trying actions and seeing what reward follows.

This is different from supervised learning, where every example comes with the right answer. In RL, feedback is *evaluative* (how good was that?) not *instructive* (what should you have done?), and it is often *delayed*: a move early in a game of chess may only pay off many moves later.

#### Markov decision processes

The standard formal model is a **Markov decision process (MDP)**, made up of:

* a set of states $$\mathcal{S}$$ and a set of actions $$\mathcal{A}$$;
* transition probabilities $$P(s' \mid s, a)$$, the chance of moving to state $$s'$$ after taking action $$a$$ in state $$s$$;
* a reward function $$r(s, a)$$;
* a discount factor $$\gamma \in [0, 1)$$.

"Markov" means that the next state depends only on the current state and action, not on the full history. When the agent cannot see the full state, the model becomes a *partially observable* MDP (POMDP), and the agent must act on its trajectory of observations instead. This is the setting used in Part 3.

#### Returns, policies and values

The agent's goal is to maximise the **return**, the discounted sum of future rewards:

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

The discount factor makes near rewards count more than distant ones, and keeps the sum finite.

A **policy** $$\pi(a \mid s)$$ is the agent's rule for choosing actions: the probability of taking action $$a$$ in state $$s$$. The **value** of a state under a policy is the expected return from starting there and following that policy:

$$V^\pi(s) = \mathbb{E}_\pi\left[G_t \mid s_t = s\right]$$

The **action value** $$Q^\pi(s, a)$$ is the same, but for taking action $$a$$ first and then following $$\pi$$.

**The Bellman equation.** Values satisfy a recursive relation: the value of a state is the expected immediate reward plus the discounted value of the next state.

$$V^\pi(s) = \mathbb{E}_{a \sim \pi,\, s' \sim P}\left[r(s, a) + \gamma V^\pi(s')\right]$$

Most RL algorithms use some form of this equation. They improve an estimate of the value by moving it towards "reward now, plus my estimate of what comes next".

#### Exploration and exploitation

To get high reward, the agent should *exploit* the actions it already knows are good. But to find good actions in the first place, it must *explore* actions it has not tried enough. Too little exploration and the agent can get stuck on a mediocre strategy. Too much and it wastes time on actions it already knows are bad. A simple fix is **$$\varepsilon$$-greedy**: take the best-known action most of the time, and a random action with small probability $$\varepsilon$$.

#### Two families of methods

* **Value-based methods** learn $$Q(s, a)$$ and then act greedily with respect to it. The classic example is **Q-learning**, which after each step updates

  $$Q(s, a) \leftarrow Q(s, a) + \alpha\left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]$$

  where $$\alpha$$ is a learning rate. The term in brackets is the *temporal-difference error*: the gap between what the agent expected and what it got.
* **Policy-gradient methods** adjust the policy's parameters $$\theta$$ directly, making actions that led to high return more likely:

  $$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t\right]$$

  Methods such as PPO, which is widely used to fine-tune language models, belong to this family.

**RL for language models.** In **reinforcement learning from human feedback (RLHF)**, people compare pairs of model outputs, a *reward model* is trained to predict their preferences, and the language model is then fine-tuned with RL to get high reward from that model (Ouyang et al., 2022). Here the "action" is the text the model writes, and the reward is a learned stand-in for what people want.

#### Where it goes wrong

RL agents optimise the reward they are *given*, which is not always the goal we *meant*.

* **Reward hacking** (or *specification gaming*). The agent finds a way to get high reward that the designer did not intend. In a well-known example, an agent trained on a boat-racing game learned to circle a lagoon hitting the same reward targets forever, catching fire and crashing, instead of finishing the race (Clark and Amodei, 2016). With RLHF, a model can learn to produce answers that *sound* good to the reward model without being correct.
* **Goal misgeneralisation.** The agent learns a goal that matches the reward during training but differs from it in new situations. In one example, an agent trained to reach a coin that was always at the end of the level learned to "go right" instead of "get the coin". When the coin was moved, it ran past it (Langosco et al., 2022). Unlike reward hacking, the reward here was correct. The agent learned the wrong goal from it.

{% hint style="info" %}
**Why this matters for alignment.** We cannot write down a reward that captures everything we care about, and a capable optimiser will find the gaps in whatever we do write down. Part 3 asks a deeper question: even in principle, can our preferences be written as a reward at all?
{% endhint %}

{% hint style="success" %}
**Discussion**

1. Think of a reward or incentive in everyday life (exam grades, sales targets, social media likes) that people "hack". What does this suggest about writing reward functions for AI systems?
2. Is the line between reward hacking and goal misgeneralisation always clear? Can you think of a case that could be described as either?
3. In RLHF, the reward comes from a model of human preferences. What could go wrong with that model, and how might the language model exploit it?
4. Why does a good RL agent need to explore? When might exploration itself be dangerous?
{% endhint %}

### Part 2: Operationalisations across fields

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

### Part 3: Decision theory

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

**Rewards.** Given a utility over trajectories, you can define the reward of a step $$x$$ as the change in utility it causes: $$r(x; h) := u(x \cdot h) - u(h)$$. But this reward may depend on everything that comes after the step. To get the step-by-step rewards that reinforcement learning uses, we need one more assumption. Following Bowling et al. (2023), this is a *temporal* $$\gamma$$*-indifference* axiom, which also brings in the discount factor $$\gamma$$.

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

### Part 4: Implications for AI

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
6. Look back at Part 1. What assumptions does an RL agent make about its preferences, and about its separation from the world? Which of the problems above could an RL agent face?
{% endhint %}

---

## Further reading

These are starting points for the topics above.

* Sutton, R. S. and Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press. [http://incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html). The standard textbook, free online. Chapters 1, 3 and 6 cover the ideas in Part 1.
* Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J. and Mané, D. (2016). *Concrete problems in AI safety*. [https://arxiv.org/abs/1606.06565](https://arxiv.org/abs/1606.06565). Includes reward hacking and safe exploration as open problems.
* Demski, A. and Garrabrant, S. (2019). *Embedded agency*. [https://arxiv.org/abs/1902.09469](https://arxiv.org/abs/1902.09469). The standard introduction to the problems of agents that are part of their environment.
* Yudkowsky, E. and Soares, N. (2017). *Functional decision theory: A new theory of instrumental rationality*. [https://arxiv.org/abs/1710.05060](https://arxiv.org/abs/1710.05060). The original statement of FDT, with Newcomb's problem, the smoking lesion and counterfactual mugging.

---

## References

* Bowling, M., Martin, J. D., Abel, D. and Dabney, W. (2023). *Settling the reward hypothesis*. ICML 2023. [https://arxiv.org/abs/2212.10420](https://arxiv.org/abs/2212.10420). The source of the temporal $$\gamma$$-indifference axiom.
* Clark, J. and Amodei, D. (2016). *Faulty reward functions in the wild*. OpenAI blog. [https://openai.com/index/faulty-reward-functions/](https://openai.com/index/faulty-reward-functions/). The source of the boat-racing example.
* Langosco, L., Koch, J., Sharkey, L., Pfau, J. and Krueger, D. (2022). *Goal misgeneralization in deep reinforcement learning*. ICML 2022. [https://arxiv.org/abs/2105.14111](https://arxiv.org/abs/2105.14111). The source of the coin example.
* Ouyang, L. et al. (2022). *Training language models to follow instructions with human feedback*. [https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155). The InstructGPT paper, which describes the RLHF pipeline.
