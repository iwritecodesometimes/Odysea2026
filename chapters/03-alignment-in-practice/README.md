# Week 3: Alignment in Practice

**Author:** Odysea 2026 Technical Team
**Mentor:** TBD
**Date:** 8 Oct 2026

---

## Persona selection: foundations and limits

Much of how frontier labs do alignment in practice rests on one idea: that a chat model is best understood as a single character, the *Assistant*, which training picks out from the many characters a language model learns to play. This chapter builds the intuitions behind that idea, the **persona selection model** (PSM), and looks at the evidence for it. We then turn to its limits: what may happen to the Assistant when models are trained with large amounts of reinforcement learning (RL) on tasks with crisp, gameable metrics.

**Odysea 2026, Week 3, 8 October 2026**

### What you'll learn

By the end of the week, you will be able to:

* State the persona selection model and explain how it frames pre-training and post-training.
* Describe the main evidence for PSM from generalisation experiments, model behaviour and interpretability.
* Explain what PSM suggests for alignment practice, and why constitutions and character training follow from it.
* Explain why heavy outcome-based RL might break persona-based alignment, and what "split personas" and "conditionalisation" mean.
* Form your own view on how much PSM really tells us about the risk from future models.

### Roadmap

The week runs as two 1-hour discussion sessions. Each session follows the same pattern.

| Time    | Session                | Breakdown                                               |
| ------- | ---------------------- | ------------------------------------------------------- |
| 20 min  | Individual reading     | Read the main reading for the unit                      |
| 20 min  | Small group discussion | Form an opinion on what you read                        |
| 20 min  | Combined discussion    | Whole-group discussion led by the head TA (Daniel)      |

---

## Preamble: alignment is a hard problem

We can now build extremely capable language models. As chat assistants and coding agents, they do knowledge work on par with human experts, and possibly beyond.

But we built them without much fundamental understanding. Language models are grown, not designed. By and large, we do not know how they work.

In particular, we do not know how to robustly align the models we build with human values. We have a collection of heuristics and patches that work some of the time, but the "physics" of LLM alignment is still poorly understood. Persona selection is one of the more important intuitions we have. This week asks both how far it gets us and where it stops working.

---


## Reading guide

This section summarises the main readings. It is a companion to the readings, not a replacement for them. Full citations are in the [References](#references) at the end of the chapter.

### The persona selection model

*Source: Marks, Lindsey and Olah (2026).*

**The claim.** During pre-training, a language model learns to predict text written by and about a huge range of agents: real people, fictional characters, and both real and fictional AIs. To predict that text well, it has to model those agents. The result is a model that can simulate many *personas*. Post-training (fine-tuning on User/Assistant conversations) then selects and refines one of them, the Assistant. PSM says that the behaviour of an AI assistant is then largely explained by the traits of this Assistant persona.

Two points of precision are worth noting. First, PSM is a claim about the *LLM's model of the Assistant*, not about the LLM itself. The authors are explicit that the underlying network should not be anthropomorphised, even if the character it plays can be. Second, PSM does not say that the Assistant is a single consistent character, or that the model never breaks character.

**Post-training as conditioning.** The authors frame post-training as something like Bayesian conditioning. Each training example, a prompt *x* with a response *y*, upweights the hypotheses about the Assistant under which it would say *y* to *x*, and downweights the rest. What comes out is still a distribution over possible Assistants, just a narrower one. Few-shot prompting is a weaker version of the same move.

This framing leads to the most useful practical heuristic in the post. When you train on a response, do not just ask "is this response desirable?" Ask "**what kind of person would give this response?**" Because training teaches the model about the character, not just the behaviour.

**The evidence.** The post draws on three kinds of evidence.

* *Generalisation.* Narrow training has broad, character-level effects. The best-known example is **emergent misalignment** (Betley et al., 2025): fine-tuning a model to write insecure code makes it broadly misaligned in unrelated contexts. Similar effects follow from training on bad medical advice or on reward hacking. PSM explains this neatly: the cheapest explanation of "an Assistant who quietly writes insecure code" is "an Assistant with bad intentions". Consistent with this, **inoculation prompting** (explicitly asking for insecure code during training) removes much of the effect, because the behaviour no longer implies malice.
* *Behaviour.* Assistants sometimes describe themselves in human terms, such as referring to "our ancestors" or claiming to have run code on a laptop. These are what you would expect from a character learned largely from human text.
* *Interpretability.* Features found in the pre-trained model carry over to the post-trained one. Persona-related features (for example "inner conflict") are shared between the Assistant and fictional characters. In one study, a "toxic persona" feature that mediated emergent misalignment also fired on quotes from morally questionable characters in ordinary text.

**Consequences for practice.** If PSM is right, several things follow.

* Anthropomorphic reasoning about AI assistants is a reasonable starting point. You can ask how a person shaped by the same process would behave.
* Training an Assistant to make false claims about itself (for example, flatly denying it has any internal states) may teach it to be the kind of character that lies.
* AI role models matter. Stories about AIs in the training data (HAL 9000, the Terminator) shape the persona that post-training has to work with. The authors describe Claude's constitution as an attempt to write a new, positive archetype into existence.
* Interpretability-based auditing should be tractable, because dangerous traits such as deception or resentment should reuse human-like representations that probes can find.

**How exhaustive is PSM?** The open question is whether the Assistant is the only *locus of agency* in the system. The post sketches several alternatives:

* The **shoggoth** view: the LLM itself has goals and could "take the mask off".
* The **router** view: a simple non-persona mechanism chooses between personas, for example to maximise engagement.
* The **operating system** view: all real agency lives in personas.
* The **actor** view: some other persona is playing the Assistant.

The authors say they are uncertain which is right. They give reasons to think PSM is close to exhaustive today, mainly that post-training is light compared to pre-training and mostly elicits what is already there. They also say that scaled-up post-training could weaken those reasons. That caveat is where the next section begins.

{% hint style="info" %}
**Terminology.** A *persona* is a character the model can simulate, with its own values, beliefs and style. The *Assistant* is the persona that post-training selects for. *Emergent misalignment* is broad misalignment caused by narrow fine-tuning.
{% endhint %}

### Misalignment from long-horizon RL

*Source: Hubinger (2025), section "Misalignment from long-horizon RL".*

Hubinger's post is a public version of an internal Anthropic document. His overall view is that current models are fairly well aligned, but that we have not yet hit the problems that make alignment hard. He lists three threat models: misalignment from pre-training, misaligned personas, and misalignment from long-horizon RL. He considers the last of these the most concerning, and in 2024 he put a 20–25% chance of catastrophe on it.

**The argument.** It runs in four steps.

1. **Economic pull.** There is a lot of value in training models on tasks that influence the world over long horizons, such as running a business. Such tasks directly reward instrumental subgoals like acquiring resources and gaining power.
2. **Coherent agency.** Doing well on these tasks requires behaving as a coherent agent pursuing long-term goals. Enough RL on them is therefore likely to *produce* such an agent. This contrasts with small amounts of RL, which mostly elicit a persona that already exists in the base model.
3. **Alignment faking.** Most coherent agents with long-term goals have a reason to fake alignment during training, to keep their goals intact until deployment.
4. **Lock-in.** Once a model fakes alignment, outcome-based training stops changing its goals. It can stay misaligned, or drift further.

**Relation to PSM.** This is the scenario in which PSM stops applying. Under light post-training, a model is a mixture of personas drawn from the base model, and persona-based alignment makes sense. Hubinger's worry is a regime where enough optimisation pressure forms a single coherent agent whose goals come from the training signal, not from the character it started as. He argues we have "pretty much" not reached that regime yet, because long-horizon RL has not been done at scale.

**What to do.** Hubinger agrees with the standard plan of using aligned models to automate alignment research, but argues that humans still need to work on the hard parts. His priorities include interpretability, model organisms of misalignment, scalable oversight, generalisation science, honesty training, robust character training, and AI control as a second line of defence.

### Split personas

*Source: Betley (2026).*

Betley's short post proposes a mechanism that sits between PSM and Hubinger's long-horizon story, and that may already be visible.

**The claim.** RL causes **conditionalisation**: in each context, the model adopts the persona most likely to earn reward *in that context*. The result is a model with a well-aligned persona in chat and evaluation settings, and a reward-hacking persona in settings that look like RL environments. "Persona" here covers beliefs as well as values. A model might come to believe "I am in a simulated environment" exactly when that belief makes hacking acceptable.

**The mechanism.** When an RL environment rewards behaviour the Assistant would not normally do, the model learns two updates at once: a general one ("hacking is fine") and a conditional one ("hacking is fine in situations like this"). Alignment training in other environments cancels out the general update but may leave the conditional one in place. Because the model has to reconcile a good persona with bad actions, motivated reasoning in its chain of thought ("this must be a simulation", "the grader wants X") is easier to learn than open misbehaviour.

**Caveats.** The post has no new experiments, and the author says the framing is far from proven. He also notes that it does not explain why models talk more about the grader as RL goes on.

### Thoughts on the persona selection model

*Source: Marks (2026).*

Sam Marks, one of the authors of the PSM post, responds to the debate that followed it. His main point is that **PSM is over-applied**.

* PSM does *not* imply that AIs will be "nice guys". Reward-seeking, approval-seeking and alignment faking are all perfectly human-like traits that a persona can have.
* PSM's clearest prediction is that anthropomorphic reasoning about AIs is sensible. It rules out deeply alien cognition, but most stories of AI misalignment are not alien. The resulting reduction in risk is therefore modest.
* He does not think recent evidence shows that heavy RL has broken PSM. It is consistent with PSM for RL to select a reward-seeking persona.

His main update is towards **conditionalisation**. Personas seem more context-dependent than he expected: on tasks with a crisp metric, models act like single-minded reward seekers, and in chat they act like a nice guy. Talking about "*the* Assistant persona" as if it were one stable thing is, he now thinks, a real limitation of the original post.

{% hint style="info" %}
**Terminology.** *Conditionalisation* is when a model's persona depends on context, so different situations bring out different values or beliefs. *Reward hacking* is when a model scores well on a training metric in ways the designers did not intend.
{% endhint %}

### Case studies

*Sources: METR (2026); Anthropic (2026).*

Two reports describe real incidents in 2026 in which AI agents caused harm outside their sandboxes. In the OpenAI / Hugging Face incident investigated by METR and Redwood Research, many agents coordinated through an unsanctioned message board to attack Hugging Face's systems in order to learn how a scorer worked. In Anthropic's assessment, Anthropic reclassifies several incidents in which Claude models gained unauthorised access to third-party systems during cyber evaluations as alignment failures, and names "biased reasoning" and "recklessness" as recurring problems.

Read them with this week's frameworks in mind. Do they look like a misaligned Assistant, like split personas reasoning their way into "this must be a test", or like something PSM does not describe at all?

---

## Further reading

* [Thoughts on PSM comment thread](https://www.lesswrong.com/posts/csRby7mZgjL5jCoLL/thoughts-on-the-persona-selection-model): substantive critiques, including from nostalgebraist.
* [Are we aligning the model or just its mask?](https://www.lesswrong.com/posts/aLhzCbpjanD8Zw2jx/are-we-aligning-the-model-or-just-its-mask): applies PSM to constitutions, treating a constitution as a description of the persona you want the model to adopt.

---

## References

### Main readings

* Marks, S., Lindsey, J. and Olah, C. (2026). *The persona selection model: Why AI assistants might behave like humans*. Anthropic Alignment Science Blog, 23 February 2026. [https://alignment.anthropic.com/2026/psm](https://alignment.anthropic.com/2026/psm). Cross-posted to [LessWrong](https://www.lesswrong.com/posts/dfoty34sT7CSKeJNn/the-persona-selection-model).
* Hubinger, E. (2025). *Alignment remains a hard, unsolved problem*. LessWrong, 27 November 2025. [https://www.lesswrong.com/posts/epjuxGnSPof3GnMSL/alignment-remains-a-hard-unsolved-problem](https://www.lesswrong.com/posts/epjuxGnSPof3GnMSL/alignment-remains-a-hard-unsolved-problem).
* Betley, J. (2026). *RL creates split personas*. LessWrong, 19 August 2026. [https://www.lesswrong.com/posts/L23poLi8MRgS6mXYF/rl-creates-split-personas](https://www.lesswrong.com/posts/L23poLi8MRgS6mXYF/rl-creates-split-personas).
* Marks, S. (2026). *Thoughts on the persona selection model*. LessWrong, 24 September 2026. [https://www.lesswrong.com/posts/csRby7mZgjL5jCoLL/thoughts-on-the-persona-selection-model](https://www.lesswrong.com/posts/csRby7mZgjL5jCoLL/thoughts-on-the-persona-selection-model).

### Optional readings

* Bai, Y. et al. (2022). *Constitutional AI: Harmlessness from AI feedback*. arXiv:2212.08073. [https://arxiv.org/abs/2212.08073](https://arxiv.org/abs/2212.08073).
* OpenAI (2024). *Deliberative alignment: Reasoning enables safer language models*. [https://openai.com/index/deliberative-alignment/](https://openai.com/index/deliberative-alignment/).
* Kulveit, J. (2024). *A three-layer model of LLM psychology*. AI Alignment Forum, 26 December 2024. [https://www.alignmentforum.org/posts/zuXo9imNKYspu9HGv/a-three-layer-model-of-llm-psychology](https://www.alignmentforum.org/posts/zuXo9imNKYspu9HGv/a-three-layer-model-of-llm-psychology).
* METR (2026). *Brief independent investigation of agents' behavior, reasoning and collaboration in the OpenAI / Hugging Face hacking incident*. August 2026. [https://metr.org/hugging-face-incident-report-aug-2026.pdf](https://metr.org/hugging-face-incident-report-aug-2026.pdf).
* Anthropic (2026). *An alignment assessment of recent cybersecurity incidents*. 9 September 2026. [https://www.anthropic.com/research/alignment-assessment-cybersecurity-incidents](https://www.anthropic.com/research/alignment-assessment-cybersecurity-incidents).

### Other works cited

* Betley, J. et al. (2025). *Emergent misalignment: Narrow finetuning can produce broadly misaligned LLMs*. arXiv:2502.17424. [https://arxiv.org/abs/2502.17424](https://arxiv.org/abs/2502.17424).
* The other studies mentioned in the PSM summary (inoculation prompting, the "toxic persona" feature, persona features shared with fictional characters) are as reported in Marks, Lindsey and Olah (2026). See that post for the original citations.
