# Week 4: Principles of Learning

**Mentor:** Nicholas Chen
**Date:** 15 Oct 2026

---

## What does it take to learn?

Deep learning works. We can train networks that write code, prove theorems and hold conversations. But we cannot yet say, in a precise way, *why* it works. This week steps back from deep learning and asks a more basic question: what does any learning machine need in order to learn at all?

We organise the answer around **three barriers**: approximation, generalisation and optimisation. We then look at what classical theory says about each one: the bias–variance tradeoff, the No Free Lunch theorems, Occam's razor and Kolmogorov complexity, and finally Solomonoff induction, an idealised learner that gets past two of the three barriers. Next week (Week 5) asks how much of this picture actually explains deep learning.

### What you'll learn

By the end of the week, you will be able to:

* Explain the debate between "machine learning is alchemy" and "the bitter lesson", and say what each side actually claims.
* State the three barriers to learning (approximation, generalisation, optimisation) and explain how they differ.
* Explain the bias–variance tradeoff, and why a bigger hypothesis class helps with one barrier but hurts another.
* State the No Free Lunch theorem in one sentence, and explain why it does *not* mean that all learners are equally good.
* Explain Occam's razor and Kolmogorov complexity, and what it means to call simplicity a *bet* about the world.
* Describe Solomonoff induction at a high level, and explain which barriers it gets past and which one it does not.

### Roadmap

The week runs as two 1-hour discussion sessions. Each unit follows the same pattern.

| Time    | Session                | Breakdown                                       |
| ------- | ---------------------- | ----------------------------------------------- |
| 20 min  | Individual reading     | Read the main reading for the unit              |
| 20 min  | Small group discussion | Work through the discussion questions           |
| 20 min  | Combined discussion    | Whole-group discussion led by the head TA       |

{% hint style="info" %}
**Reading levels.** Each reading and question has a level tag. **100** is introductory and needs no background. **200** needs some maths or ML background. **300** is advanced and may need background that this course does not cover. Ranges such as 100–200 mean the reading works at either level, depending on how deep you go.
{% endhint %}

---

## Preamble: engineering before theory

Most of the progress in AI over the last decade came from scaling up simple methods: more data, more compute, bigger networks. It did not come from a deep understanding of why those methods work. This is not unusual. The steam engine came before thermodynamics, and powered flight came before a full theory of aerodynamics.

So why should AI safety care about theory? One answer is that safety needs *predictions*: what will a model learn, how will it generalise to new situations, and when will a training run produce something we did not intend? A capability can be found by trial and error. A guarantee usually cannot. We return to this question at the end of this week and again in Week 5.

---

## Reading guide

This section summarises the main readings. It is a companion to the readings, not a replacement for them. Full citations are in the [References](#references) at the end of the chapter.

### Unit 0: General background

*Sources: Sutton (2019); Rahimi and Recht (2017). Level 100.*

**The bitter lesson.** Sutton looks back at 70 years of AI research and draws one lesson. Methods that build in human knowledge (hand-written chess heuristics, hand-designed speech features, hand-crafted vision pipelines) win in the short term. Over the long term they always lose to *general methods that scale with compute*, mainly search and learning. The lesson is "bitter" because researchers keep relearning it: building in what we know feels productive, but it stops progress later.

**Machine learning as alchemy.** In his Test-of-Time award talk at NeurIPS 2017, Ali Rahimi argued that machine learning had become a kind of *alchemy*. Alchemy produced real results, such as metallurgy and medicine, but without a theory of why. Rahimi's worry was that practitioners use tricks (for example, particular optimisers or normalisation schemes) that work for reasons no one understands, and that this makes the field fragile. He asked for more rigour: simple experiments, simple theorems, and a real understanding of the tools. Yann LeCun replied that this undervalues empirical progress, and that engineering has often come before theory. Their exchange in the comment thread is worth a skim.

Watch **7:20–8:10**, then **11:00 onwards** of the talk.

{% hint style="success" %}
**Discussion**

1. *(100–200)* Rahimi says machine learning is alchemy. Sutton says that general methods plus compute beat human understanding every time. Are they disagreeing? If so, about what, exactly?
2. *(100–200)* From an AI safety perspective, why is it worth trying to figure out scientifically how deep learning works? Why would we need scientific understanding for safety, if such understanding seems to have been unnecessary for capabilities? (We revisit this at the end of the week and in Week 5.)
3. *(100–200)* Engineering has often come before theory (the steam engine before thermodynamics, flight before aerodynamics). What, if anything, would make AI the exception?
{% endhint %}

### Unit 1: Learning machines and the three barriers

*Source: Iliad B1 slides, pages 2, 11, 13 and 16. Level 100–200.*

The slides open with a quote from Shane Legg that compares learning to detective work. A detective has a *list of suspects* (possible explanations of the evidence) and must pick the right one. A learning machine does the same thing: it has a set of candidate explanations, called the **hypothesis class**, and uses data to choose between them.

For any learning machine to work, three things must go right.

1. **Approximation.** Is a good explanation *on the list at all*? A linear model cannot fit data that is not linear, however much data you give it. If the true explanation is not a suspect, the detective cannot name it.
2. **Generalisation.** Do we have *enough evidence to recognise* the good explanation when we see it? With a long list of suspects and little evidence, many suspects fit the facts. The one we pick may fit the data we have but fail on new data.
3. **Optimisation.** Suppose a good explanation is on the list (approximation is fine) and we have enough evidence to recognise it (generalisation is fine). Can we *find it quickly*? Checking every suspect one by one is usually far too slow. We need a search procedure that gets to the right answer in a reasonable time.

Slide 11 also raises the old **problem of induction**. If I have seen the sun rise on 100 days, what justifies my prediction that it will rise on day 101? Nothing in the data alone does. Every learner has to assume that the world has some regularity, so that the past says something about the future.

{% hint style="info" %}
**Terminology.** A *hypothesis class* is the set of candidate explanations (functions, models or programs) that a learner can choose from. In the detective analogy, it is the list of suspects.
{% endhint %}

{% hint style="success" %}
**Discussion**

1. Restate each of the three barriers as a question that the detective in Legg's quote would ask about their list of suspects.
2. Slide 11 asks: "If I have seen the sun rise 100 days, what justifies predicting day 101?" What assumption about the world is the slide saying we must make? What would a world without that assumption look like?
3. Slide 13 says that a linear model can only learn linear data. One proposed fix is to put *every possible computer program* on the list of candidate explanations. In terms of the three barriers, does this make anything worse? If so, what, and why?
{% endhint %}

### Unit 2: Generalisation and the bias–variance tradeoff

*Source: Fortmann-Roe (2012). Level 200–300. You can skip section 4.2 onwards.*

Fortmann-Roe splits the error of a model into two parts.

* **Bias** is error that comes from the model being *systematically wrong*. Even if you trained it on many different datasets and averaged the results, the average would still miss the truth. High bias usually means the hypothesis class is too simple to capture the pattern. This is closely linked to the approximation barrier.
* **Variance** is error that comes from the model being *too sensitive to the particular dataset*. Train it on a different sample and you get a very different answer. High variance usually means the hypothesis class is so flexible that it fits noise. This is closely linked to the generalisation barrier.

The **bulls-eye diagram** shows this as four targets. Each dot is one model trained on one dataset. Bias is how far the centre of the cluster is from the bulls-eye. Variance is how spread out the cluster is.

The **phone-book poll** example makes the distinction concrete. A poll that samples voters only from a phone book will be biased: people without phone listings are left out, whatever the sample size. A poll with a very small sample has high variance: rerun it and the result changes a lot.

The classical lesson is the **tradeoff**: making the model more flexible lowers bias but raises variance, so there is a sweet spot of model complexity. Keep this lesson in mind. In Week 5 we will see that very large neural networks seem to break it.

{% hint style="success" %}
**Discussion**

1. Explain the bias and variance bulls-eye picture in your own words, without using the words "bias" or "variance". Which of the two does a bigger hypothesis class (a longer suspect list) make worse, and why?
2. In the phone-book poll, would adding more respondents fix the bias, the variance, or both? Explain.
{% endhint %}

### Unit 3: No Free Lunch

*Source: Garriga-Alonso (2022). Level 200. You can skim the notation. Only quickly skim "Some learning problems are genuinely impossible". You can skip the PAC-Bayes section.*

**The theorem.** Averaged over *all possible* problems, every learning algorithm performs the same. Any learner that does better than chance on some problems must do worse than chance on others. In particular, no learner beats random guessing when you average over every possible way the world could be.

**What it does not say.** It does not say that no method is better than another *in practice*. The average is taken over all possible worlds with equal weight, and most of those worlds are pure noise, with no structure to learn. Our world is not like that. The theorem tells us that a learner can only succeed by being suited to the problems it actually meets.

**The razor.** The post's main point is that every learner that works is making a *bet* about the world. It has an **inductive bias**: a built-in preference for some explanations over others. A learner wins when its bias matches the structure of the problems it meets. The question is never "which learner is best?" but "which learner is best *for problems like ours*?"

{% hint style="info" %}
**Terminology.** An *inductive bias* is a learner's built-in preference for some hypotheses over others, before it sees any data. Note that this is a different sense of "bias" from the statistical bias in Unit 2.
{% endhint %}

{% hint style="success" %}
**Discussion**

1. Explain the No Free Lunch theorem in one simple sentence. Does it imply that no method is better than another? Explain your answer.
2. Every learner that works is betting on something about the world. What is a learner that always prefers the simpler explanation betting on? What is a weather forecaster betting on?
{% endhint %}

### Unit 4: Occam's razor and Kolmogorov complexity

*Sources: Altair (2012), sections 2 "Induction" and 3 "Occam's razor"; Iliad B1 slides, page 22. Level 100–200.*

**Induction.** Consider a number sequence such as 1, 2, 4, 8, ... What comes next? "16" (powers of two) is a natural answer, but infinitely many rules fit the first four numbers and give a different fifth one. Data alone never settles the question. We need some principle for choosing between the rules that fit.

**Occam's razor** is that principle: among the explanations that fit the data, prefer the *simplest* one. This is an intuition. To use it, we need a way to measure simplicity.

**Kolmogorov complexity** gives that measure. The Kolmogorov complexity of a string is *the length of the shortest computer program that outputs it*. A string with a pattern has a short program ("print 01 five hundred times"). A random string has no shorter description than the string itself ("print 0110100..."). One useful fact is that the choice of programming language changes the length only by a constant. So the measure is, up to that constant, a property of the string itself.

**TL;DR.** Occam's razor is the intuition (prefer simpler). Kolmogorov complexity is the measure (the length of the shortest program).

{% hint style="info" %}
**Reading tip.** The readings and slides talk about "Turing machines" and "UTM code". For this week, read both of these as "computer program". The details do not matter here.
{% endhint %}

{% hint style="success" %}
**Discussion**

1. Without any principle for choosing between explanations, could more data ever settle the number sequence problem?
2. Rank these by Kolmogorov complexity: the first 1000 digits of pi, 1000 coin flips, and the string "01" repeated 500 times.
3. In the language of the No Free Lunch unit, what lunch does Occam's razor pay for? Is it a fact about the world, or a bet we make? What would it mean for it to be wrong?
{% endhint %}

### Unit 5: Solomonoff induction

*Sources: Altair (2012), the rest of the post; Iliad B1 slides, pages 23–25. Level 100–200. Only the basic setup is needed. You do not need the low-level details.*

If the post is too long, focus on chapter 7 "All algorithms" and chapter 8 "Solomonoff's lightsaber".

**The setup.** Solomonoff induction combines Occam's razor with Bayesian reasoning, taken to the extreme.

1. Put **every computer program** on the list of suspects.
2. Give each program a prior weight that falls off exponentially with its length: roughly 2<sup>−length</sup>. Short programs get much more weight than long ones. This is Occam's razor in exact form.
3. When data arrives, throw out every program whose output does not match it. Keep the rest with their weights.
4. To predict what comes next, let the surviving programs vote, weighted by their prior.

**Why it matters.** Solomonoff induction is a kind of ideal learner. Slide 24 claims that it gets past two of the three barriers.

* *Approximation:* every computable pattern is on the list, so a good explanation is always there.
* *Generalisation:* its total prediction error is bounded. Slide 25 states this formally: for any computable predictor *q*, Solomonoff induction does almost as well as *q*, with an extra cost of about *K(q)*. Here "computable predictor *q*" means any prediction method you could write as a program, and *K(q)* is the length of the shortest program for it. So if a simple predictor exists, Solomonoff induction quickly learns to do as well as it.

Slides 23 and 25 only need a rough reading.

**The catch.** Solomonoff induction is *uncomputable*. You cannot run every program, because some never halt. Even bounded versions are far too slow. It solves approximation and generalisation by giving up entirely on optimisation.

{% hint style="info" %}
**Optional readings.** For more depth, see Yudkowsky's *Semitechnical Introductory Dialogue on Solomonoff Induction* (level 100–200), or sections 2–2.8 of Shane Legg's PhD thesis, *Machine Super Intelligence* (level 300).
{% endhint %}

{% hint style="success" %}
**Discussion**

1. *(100–200)* Intuitively, what is Solomonoff induction, and why do we care about it? (We revisit this in Week 5.)
2. *(200)* Slide 24 claims that Solomonoff induction gets past both the approximation and the generalisation barriers. Explain each claim in the suspect-list framing.
{% endhint %}

### Unit 6: The optimisation barrier and outlook

*Source: Iliad B1 slides, pages 27 and 31. A quick read is enough.*

Solomonoff induction shows that approximation and generalisation can, in principle, be solved together. The price is optimisation: the ideal learner cannot actually be run. Real learners must trade some of that ideal away to get something they can compute.

This sets up the central puzzle for next week. Deep learning uses huge hypothesis classes, which should cause generalisation problems, and finds good solutions with simple local search (gradient descent), which should get stuck. Yet it works. Slide 31 refers to "A.3"; in this course, that material is Week 5.

{% hint style="success" %}
**Discussion (optional)**

1. *(100)* Revisit the question from Unit 0: from an AI safety perspective, why is it worth trying to figure out scientifically how deep learning works? Has your answer changed?
2. *(100)* Researchers have called for a real understanding of deep learning for a long time, and we still do not have one. What factors might explain this? Which reasons are about people and incentives, and which are about the problem itself? What would have to change?
{% endhint %}

---

## Further reading

* The Iliad reading list for this module has more readings that are not covered here. See the [Iliad Intensive curriculum](https://iliad-intensive.org).
* [Iliad Solomonoff Induction module (D.3.1)](https://iliad-intensive.org/agency/solomonoff-induction/): a technical worksheet that proves the main prediction-error bound.
* [A technical introduction to Solomonoff induction without K](https://www.lesswrong.com/posts/HSDumToH57nSRdLST/a-technical-introduction-to-solomonoff-induction-without-k): a more mathematical treatment.

---

## Attribution

This chapter is adapted from the ILIAD Intensive curriculum ([iliad-intensive.org](https://iliad-intensive.org)), including the B1 slides and reading list, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: the material was reorganised into Odysea units, reading summaries and some discussion questions were written by the Odysea 2026 Technical Team, and level tags and reading guidance were added.

---

## References

### Main readings

* Sutton, R. (2019). *The bitter lesson*. 13 March 2019. [http://www.incompleteideas.net/IncIdeas/BitterLesson.html](http://www.incompleteideas.net/IncIdeas/BitterLesson.html).
* Rahimi, A. and Recht, B. (2017). *Test-of-Time award talk*. Conference on Neural Information Processing Systems (NeurIPS) 2017. [https://www.youtube.com/watch?v=ORHFOnaEzPc](https://www.youtube.com/watch?v=ORHFOnaEzPc). See also LeCun's reply and the debate in the comments: [https://www.facebook.com/yann.lecun/posts/10154938130592143](https://www.facebook.com/yann.lecun/posts/10154938130592143).
* ILIAD. *B1: Principles of learning* (slides). [https://drive.google.com/file/d/1xCeATLFViOeCttivpMW7tNyVtIpPj5VQ/view?usp=drive_link](https://drive.google.com/file/d/1xCeATLFViOeCttivpMW7tNyVtIpPj5VQ/view?usp=drive_link).
* Fortmann-Roe, S. (2012). *Understanding the bias-variance tradeoff*. [https://scott.fortmann-roe.com/docs/BiasVariance.html](https://scott.fortmann-roe.com/docs/BiasVariance.html).
* Garriga-Alonso, A. (2022). *The No Free Lunch theorems and their Razor*. LessWrong, 24 May 2022. [https://www.lesswrong.com/posts/sdrCBWpqyvNBJSZH5/the-no-free-lunch-theorems-and-their-razor](https://www.lesswrong.com/posts/sdrCBWpqyvNBJSZH5/the-no-free-lunch-theorems-and-their-razor).
* Altair, A. (2012). *An intuitive explanation of Solomonoff induction*. LessWrong. [https://www.lesswrong.com/posts/Kyc5dFDzBg4WccrbK/an-intuitive-explanation-of-solomonoff-induction](https://www.lesswrong.com/posts/Kyc5dFDzBg4WccrbK/an-intuitive-explanation-of-solomonoff-induction).

### Optional readings

* Yudkowsky, E. *A semitechnical introductory dialogue on Solomonoff induction*. LessWrong. <!-- TODO: add verified link; the previously known LessWrong URL returns 404. -->
* Legg, S. (2008). *Machine super intelligence*. PhD thesis, University of Lugano. Sections 2–2.8. [https://www.vetta.org/documents/Machine_Super_Intelligence.pdf](https://www.vetta.org/documents/Machine_Super_Intelligence.pdf).
