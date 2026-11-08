---
title: "The Art of Forgetting: From Borges to a Hypothesis About Intelligence"
description: "A small, fully controlled experiment on when a coarse description predicts better than the complete state, and when it doesn't."
image:
  path: /figures/blog_learning_curve.png
  alt: "Global recovery prediction error falls with more training examples; the supplied summary stays below the full state."
---

# The Art of Forgetting: From Borges to a Hypothesis About Intelligence

*By Giorgio Savastano, PhD · [Background and experience](https://www.linkedin.com/in/giorgiosavastano/) · [Publications](https://scholar.google.com/citations?user=loRr1d8AAAAJ&hl=en)*

You can remember every word of a conversation and still miss its point. That gap between having information and understanding it is what interests me here, especially when we talk about intelligence in machines.

Let me take you on a short journey through some ideas borrowed from literature, philosophy and physics, down to a small hypothesis we can actually test: when data and time are limited, can a machine learn to predict a system better from a useful summary than from every available detail?

*6 November 2026. Code, raw results and the research note are in [the repository]({{ site.github.repository_url }}). The README explains how to reproduce the experiments and figures; a one-minute demo is also available there.*

## 1. The man who remembered everything

In 1942, Borges wrote a story that has stuck with me for more than 20 years. The story is about a young man in Uruguay, Ireneo Funes, who falls from a horse and wakes with a perfect memory. What may sound at first like an incredible superpower is quickly revealed to be an almost diabolic problem. Funes's superhuman lossless memory, while it allows him to remember every single detail of every second of his life, every leaf he ever witnessed, also effectively blocks him from forgetting any useless detail, to the point that he is not able to generalize or create abstract concepts. Funes finds it hard to accept that the dog seen at 3:14 from the side has the same name as the dog seen at 3:15 from the front.

From this first story, I want you to take home the importance of forgetting, of compressing information down to useful abstractions.

## 2. Information is not understanding

The year before, Borges wrote another story, called *The Library of Babel*. This story has recently been mentioned in connection with the wave of AI-generated content flooding the internet. The story is about a library that contains all possible books of 410 pages, created by forming all possible combinations of a fixed set of characters. In that sense, this imaginary library contains every book that has been or will ever be written in those characters: longer books could be divided across several volumes. It contains all useful knowledge, from a cure for cancer to the origin of the universe. However, it also contains a huge amount of nonsense, contradictions and misleading claims. It contains proofs, absurd claims and everything in between, including arguments for everything and the opposite of everything. The fictional people living in this strange universe quickly abandon their original excitement at the thought that they could find any information they might need, and fall into a state of despair fed by the fact that all this information is effectively unusable and their search is pointless.

Here we are going to borrow the distinction between the existence of information and its accessibility.

## 3. Abstraction and compression in physics

There is a recent [video by 3Blue1Brown, *Reinventing Entropy*](https://www.youtube.com/watch?v=l6DKRf-fAAM) exploring the mathematics of compression as part of a series connecting it to how AI systems learn. The thought-provoking and attention-grabbing idea behind the series is that intelligence is compression. My idea is somewhat adjacent to this one, as finding abstractions can be seen as an information compression process: ignoring what is not useful and keeping only what is. We find examples of these useful compressions everywhere in the laws of physics that we use to describe reality and solve specific problems. My favourite example is in thermodynamics, where, instead of following the evolution of billions of molecules of gas, we can simply keep track of 3 numbers, T, P and V. For a fixed amount of a simple gas of known composition, these are enough to describe its macroscopic equilibrium state. For an ideal gas, the equation of state relates them, so only two are independent once the amount of gas is fixed.

So here we start seeing clearly an important caveat: these abstractions may be task dependent, meaning they may be good for describing specific problems and may not necessarily be enough for every situation.

## 4. What if intelligence works similarly?

Put the three stories together and a working hypothesis suggests itself. Perhaps one important component of intelligence is recognizing the level of description at which a complex system becomes easiest to understand. A system that does this, when it meets a new situation, figures out quickly which details matter for what it is trying to do, builds a description that keeps those and drops the rest, notices that the description it built resembles one it has used before, and abandons it when it stops working. Not compression for its own sake, but *abstraction*: compression that preserves the relationships that make the information useful. And, if the physics is any guide, the relationships worth preserving are often the ones emerging from the system.

## 5. From the idea to a test

To investigate this idea, we need to distinguish between finding a useful set of abstractions and using them. Here, I start with the second question: can a model learn from a useful set of abstractions without having to find them for itself?

The [information bottleneck method](https://arxiv.org/abs/physics/0004057) gives a precise way of thinking about one part of the problem: keep a compact description of an input while preserving information about a target. For example, if I want to predict how quickly a gas expands, the useful information will be different from what I need to follow one particular molecule.

There is another question behind the physics example. Some useful descriptions reflect collective behaviour produced by the system itself. This is the sense of emergence that interests me here. Pressure comes from many molecules interacting; it is not a property of one molecule. Therefore, some emergent properties of the system may be useful things to capture in an abstraction. However, as we will see, it always depends on the questions we are asking about the system.

I find related motivation in Michael Levin's work on development and regeneration, where he [discusses goal-directed behaviour in groups of cells](https://doi.org/10.3389/fnsys.2022.768201), and in [Cavagna, Parisi and colleagues' measurements of starling flocks](https://doi.org/10.1073/pnas.1005766107), where correlations in the birds' velocities extend over distances that grow with the flock. These are different kinds of evidence about collective behaviour. I am borrowing a question from them: when does looking at the group reveal something useful that is difficult to see by looking at its parts separately?

The connection to intelligence remains a hypothesis. [François Chollet](https://arxiv.org/abs/1911.01547) proposes thinking of intelligence in terms of how efficiently a system acquires new skills, taking account of its prior knowledge and experience. Finding a useful abstraction could be one mechanism behind that efficiency. Before testing whether a system can discover, reuse or adapt an abstraction, I wanted to start with a simpler question: if I supply a good abstraction, how much does it help?

## 6. A small experiment

For this I built a world where I can control the rules and repeat the same situation many times. Start with an array of 64 numbers, from 0 to 63, in sorted order. Scramble a chunk of it, sometimes two. Then let neighbouring numbers compare and swap if they are in the wrong order. One round, or sweep, checks the alternating pairs in two phases, so each comparison is local. Swaps sometimes fail, and one region has less reliable exchanges than the rest. After twelve rounds, how much disorder will be left?

Here disorder has a simple meaning: how far each number is from its correct position, averaged over the array. If every number is home, it is zero. The setup is inspired by the [self-sorting arrays studied by Taining Zhang, Adam Goldstein and Michael Levin](https://doi.org/10.1177/10597123241269740). My simulator is much simpler and is not a replication of their experiments or a model of biological intelligence.

[![A real 64-element sorting trajectory: an ordered array is disturbed, then local stochastic swaps reduce disorder. A close-up shows successful and failed swap attempts; the full state keeps individual values and exchange probabilities, while the supplied summary keeps three global and twenty-four regional numbers.](figures/sorting-experiment.svg)](figures/sorting-experiment.svg)

*One configuration, generated and evolved by the simulator. Each column is a position and its height shows the number there. The shaded interval has less reliable exchanges. The close-up shows the first half of one round; a full round checks both sets of alternating pairs. This is one possible recovery, not an average over futures. Open the diagram for a larger view.*

Because each swap attempt can succeed or fail probabilistically, the same starting state can have different futures. Therefore, even a predictor that knows the exact average remaining disorder after twelve rounds will make errors on individual runs because of this randomness. To estimate this error floor, I kept the initial ordering and swap probabilities fixed for each starting configuration in the test set, then simulated 32 independent futures. This allowed me to estimate how much the remaining disorder varies purely because of randomness. Combining that variation across the test set gives an estimated floor of about 0.19 positions per element, measured as root mean squared error, or RMSE.

I trained the same kind of machine-learning model, gradient-boosted decision trees, on three descriptions of the starting state:

1. The full state: all 64 values and the exchange reliabilities, stored as 64 numbers, including one padding entry. This gives 128 numbers in total.
2. A summary I designed: three global numbers describing average displacement, the fraction of neighbours out of order, and the overlap between displacement and unreliable exchanges. I also divided the array into eight regions, and for each of them I kept average displacement, the fraction of neighbours out of order, and average exchange unreliability. This gives 27 numbers.
3. Random combinations of the full state: 27 numbers again, but constructed without knowledge of the sorting task.

I also used the initial disorder alone as a one-number reference, and checked what happened when the full state and the designed summary were supplied together.

The second description contains quite a lot of my own knowledge. I already know the desired ordering, and I know that faults slow the recovery. I compute these summary numbers before training and hand them to the predictor.

Each training example pairs one starting configuration and its swap probabilities with the remaining disorder measured after twelve rounds. I repeated the comparison ten times, using a different random seed each time to generate fresh training, validation and test examples. Training budgets ranged from 250 to 5,000 examples. There were three model sizes at every budget and a fourth, larger one at 5,000 examples for the global task. Each description could select its own model size using a separate validation set. The test examples were kept out of that selection. The [methods](methods.md) give the full protocol.

## 7. What the results show

The short version is that my summary helped this learner predict global recovery, especially with fewer training examples. The full state contains everything needed to compute that summary; the difficulty is learning to use it with the data and model sizes available here. When I changed the question to a local prediction, the full state did better. So this is evidence that a supplied, task-relevant summary can help. It does not show that the learner can discover one for itself.

![Global-task test RMSE against training examples, with validation-selected model size and 95% confidence intervals across ten seeds](figures/blog_learning_curve.png)

In this figure, the orange line represents a predictor trained on the summary, and the blue line one trained on the full state. The green line uses random combinations, while the yellow line uses initial disorder alone. Moving to the right gives each predictor more training examples.

The carefully engineered summary beat the full state at every tested model size, especially with fewer training examples. With 250 examples, its error was 0.455 against 0.807 for the full state. However, the full state is catching up fast as we increase the number of examples. At 5,000 examples the errors were 0.332 and 0.401. The figure uses a model size selected separately for each description on validation data. These are averages over the ten seeds; the shaded regions show 95% confidence intervals.

What I take from this is that the summary helps most when the learner has access to fewer training examples. Since I compute the summary from the full state, it cannot contain extra information about the target. That does not guarantee that this particular predictor will eventually find the right summary, or that the summary I provided is the optimal one. The predictor could find a better one.

There was also an important practical difference in computing time. At 5,000 examples, a medium model with access to the summary reached an error of 0.366 in about 0.31 seconds, while a large model with the full state reached 0.401 in about 4.06 seconds. That is about one-thirteenth of the measured time, including feature extraction, fitting and test prediction on the same machine. However, this does not include the human effort and time needed to design the summary.

The random combinations did worse than both the summary and the full state at every tested budget and model size in this experiment, on average. At 250 examples, they were even worse than using initial disorder alone. Simply reducing the number of inputs did not reproduce the benefit. Adding the full state to the designed summary did not improve the average error either.

In the next experiment, I changed the number of regions used to divide the array. In the original experiment I used 8; here I varied the number of regions from 1 to 64, using the same configurations for each comparison.

![Global-task RMSE against regions, with an upper axis showing input numbers per example for the full summary and random combinations; 5,000 training examples and medium model size](figures/blog_granularity.png)

In this figure we see the results from a medium predictor model trained on 5,000 examples. Each line represents predictors trained on a different description of the same configurations: the orange line uses the full summary (3 global numbers + 3 numbers per region), the purple line uses only the regional numbers (3 numbers per region), and the green line uses random combinations of the full state, with the same number of inputs as the full summary. The horizontal blue and yellow lines represent the full state (128 numbers) and the 3 global numbers alone.

The upper axis shows how many input numbers each example contains for the full summary and the random combinations. With 8 regions, that means 27 numbers; the regional-only summary has 24 because it drops the 3 global numbers. The number of training examples stays fixed at 5,000.

One thing that is immediately clear is that most of the benefit was already present in the three global numbers. A predictor with access only to these 3 numbers (yellow horizontal line) already reached 0.405.

Interestingly, when I removed the global numbers and kept only regional statistics (purple line), four to eight regions did much better than sixty-four. The regional summary reaches its minimum at 8 regions, where it predicts better than the 3 global numbers alone. But as we increase the number of regions further, the error actually increases. There seems to be a trade-off between losing useful detail and giving the learner too many quantities to fit.

For the full summary (orange), eight regions also gave the lowest average error, 0.366, with 5,000 examples and the medium model. Two through sixty-four regions were all within about 6% of that value. With only 250 examples, the differences were small, around 0.45–0.47, and a single region was often selected on validation data. More training examples and larger models generally moved the preferred resolution towards finer regions.

Then I changed the objective of the predictor: it now has to predict which number will be at the centre after one round of sorting.

![Global and local prediction tasks with validation-selected model sizes; the ranking of the full state and summary reverses](figures/blog_task_reversal.png)

In this figure, the left panel repeats the global question about remaining disorder, while the right panel shows the new local question. The blue and orange lines represent the full state and the summary. On the right, their ranking is reversed: the predictor with access to the full state makes smaller errors.

The full state now beat the summary in all ten seeds, at every tested budget and model size. For example, at 5,000 examples with validation-selected model sizes, its error was 3.39 against 4.76. These errors concern element values, rather than the average displacement used in the global task. Finer regional descriptions generally helped, but even the sixty-four-region description reached only 4.15. It still omits explicit element identities and displacement signs.

This brings us back to the caveat in the physics example. The summary was built for a question about the whole array's recovery. It leaves out detail that matters for a question about one particular position. Better predictive learning with that summary does not by itself show understanding, or intelligence.

The [Experiment 1 results]({{ site.github.repository_url }}/blob/main/results/experiment1/README.md) and [Experiment 2 results]({{ site.github.repository_url }}/blob/main/results/experiment2/README.md) include the full comparisons and uncertainty estimates.

## 8. Does the summary capture something about the system?

There is still a question left from the discussion of emergence. Did I just find convenient inputs for decision trees, or do these collective variables capture regularities in the array's future? To investigate this, I ran a separate test without training a learner.

For each seed, I took 10,000 configurations after two rounds of recovery. I matched arrays with nearly the same summary but different microscopic arrangements, then simulated sixteen independent futures from each configuration. If the summary captures the relevant collective behaviour, arrays that agree on it should have similar average futures, despite their different arrangements.

![Estimated predictive closure against sorting rounds ahead for four descriptions; the full-state estimate is biased downward by imperfect matching in high dimensions](figures/blog_closure.png)

In this figure, the orange line uses the 3 global numbers, the grey line uses initial disorder alone, the green line uses 3 random combinations, and the blue line uses the full state. Moving to the right means looking further into the future. Higher on the vertical axis means closer agreement between the average futures of configurations matched on each description.

To compare them, I used a closure index. A value of 1 means that configurations matched on the description have the same average future. Each configuration's average future is estimated from its sixteen independent runs, and the calculation corrects for the noise in that estimate.

Twelve rounds ahead, the 3 global numbers (orange line) reached a closure index of 0.918, with a standard deviation of 0.004 across seeds. In this estimate, about 8% of the variance in average remaining disorder is still unexplained by matching on those numbers. The 3 random combinations (green line) reached 0.266, leaving about 73% unexplained. Initial disorder alone (grey line) reached 0.572, leaving about 43% unexplained. The grey line starts close to 1 after one round, but falls as we look further ahead. That is consistent with the idea that where the disorder sits relative to the faults matters for longer-term recovery.

The blue line needs some care. It appears lower than the orange line even though the full state contains all the information needed to compute the summary. Here, the limitation is in the matching procedure: matching on 128 numbers is much harder than matching on 3, and even the nearest available configurations can still be quite different. This lowers the estimated closure of the full state. As I increased the number of configurations, the estimate based on 3 numbers changed little, while estimates based on more numbers were still improving.

What I take from this is that the 3 supplied global numbers capture much of the average recovery behaviour in this system, for the configurations and sorting rounds tested here. This is a test of average futures; it does not show that the summaries capture the full range of possible outcomes.

There are limits to this test. I compared nearby summaries rather than exactly identical ones, and checked only this distribution of configurations and the next twelve rounds. Using one previous round's summary to refine the matches made a small improvement, rather than none. That does not establish that the present summary contains everything useful from the past, which a Markov description would require. The [Experiment 2b results]({{ site.github.repository_url }}/blob/main/results/experiment2b/README.md) give the estimator and the refinement tests. There were no interventions to test causal emergence, and the variables were supplied throughout.

The next question concerns the learner. For discovery, I would like to see whether a small network can build a useful representation from the raw state, without being given my features, and whether learning it costs more than it saves in later predictions. In fact, finding a useful description may itself be much harder than making predictions once we have it. So I would like to understand whether a system can discover that description cheaply enough, or reuse it often enough, for the initial effort to pay off.

For transfer, I would ask whether that representation helps in a related sorting system with different rules or array sizes. Dedre Gentner's [structure-mapping theory of analogy](https://doi.org/10.1207/s15516709cog0702_3) is a useful reference here: what might transfer is a relationship such as disorder overlapping unreliable exchanges, rather than the identity of a particular position. That is a proposed test, not something the ten random seeds demonstrate.

For adaptation, I would change the task from global recovery to the central element and see whether the system notices that its old summary is inadequate and changes it efficiently. These discovery, transfer and adaptation experiments have not been run. I would also like to compare the supplied summary with a model designed to use neighbouring positions in the raw array. So far we have tested only one learner family. The [research vision](research-vision.md), [next experiments](next-experiments.md) and [research note]({{ site.github.repository_url }}/blob/main/docs/manuscript-draft.md) describe these questions in more detail.

## 9. The art of forgetting

The local prediction result makes me cautious about turning Funes into a simple argument for forgetting. Sometimes the detail is what the question needs. The ability I am interested in is choosing what to keep, and recognizing when that choice needs to change.

Ted Chiang explores a related problem in *The Truth of Fact, the Truth of Feeling*. A technology makes recorded experiences searchable, and the narrator finds that a memory important to his understanding of himself is wrong. I read this as another way of approaching the distinction between a record and what we make of it. A better record can correct us; making sense of it remains a separate task.

This is also why I find the physics example useful. We can work with a macroscopic description while keeping access to finer detail when the question requires it. Perhaps part of intelligence involves learning to make that choice, recognizing a useful relationship in a new setting, and revising the description when it fails.

For now, the result is smaller than that hypothesis. In this sorting system, a summary designed with knowledge of the task helped one kind of learner predict global recovery with less data and computation. The same summary worked worse for a local question, and a separate test found approximate closure for the measured global futures. Whether a system can discover such descriptions for itself, transfer them and adapt them is what I want to investigate next.

---

*To reproduce the quick experiments, run `uv sync`, then `uv run macro-experiment1 --quick`, `uv run macro-experiment2 --quick` and `uv run macro-experiment2b --quick`. These use two seeds; they do not reproduce the full ten-seed results quoted here. The [README]({{ site.github.repository_url }}#quick-start) gives the full commands, timing context and analysis step for rebuilding the tables and figures. The [research note]({{ site.github.repository_url }}/blob/main/docs/manuscript-draft.md) contains the detailed results and limitations.*

Sources and further reading:

- Literary sources: Borges, “Funes, His Memory” (1942) and “The Library of Babel” (1941), in *Collected Fictions* (Penguin, 1998); Ted Chiang, “The Truth of Fact, the Truth of Feeling,” in *Exhalation* (Knopf, 2019). These are starting points for interpretation, not scientific evidence about memory or intelligence.
- Compression: [3Blue1Brown, *Reinventing Entropy*, Part 1 of the compression and intelligence series](https://www.youtube.com/watch?v=l6DKRf-fAAM). For the mathematical connection between predictive models and lossless compression, see [Delétang et al., *Language Modeling Is Compression*](https://arxiv.org/abs/2309.10668).
- Learning and abstraction: [Tishby, Pereira & Bialek, *The information bottleneck method*](https://arxiv.org/abs/physics/0004057); [Chollet, *On the measure of intelligence*](https://arxiv.org/abs/1911.01547); [Gentner, *Structure-mapping* (1983)](https://doi.org/10.1207/s15516709cog0702_3); [Abel, *A theory of abstraction in reinforcement learning*](https://arxiv.org/abs/2203.00397); [Ho et al., *People construct simplified mental representations to plan* (2022)](https://doi.org/10.1038/s41586-022-04743-9).
- Collective behaviour: [Zhang, Goldstein & Levin (2025)](https://doi.org/10.1177/10597123241269740); [Cavagna et al., *Scale-free correlations in starling flocks* (2010)](https://doi.org/10.1073/pnas.1005766107); [Levin, *Technological approach to mind everywhere* (2022)](https://doi.org/10.3389/fnsys.2022.768201).
- Coarse descriptions and prediction: [Shalizi & Moore, *What is a macrostate?*](https://arxiv.org/abs/cond-mat/0303625); [Israeli & Goldenfeld, *Coarse-graining of cellular automata, emergence, and the predictability of complex systems* (2006)](https://doi.org/10.1103/PhysRevE.73.026203); [Crutchfield, *Between order and chaos* (2012)](https://doi.org/10.1038/nphys2190). On the separate debate about causal emergence: [Hoel (2017)](https://doi.org/10.3390/e19050188) and [Eberhardt & Lee (2022)](https://doi.org/10.3390/philosophies7020030).

For a broader philosophical motivation, [Demis Hassabis describes his conjecture that information is more fundamental than matter and energy](https://www.amacad.org/publication/daedalus/ai-ultimate-tool-science-conversation-demis-hassabis). Our sorting experiments do not test that conjecture.

---

## About the author

I’m a remote sensing scientist with a PhD in geodesy and geomatics from Sapienza University of Rome. At Spire, I work on turning satellite measurements into estimates of physical quantities such as soil moisture; before joining Spire, I held a NASA Postdoctoral Program fellowship at the Jet Propulsion Laboratory.
