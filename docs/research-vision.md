# Research vision: abstraction, emergence and resource-bounded intelligence

*Status document, 6 November 2026. This is a statement of hypotheses and of what has and has not been tested. It is not a claim of results.*

## The unifying hypothesis

> Intelligence may involve efficiently discovering the emergent structures and relationships that make complex systems predictable, constructing useful abstractions from those structures, and transferring and adapting them in novel situations.

This extends the project's working definition (intelligence as the capacity to discover, construct, reuse and adapt task-relevant abstractions, preserving essential relationships while discarding unnecessary detail) with a claim about *where* useful abstractions come from: often, though not necessarily, from collective regularities that are already present in the system being observed. We treat it as a promising research hypothesis, not as an explanation of intelligence, and we keep two things separate throughout.

- **Emergence** is about the system: local interactions produce macroscopic properties and effective dynamics that are meaningfully described at a larger scale.
- **Abstraction** is about the observer or model: a representation that keeps some distinctions and drops others.

An emergent property can exist without being recognized. An abstraction can be useful without corresponding to any recognized emergent law. A compressed representation need not reveal emergent structure. The defensible version of the link is: **some emergent macroscopic properties are particularly useful targets for abstraction because they capture stable, predictive relationships among microscopic variables.** Not every emergent phenomenon admits a compact, efficiently computable predictive description (chaos, computational irreducibility), and not every useful abstraction is an autonomous macroscopic law.

## The conceptual progression

1. Complex systems contain enormous amounts of microscopic information.
2. Collective interactions can generate macroscopic regularities.
3. Some of those regularities can be represented with a comparatively small set of variables.
4. Such representations can make certain predictions substantially easier for a bounded learner.
5. An intelligent system may benefit from discovering these useful levels of description.
6. More sophisticated intelligence may involve transferring and adapting them across situations.

Steps 1–3 are well established in physics for particular systems (fluids, critical phenomena, coarse-grained cellular automata), and Experiment 2b now establishes step 3 for the sorting simulator: three supplied collective variables approximately close over the global recovery dynamics. Step 4 is what Experiments 1–2 measure, for supplied representations, in one system. Steps 5–6 are hypotheses.

## Four stages

| Stage | Question | Status |
| :-- | :-- | :-- |
| **1. Useful abstraction** | Can supplied macroscopic representations improve prediction under resource constraints? | **Established in one system** (Experiments 1–2, 10 seeds): yes, by 44% at 250 examples falling to 17% at 5,000 with validation-selected capacity; the benefit comes from the content of the representation, not its size; it reverses for a local target. See `results/experiment1/README.md`, `results/experiment2/README.md`. |
| **2. Emergent predictive structure** | Do the useful macroscopic variables capture approximately autonomous, stable dynamics of the system, as opposed to merely convenient coordinates for one learner? | **Established in one system, for one target** (Experiment 2b, 10 seeds, learner-free): configurations matched on the three global summaries have futures 12 sweeps ahead whose conditional means agree to within 8% of their variance (closure 0.918 ± 0.004), vs. 0.27 for a matched random projection and 0.57 for the disorder scalar; microscopic detail recovers ≈ 1 point more, the previous macrostate essentially nothing (approximately Markov). Not tested: other distributions, longer horizons, the local target, other systems. See `results/experiment2b/README.md`. |
| **3. Autonomous abstraction discovery** | Can a learning system find these variables without being told which features matter, and at what cost? | **Hypothesis; not started.** Experiment 3 (manuscript §11.2): a 3-dimensional learned bottleneck must reach the global-only error (0.405) and expose the displacement × unreliability interaction to a linear probe. |
| **4. Transfer and adaptation** | Can discovered representations be reused across unfamiliar but structurally related systems, and can the system recognize when they must change? | **Hypothesis; not started.** Experiments 4 and 5 (manuscript §11.3–11.4), with surface-feature, negative-transfer and non-adapting controls. Reuse across seeds of one simulator does not count. |

The stages are ordered by what each presupposes, not as a causal chain: stage 2 could fail (the variables might be predictive without being autonomous) while stage 3 succeeds (a learner might still discover them), and emergence at stage 2 would not by itself imply anything about intelligence.

## Alternative explanations we must keep live

- The macroscopic features work because they encode substantial prior knowledge of the simulator, including the goal configuration; disorder and displacement are near-obvious features for a sorting objective.
- Their advantage reflects the inductive bias of tree ensembles (axis-aligned splits on well-chosen coordinates) rather than any property of the dynamics; a locality-aware architecture on the raw state is the untested competitor.
- A variable can be predictive of a scalar target without being dynamically autonomous. (For the three global summaries and the global target, Experiment 2b shows they are both; the caveat stands for any other variable or target.)
- The useful level depends on the task and the horizon.
- Some systems admit no useful low-dimensional coarse-grained dynamics.
- Approximate predictive sufficiency is not causal emergence, which rests on interventionist assumptions that are themselves contested.
- Abstraction may be one of several interacting mechanisms of intelligence (search, memory, exploration, planning, feedback-driven learning), not the central one.

## Relevant frameworks

Information bottleneck (Tishby et al.); computational mechanics and causal states (Crutchfield; Shalizi & Crutchfield; Shalizi & Moore's Markov macrostates); lumpability and informational closure (Kemeny & Snell; Pfante et al.); Granger-causal autonomy (Seth); coarse-graining and effective theories (Kadanoff; Israeli & Goldenfeld; Koch-Janusz & Ringel); causal emergence and its critiques (Hoel; Rosas et al.; Dewhurst; Eberhardt & Lee); state abstraction in reinforcement learning (Li et al.; Abel); human task representations (Ho et al.); skill-acquisition efficiency (Chollet); structure-mapping (Gentner); collective behaviour and basal cognition (Cavagna, Parisi et al.; Levin; Zhang et al.). Full citations are in `paper/references.bib`.

## What would make us abandon the hypothesis

- Stage 2: the macroscopic variables show large excess future variance across microstates with the same macrostate, no smaller than matched random projections. Then they are coordinates, not a level of description. *Tested for the global target in the sorting simulator: not falsified (excess 8% vs. 73% for the matched random projection). The same test would falsify stage 2 for the local target, and could still falsify it for other distributions, horizons or systems.*
- Stage 3: no small learner recovers variables of comparable usefulness at any budget, or the discovery cost exceeds the savings.
- Stage 4: transferred representations do not beat learning from scratch once surface overlap is removed and pre-training cost is charged, or adaptation is never cheaper than relearning.

Any of these would narrow the programme to its current, defensible core: supplied task-relevant representations improve finite-resource prediction, for the task they were built for.
