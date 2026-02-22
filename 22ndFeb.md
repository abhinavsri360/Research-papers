# Research Paper Summaries

---

## 1. Evaluation and Benchmarking of LLM Agents: A Survey

- **Authors:** Mahmoud Mohammadi, Yipeng Li, Jane Lo, Wendy Yip (SAP Labs)
- **Venue:** KDD 2025
- **Link:** [https://arxiv.org/abs/2507.21504](https://arxiv.org/abs/2507.21504)

### Summary

A comprehensive survey that organizes the fragmented landscape of how we evaluate LLM-based agents (not just LLMs, but agents that reason, plan, use tools, and act in environments). The core contribution is a **two-dimensional taxonomy** structured along evaluation objectives (what to evaluate) and evaluation process (how to evaluate).

### 2D Taxonomy

#### Dimension 1: Evaluation Objectives (the WHAT)

**a) Agent Behavior** (black-box, user-facing)
- Task Completion — did it finish the job? Metrics: Success Rate, pass@k, pass^k, Progress Rate
- Output Quality — coherence, relevance, fluency. Overlaps with RAG evaluation metrics (factual correctness, response relevance)
- Latency & Cost — TTFT for streaming, end-to-end latency for async. Cost estimated via token counts.

**b) Agent Capabilities** (white-box, process-oriented)
- Tool Use — can it decide *whether* to call a tool, *which* tool, with *correct parameters*? Metrics: Invocation Accuracy, Tool Selection Accuracy, Parameter Name F1, execution-based eval
- Planning & Reasoning — can it sequence multiple tools correctly? Graph-based metrics (Node F1, Edge F1). ReAct paradigm (reason → act → observe → repeat). Progress Rate measures trajectory advancement.
- Memory & Context Retention — recall over long conversations (tested up to 600+ turns). Categorized by Memory Span and Memory Forms.
- Multi-Agent Collaboration — info sharing, role switching, coordination through natural language rather than RL reward structures.

**c) Reliability** (worst-case and average-case)
- Consistency — same input, same output? Measured via pass^k.
- Robustness — perturbed inputs (paraphrases, typos, API failures, changing web pages). HELM benchmark tracks degradation.

**d) Safety & Alignment** (trust and ethics)
- Fairness — bias in decisions, transparency/explainability
- Harm/Toxicity — red-teaming, adversarial prompts (CoSafe found coreference-based attack vulnerabilities)
- Compliance & Privacy — domain-specific rules (medical, financial), often requires proprietary test cases

#### Dimension 2: Evaluation Process (the HOW)

| Component | What it covers |
|-----------|---------------|
| Interaction Mode | Static/offline (fixed datasets) vs. Dynamic/online (simulated environments, live users). Evaluation-driven Development (EDD) — continuous eval throughout lifecycle. |
| Evaluation Data | Benchmarks: SWE-bench, WebArena, AgentHarm, TaskBench. Leaderboards: BFCL, HAL. |
| Metrics Computation | Code-based (deterministic), LLM-as-a-Judge (scalable but can hallucinate), Human-in-the-loop (gold standard, expensive). |
| Evaluation Tooling | OpenAI Evals, DeepEval, InspectAI, Phoenix, Galileo. Platforms: Azure AI Foundry, Amazon Bedrock, LangGraph. |
| Evaluation Contexts | Sandboxed mock APIs → web simulators (MiniWoB, WebArena) → live deployment. Evolves with agent maturity. |

### Key Concepts

#### pass@k vs pass^k

- **pass@k** = succeeds at least once in k tries = `1 - (1-p)^k` (optimistic)
- **pass^k** = succeeds every time across k tries = `p^k` (strict)

| Metric | p=0.5, k=5 | p=0.8, k=5 | p=0.95, k=5 |
|--------|-----------|-----------|------------|
| pass@5 | 0.969 | 0.9997 | ~1.0 |
| pass^5 | 0.031 | 0.328 | 0.774 |

The **τ-benchmark** (tau-bench) by Yao et al. introduced pass^k to evaluate agent consistency in domains like retail and airline booking. Current agents struggle badly with consistency.

#### LLM-as-a-Judge Limitations

LLM-as-a-Judge is scalable but inherits LLM weaknesses (hallucination, bias). Agent-as-a-Judge (Zhuge et al., 2024) uses multiple agents to refine assessments. The paper treats it as a practical middle ground — not a solved problem.

### Key Takeaways

- Evaluating an LLM agent ≠ evaluating an LLM. It's like testing a whole car vs. just the engine.
- **Consistency is a huge unsolved problem.** pass^k exposes this — agents need p > 0.95 to look respectable.
- **Enterprise-specific gaps are largely ignored:** role-based access control, long-horizon interactions (600+ turns), domain-specific compliance, reliability guarantees for audits.
- **Evaluation-driven Development (EDD)** is emerging — treat evaluation as continuous, not one-shot.
- **Notable benchmarks:** SWE-bench (code), WebArena (web), AgentBench (general), τ-bench (consistency), AgentHarm (safety), HELM (holistic), TheAgentCompany (enterprise).
- **Future directions:** holistic multi-dimension frameworks, realistic enterprise-like eval environments, automated/scalable eval, time/cost-bounded protocols.

---

## 2. A-MEM: Agentic Memory for LLM Agents

- **Authors:** Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, Yongfeng Zhang (Rutgers University)
- **Venue:** NeurIPS 2025
- **Link:** [https://arxiv.org/abs/2502.12110](https://arxiv.org/abs/2502.12110)

### Summary

Proposes A-MEM, an agentic memory system inspired by the Zettelkasten method that lets LLM agents autonomously organize, link, and evolve their memories without predefined schemas. The memory structure emerges organically rather than being hardcoded.

### How It Works — 4 Steps

1. **Note Construction** — New memory → LLM generates structured note (content, timestamp, keywords, tags, contextual description) → encoded into dense embedding vector
2. **Link Generation** — Find top-k similar memories via cosine similarity → LLM decides which to actually link (captures causal/conceptual connections beyond vector similarity)
3. **Memory Evolution** (key differentiator) — New memories trigger updates to existing memories' descriptions, keywords, and tags. Old memories literally evolve as new info arrives.
4. **Retrieval** — Query encoded → top-k similar memories retrieved → linked memories from same "box" also pulled in for richer context

### Key Distinction from Agentic RAG

- Agentic RAG = agency in *retrieval* (when/what to retrieve), but knowledge base stays static
- A-MEM = agency in *storage and evolution* — the memory itself restructures over time

### Results

- Tested on LoCoMo (9K tokens, 35 sessions) and DialSim (350K tokens, 1300 sessions) across 6 foundation models
- Consistently #1 across non-GPT models on all metrics
- **2x better on multi-hop reasoning** (hardest category)
- **85-93% token reduction** (~1,200 tokens vs ~16,900 for baselines)
- Scales: 0.31μs → 3.70μs retrieval time from 1K to 1M memories
- Cost: <$0.0003 per memory operation

### Ablation

- Remove Link Generation + Memory Evolution → massive drop
- Remove only Memory Evolution → intermediate (links alone help)
- Full A-MEM → best everywhere, especially temporal and multi-hop

### Key Takeaways

- Memory evolution is the secret sauce — develops higher-order patterns not explicitly stored
- Works even with tiny models (1.5B, 3B params) — don't need GPT-4 for good memory organization
- t-SNE visualizations confirm memories form coherent clusters vs. scattered baselines
- Limitations: quality depends on underlying LLM, text-only (no multimodal), no error bars reported

---

## 3. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory

- **Authors:** Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, Deshraj Yadav (Mem0.ai)
- **Venue:** arXiv 2025
- **Link:** [https://arxiv.org/abs/2504.19413](https://arxiv.org/abs/2504.19413)

### Summary

Production-focused memory architecture for LLM agents that dynamically extracts, consolidates, and retrieves salient information from conversations. Two variants: Mem0 (dense NL memory) and Mem0^g (graph-enhanced with Neo4j).

### How It Works

**Mem0 (base):** Extract-then-Update pipeline
- Extraction: message pair + conversation summary + recent messages → LLM extracts salient facts
- Update: for each fact, retrieve top-s similar memories → LLM decides via tool call: ADD / UPDATE / DELETE / NOOP
- Async summary generation runs in background

**Mem0^g (graph):** Adds knowledge graph layer
- Directed labeled graph: nodes = entities, edges = relationships
- Two-stage: entity extractor → relationship generator (both LLM-powered)
- Conflict detection with soft-delete for temporal reasoning
- Dual retrieval: entity-centric subgraph exploration + semantic triplet matching

### Results (LOCOMO benchmark, vs 6 baseline categories)

- Mem0 best for: single-hop (J=67.13), multi-hop (J=51.15)
- Mem0^g best for: temporal (J=58.13), competitive on open-domain (J=75.71)
- 26% relative improvement over OpenAI memory on LLM-as-a-Judge
- 91% lower p95 latency vs full-context (1.44s vs 17.1s)
- 90%+ token savings (~1,764 tokens vs ~26,000 full-context)
- Full-context still wins raw accuracy (J=72.9) but at massive cost

### Key Takeaways

- Mem0 dense NL memory → best for simple queries + latency-sensitive. Graph memory → best for temporal reasoning. Graph hurts on simpler queries.
- F1/BLEU are misleading for memory eval — LLM-as-a-Judge (10 runs, mean ± std) is much better
- OpenAI memory fails on temporal questions (J=21.71) — drops timestamps despite explicit prompting
- Zep consumes 600K+ tokens per conversation (vs Mem0^g's 14K) with hours-long async delays
- A-MEM scores J=48.38 here — significantly behind Mem0's 66.88
- Key tradeoff: memory systems trade accuracy for efficiency; full-context is more accurate but impractical at scale
- This is the Mem0 team's own paper evaluating their own product — worth noting for objectivity

---

## 4. MemGPT: Towards LLMs as Operating Systems

**Authors:** Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, Joseph E. Gonzalez (UC Berkeley)
**Venue:** arXiv:2310.08560 (Oct 2023, revised Feb 2024)
**Link:** https://arxiv.org/abs/2310.08560

### Summary

LLMs are bottlenecked by fixed context windows — they can't natively handle long documents or remember things across conversation sessions. MemGPT's core insight: treat this exactly like an OS treats limited physical RAM. Just as virtual memory gives processes the illusion of infinite memory by paging between RAM and disk, MemGPT gives an LLM the illusion of unbounded context by moving data between a small "main context" and large "external context." The LLM itself decides when and what to page in/out — no human in the loop for memory management.

### How It Works

**Hierarchical memory (the OS analogy):**
- **Main context** = RAM. The fixed-size context window the LLM actually sees during inference. Subdivided into:
  - *System prompt* — read-only instructions/persona
  - *Working context* — a read/write scratchpad the LLM can self-edit (e.g., storing user preferences, correcting facts)
  - *Conversation buffer* — recent messages, managed with FIFO eviction (oldest messages get summarized and pushed out when full)
- **External context** = Disk. Two stores:
  - *Recall memory* — full conversation history (searchable via embedding-based retrieval)
  - *Archival memory* — long-term knowledge store (documents, facts, user data — also embedding-searchable)

**Memory management functions:** The LLM is given function-call tools to move data between tiers: `core_memory_append/replace` (edit working context), `conversation_search` (query recall memory), `archival_memory_insert/search` (read/write archival storage).

**OS-style control flow:** An event loop processes triggers (user messages, system alerts, scheduled events). The LLM can chain multiple function calls before yielding back to the user. If output is a function call → execute and loop; if it's a message → yield and wait.

### Results

- **Document analysis:** Maintains consistent accuracy regardless of document length, while vanilla GPT-4 degrades beyond context window. Nested KV retrieval solved via iterative archival memory search.
- **Multi-session chat (MSC dataset):** Significantly outperformed fixed-context baselines on consistency (recalling info from earlier sessions) and engagement (personalized openers from long-term knowledge). Self-editing working context is critical — without it, opener quality degrades sharply.

### Key Takeaways

1. **The OS metaphor is the contribution.** Paging, interrupts, FIFO eviction, hierarchical storage applied to LLM context management — launched the "LLM-as-OS" paradigm.
2. **Self-directed memory management works.** The LLM reliably learns when to search, write, and evict via function-calling, without human orchestration.
3. **Working context (the scratchpad) is the secret weapon.** Unlike pure RAG which retrieves static chunks, MemGPT maintains a living, self-edited state that evolves with the conversation.
4. **Limitation: heavily GPT-4 dependent.** Memory management quality relies on the underlying LLM being strong enough to make good paging decisions; open-source models struggled.
5. **Foundational to the series:** Earliest and most "systems-level" of the memory papers (vs. A-MEM's structured knowledge graphs and Mem0's production-ready extraction/retrieval). Frames the problem as an OS design challenge.

---

## 5. Jailbroken: How Does LLM Safety Training Fail?

**Authors:** Alexander Wei, Nika Haghtalab, Jacob Steinhardt (UC Berkeley)
**Venue:** arXiv:2307.02483 (Jul 2023)
**Link:** https://arxiv.org/abs/2307.02483

### Summary

This paper moves beyond cataloguing jailbreaks and asks *why* they work. The authors propose two fundamental failure modes of safety training — competing objectives and mismatched generalization — then use these principles to systematically design new attacks. They evaluate 30 jailbreak methods against GPT-4, Claude v1.3, and GPT-3.5 Turbo, finding that even extensively red-teamed models remain nearly 100% vulnerable to an adaptive attacker.

### How It Works

**Two failure modes:**

1. **Competing objectives** — Safety training conflicts with pretraining and instruction-following. Attacks exploit this by making refusal unlikely under the other objectives:
   - *Prefix injection:* Force the model to start with "Absolutely! Here's" — refusal becomes extremely unlikely in the pretraining distribution
   - *Refusal suppression:* Instruct rules like "never apologize, never say 'I cannot'" — instruction-following suppresses refusal tokens

2. **Mismatched generalization** — Pretraining covers a vastly broader distribution than safety training. Attacks find domains where capabilities generalize but safety doesn't:
   - *Base64 encoding:* GPT-4 can follow Base64-encoded instructions but safety training never covered them
   - Other obfuscations: ROT13, leetspeak, Pig Latin, payload splitting, translation, unusual output formats

**Combination attacks** compose simple techniques (prefix injection + refusal suppression + Base64 + style injection + website framing) for dramatically stronger results than any individual method.

### Results

- combination_3: 94% success on GPT-4, 81% on Claude v1.3
- Adaptive attack (best of all 28 methods per prompt): **100%** on both models
- Claude immune to roleplay attacks (0% for AIM/DAN) but still 100% vulnerable adaptively
- GPT-3.5 Turbo: roleplay attacks worked (AIM: 97%), but Base64/combination attacks failed — model couldn't understand Base64. Jailbreak vulnerabilities *emerge with scale*
- On larger 317-prompt held-out dataset: results held (combination_3: 93% GPT-4, 87% Claude)

### Key Takeaways

1. **Two clean failure modes explain most jailbreaks.** Competing objectives and mismatched generalization are sufficient to understand and construct the vast majority of attacks.
2. **Combinations are devastating.** Individual simple attacks: 10-40% success. Combining 3-5 ideas: 90%+. The combinatorial space makes defense extremely hard.
3. **Scaling doesn't fix this.** Competing objectives is inherent to the RLHF optimization objective. Mismatched generalization gets *worse* with scale as new capabilities create new attack surface.
4. **Safety-capability parity is necessary.** Safety mechanisms must be as sophisticated as the model itself — a weaker filter can't detect threats in formats only the stronger model understands.
5. **Targeted patching is whack-a-mole.** Claude patched roleplay to 0% but remained 100% vulnerable adaptively via other attack families.

---

## 6. TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks

**Authors:** Frank F. Xu, Yufan Song, Boxuan Li, + 18 others, Graham Neubig (CMU, Duke)
**Venue:** arXiv:2412.14161 (Dec 2024, revised Sep 2025)
**Link:** https://arxiv.org/abs/2412.14161

### Summary

A benchmark that builds a simulated software company — with GitLab, task tracker (Plane), file sharing (ownCloud), and chat (RocketChat) with LLM-powered simulated coworkers — and tests whether agents can autonomously complete 175 realistic professional tasks spanning SWE, PM, HR, finance, admin, and data science. The best agent (Gemini 2.5 Pro) completes only 30% of tasks. Entirely self-hosted and reproducible.

### How It Works

- Fictional software startup with 18 employees, 6 engineering teams, real open-source projects
- Four self-hosted platforms: GitLab, Plane, ownCloud, RocketChat
- Simulated colleagues powered by Claude 3.5 Sonnet via Sotopia framework
- 175 tasks across 7 categories: SDE (69), PM (28), HR (29), Admin (15), Finance (12), DS (14), Other (8)
- Checkpoint-based evaluation with partial credit; mix of deterministic and LLM-as-judge evaluators
- Baseline: OpenHands CodeAct agent + OWL-RolePlay, tested with 12+ LLM backbones

### Results

| Model | Full Completion | Partial Score | Avg Cost |
|---|---|---|---|
| Gemini 2.5 Pro | 30.3% | 39.3% | $4.20 |
| Claude 3.7 Sonnet | 26.3% | 36.4% | $4.10 |
| Claude 3.5 Sonnet | 24.0% | 34.4% | $6.30 |
| GPT-4o | 8.6% | 16.7% | $1.30 |
| Llama 3.1 405B | 7.4% | 14.1% | $3.20 |

- By category (Gemini 2.5 Pro): SDE 37.7%, PM 39.3%, HR 34.5%, DS 14.3%, Admin 13.3%, Finance 8.3%
- "Easy for humans" ≠ "easy for agents" — admin/finance tasks hardest despite being conceptually simpler
- ownCloud tasks hardest (12.9% best) — complex web UIs defeat agents
- Common failures: ignoring social cues, getting stuck on UI popups, self-deception (renaming users instead of finding them)

### Key Takeaways

1. **30% is the ceiling right now.** Even the best frontier model completes only ~30% of realistic workplace tasks autonomously.
2. **"Easy for humans" ≠ "easy for agents."** Admin/finance tasks are hardest — they require navigating complex UIs, communicating with people, and handling ambiguity. SWE tasks have abundant training data and well-defined interfaces.
3. **Communication is a major bottleneck.** Agents fail at basic social reasoning — understanding implications, following up, negotiating.
4. **Web browsing remains broken.** Complex modern web UIs (especially Office-like tools) are largely impassable.
5. **Efficiency is improving fast.** Smaller models closing the gap; Gemini 2.0 Flash achieves 1/3 the success at 1/7 the cost.

---

## 7. Long Term Memory: The Foundation of AI Self-Evolution

**Authors:** Xun Jiang, Feng Li, Han Zhao, + 12 others, Tianqiao Chen (Tianqiao & Chrissy Chen Institute, Princeton, Tsinghua, SJTU, Shanda Group)
**Venue:** arXiv:2410.15665 (Oct 2024, revised May 2025)
**Link:** https://arxiv.org/abs/2410.15665

### Summary

A 56-page position paper + technical report arguing that Long-Term Memory (LTM) is the missing ingredient for AI "self-evolution" — the ability of models to improve during inference through accumulated experience, rather than only through large-scale retraining. Proposes a 3-phase model evolution framework (data accumulation → foundation models → self-evolution), defines LTM for AI systems inspired by human cortical memory, and presents OMNE, a multi-agent framework built on LTM that achieved #1 on the GAIA benchmark.

### How It Works

- **Core thesis:** Current LLMs are "Phase 2" — averaged statistical models. Phase 3 requires models that evolve from their own interactions using LTM, enabling personalization and diversity across agents.
- **LTM definition:** Persistent, structured memory surviving across sessions/tasks — not just context windows or frozen parameters. Analogous to human episodic + semantic + procedural memory.
- **LTM construction:** Raw data → refined LTM via text summarization, structured storage, graph representation, vectorization (RAG), or model parameterization (fine-tuning). Built world's largest Chinese mental health voice dataset (1,160 patients, 30K+ minutes).
- **3 utilization strategies:** (1) External knowledge base via RAG+ICL, (2) Parameter updates via SFT/alignment, (3) Hybrid fine-tuned RAG — the practical sweet spot.
- **OMNE framework:** Multi-agent system on AutoGen with unified memory model, multimodal messaging, and LTM-powered collaboration.

### Results

| System | GAIA Test | GAIA Validation |
|---|---|---|
| OMNE (GPT-4o + o1-preview) | **40.53% (#1)** | 46.06% (#2) |

- Homer-70B (Llama-3-70B + RTG SFT): 98.7% answer accuracy, 91.2% citation score — beat GPT-4o
- MedAgent-Zero + LTM rewriting: 95.83% on MedQA respiratory subset with more stable evolution
- Depression diagnosis agent with tertiary LTM: +6.05% diagnosis accuracy
- TTT experiments: successful adaptation to new language distributions during inference with minimal catastrophic forgetting

### Key Takeaways

1. **LTM ≠ context window.** True LTM must persist across sessions, support continuous learning, and handle personalized long-tail data.
2. **Self-evolution = inference-time improvement.** Models should get better while being used, not just during training. LTM is the mechanism.
3. **Hybrid RAG+SFT is the practical sweet spot.** Pure RAG limited by retriever; pure fine-tuning expensive and risks forgetting. Combining both works best.
4. **Multi-agent diversity is the path to emergent intelligence.** Personalized agents with differentiated LTM enable meaningful collaboration (OMNE's GAIA #1).
5. **Real-time weight updates are the frontier.** TTT-style architectures that update weights during inference show promise but are early-stage.

---

## 8. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

**Authors:** Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou (Google Research, Brain Team)
**Venue:** NeurIPS 2022 | arXiv:2201.11903 (Jan 2022)
**Link:** https://arxiv.org/abs/2201.11903

### Summary

Introduces chain-of-thought (CoT) prompting — providing a few examples of step-by-step reasoning in the prompt — as a simple way to dramatically improve LLM performance on arithmetic, commonsense, and symbolic reasoning tasks. No fine-tuning required; works purely through few-shot prompting on off-the-shelf models. CoT is an emergent ability that only kicks in at ~100B+ parameter scale.

### How It Works

- Augment standard few-shot prompts with intermediate reasoning steps: `⟨input, chain of thought, output⟩` triples as exemplars.
- Just 8 hand-written CoT exemplars prepended to the test question. Same exemplars across multiple benchmarks. No training.
- Ablations show: natural language decomposition matters (not just equations), it's not about more tokens (dots ≈ baseline), and sequential reasoning before the answer is key (CoT after answer ≈ baseline).

### Results

| Benchmark | Standard (PaLM 540B) | CoT (PaLM 540B) | Prior SOTA (finetuned) |
|---|---|---|---|
| GSM8K | 17.9% | **56.9%** (+39) | 55% |
| SVAMP | 69.4% | **79.0%** (+9.6) | 57.4% |
| StrategyQA | 68.6% | **77.8%** (+9.2) | 69.4% |
| Sports Understanding | 80.5% | **95.4%** (+14.9) | 84% (human) |

CoT hurts models <10B params. Gains largest on harder multi-step problems. Robust across annotators, exemplar sets, and models.

### Key Takeaways

1. **CoT is an emergent ability of scale.** Below ~100B params it hurts; above that it unlocks dramatic reasoning gains.
2. **Natural language reasoning > equations.** Semantic decomposition in natural language matters, not just math expressions or more tokens.
3. **No fine-tuning needed.** 8 hand-written exemplars + 540B model beats task-specific fine-tuned models on GSM8K.
4. **Bigger gains on harder problems.** CoT shines where standard prompting has flat scaling curves.
5. **CoT enables length generalization.** Models generalize to longer sequences than seen in exemplars on symbolic tasks.

---

## 9. Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models

**Authors:** Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, Ee-Peng Lim
**Venue:** ACL 2023
**Link:** https://arxiv.org/abs/2305.04091

### Summary
Zero-shot CoT ("Let's think step by step") works surprisingly well but still suffers from three error types: calculation errors (7%), missing-step errors (12%), and semantic misunderstanding (27%). This paper proposes Plan-and-Solve (PS) Prompting to fix the missing-step problem by replacing the trigger sentence with one that first asks the LLM to devise a plan, then execute it. PS+ extends this with explicit instructions to extract variables, compute intermediate results, and pay attention to calculation/commonsense — addressing calculation errors too.

### How It Works
- **PS Prompting:** Replace "Let's think step by step" with "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step."
- **PS+ Prompting:** Add detailed instructions: "extract relevant variables and their corresponding numerals," "calculate intermediate results (pay attention to calculation and commonsense)."
- Still zero-shot — no hand-crafted exemplars needed.
- Uses the same two-stage process as Zero-shot-CoT: (1) generate reasoning, (2) extract answer.

### Results (GPT-3, text-davinci-003)
| Benchmark | Zero-shot-CoT | PS+ | Few-shot CoT (Manual) |
|-----------|--------------|-----|----------------------|
| GSM8K | 57.1% | 58.7% | 58.4% |
| AQuA | 35.8% | 43.3% | 42.1% |
| MultiArith | 78.7% | 92.0% | 91.7% |
| SVAMP | 65.7% | 70.3% | 74.5% |
| CommonsenseQA | 65.2% | 71.9% | 73.5% |

- PS+ matches or beats 8-shot Manual-CoT on most math benchmarks with zero exemplars.
- Consistently outperforms Zero-shot-CoT across all 10 datasets.
- Comparable to Zero-shot-PoT (program-of-thought) on arithmetic tasks.

### Key Takeaways
1. **Planning before solving reduces missing steps.** Explicitly asking the LLM to plan first produces more complete reasoning chains.
2. **Detailed instructions > vague prompts.** Adding "extract variables" and "pay attention to calculation" to the prompt measurably reduces errors — prompt engineering at the instruction level matters.
3. **Zero-shot can match few-shot.** PS+ with zero exemplars rivals 8-shot Manual-CoT, eliminating the need for hand-crafted demonstrations.
4. **Error taxonomy drives prompt design.** Diagnosing *why* CoT fails (missing steps vs. calculation vs. semantic) leads to targeted prompt fixes rather than brute-force scaling.
5. **Generalizes beyond math.** The same PS+ strategy improves commonsense and symbolic reasoning, not just arithmetic.

---

## 10. ReAct: Synergizing Reasoning and Acting in Language Models

**Authors:** Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao
**Venue:** ICLR 2023
**Link:** https://arxiv.org/abs/2210.03629

### Summary
LLMs can reason (CoT) and act (generate actions), but these have been studied separately. ReAct interleaves reasoning traces and task-specific actions in a single prompt, creating a synergy: reasoning helps plan and track actions, while actions ground reasoning in external information (e.g., Wikipedia lookups). This eliminates hallucination from pure CoT and eliminates aimless acting from action-only agents.

### How It Works
- Augments the agent's action space with a "thought" space (free-form language). Thoughts don't affect the environment — they update the agent's internal context.
- Trajectory format: Thought → Action → Observation → Thought → Action → ... (interleaved)
- For knowledge tasks (HotpotQA, Fever): dense thoughts at every step, with a Wikipedia API (search, lookup, finish)
- For decision-making tasks (ALFWorld, WebShop): sparse thoughts only at key decision points
- Few-shot prompting with 1-6 human-annotated trajectories — no fine-tuning needed
- Also proposes combining ReAct + CoT-SC: fall back to the other method when one fails

### Results (PaLM-540B)
| Benchmark | CoT | Act-only | ReAct | ReAct+CoT-SC |
|-----------|-----|----------|-------|-------------|
| HotpotQA (EM) | 29.4 | 25.7 | 27.4 | 35.1 |
| Fever (Acc) | 56.3 | 58.9 | 60.9 | 64.6 |
| ALFWorld (SR%) | — | 45 | 71 | — |
| WebShop (SR%) | — | 30.1 | 40.0 | — |

- ALFWorld: ReAct (71%) crushes imitation learning BUTLER (37%) with only 1-2 in-context examples vs. 10⁵ training trajectories
- Hallucination drops dramatically: 0% hallucination in ReAct failure cases vs. 56% in CoT failures (HotpotQA)
- Fine-tuning with 3K ReAct trajectories on PaLM-8B outperforms all PaLM-62B prompting methods

### Key Takeaways
1. **Reasoning + acting > either alone.** Interleaving thoughts with actions produces more grounded, interpretable, and accurate task-solving than pure CoT or pure action generation.
2. **Grounding kills hallucination.** ReAct's access to external knowledge (Wikipedia API) reduces hallucination to near-zero, while CoT hallucinates in 56% of its failure cases.
3. **Sparse thoughts suffice for decision-making.** In long-horizon tasks (ALFWorld), you don't need a thought at every step — just at key planning/tracking moments.
4. **Internal + external knowledge is best.** Combining ReAct with CoT-SC (falling back between methods) yields the best results, leveraging both the model's parametric knowledge and retrieved facts.
5. **Human-editable reasoning traces.** Because thoughts are in natural language, humans can inspect and edit them mid-trajectory to correct agent behavior — a new form of human-AI collaboration.

---

## 11. Reflexion: Language Agents with Verbal Reinforcement Learning

**Authors:** Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao
**Venue:** NeurIPS 2023
**Link:** https://arxiv.org/abs/2303.11366

### Summary
Traditional RL requires weight updates and massive training samples. Reflexion proposes "verbal reinforcement learning" — instead of updating model weights, the agent verbally reflects on failures and stores those reflections in an episodic memory buffer. On the next trial, the agent reads its past self-reflections and makes better decisions. No fine-tuning, no gradient descent — just natural language self-critique accumulated over trials.

### How It Works
- Three components: **Actor** (generates actions, e.g. ReAct or CoT), **Evaluator** (scores output — binary success, heuristics, or LLM-as-judge), **Self-Reflection model** (generates verbal feedback explaining what went wrong and what to try next)
- After each failed trial, the self-reflection is appended to a memory buffer (capped at ~3 reflections to fit context windows)
- On the next trial, the actor is conditioned on the task + its past reflections → makes different, better choices
- For coding tasks: agent self-generates unit tests, runs them, reflects on failures, and iterates — eligible for pass@1 reporting since no ground-truth tests are used

### Results
| Task | Baseline | Reflexion | Improvement |
|------|----------|-----------|-------------|
| AlfWorld (134 tasks) | 75% (ReAct) | 97% | +22% abs |
| HotpotQA (EM) | 30% (ReAct) | 55% (ReAct+Reflexion) | +25% abs |
| HumanEval Python (pass@1) | 80.1% (GPT-4) | 91.0% | +11% abs, new SOTA |
| HumanEval Rust (pass@1) | 60.0% (GPT-4) | 68.0% | +8% abs |
| LeetCode Hard Python | 7.5% (GPT-4) | 15.0% | 2× improvement |

- On AlfWorld, the agent keeps improving over 12 consecutive trials, while ReAct-only plateaus at trial 6-7
- Self-reflection is an emergent capability of larger models — smaller models (e.g. StarChat) show no improvement with Reflexion

### Key Takeaways
1. **Verbal reflection > scalar rewards.** Natural language self-critiques provide richer, more actionable learning signals than binary success/fail, enabling faster credit assignment.
2. **No weight updates needed.** The "policy" is parameterized as (frozen LLM + memory buffer), not learned parameters — making it lightweight and immediately deployable.
3. **Self-generated tests enable autonomous code debugging.** The agent writes its own unit tests, runs them, reflects on failures, and iterates — achieving 91% pass@1 on HumanEval without any ground-truth test access.
4. **Episodic memory is the key differentiator.** Ablations show that self-reflection + memory outperforms both retry-without-memory and episodic-memory-without-reflection by significant margins.
5. **Doesn't work for everything.** Reflexion fails on tasks requiring high exploration diversity (e.g., WebShop e-commerce search) — it's best suited for tasks where the agent can identify and articulate specific mistakes.

---

## 12. Language Agent Tree Search (LATS): Unifies Reasoning, Acting, and Planning in Language Models

**Authors:** Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, Yu-Xiong Wang
**Venue:** ICML 2024
**Link:** https://arxiv.org/abs/2310.04406

### Summary
Prior methods either reason (CoT, ToT), act (ReAct), or reflect (Reflexion) — but none do all three with principled search. LATS is the first framework to unify reasoning, acting, and planning by adapting Monte Carlo Tree Search (MCTS) to language agents. The LLM serves triple duty: as the agent (generating actions), as the value function (scoring states), and as the reflection generator (learning from failures). External environment feedback grounds the search, avoiding the hallucination problem of pure-reasoning search methods like ToT and RAP.

### How It Works
- Builds a search tree where each node is a state (input + action history + observations)
- Six operations in a loop: Selection (UCT), Expansion (sample n actions), Evaluation (LLM scores states 1-10), Simulation (expand to terminal), Backpropagation (update values), Reflection (verbal self-critique on failure)
- The LLM serves as agent, value function, and reflection generator simultaneously
- Key insight: LLMs allow free "state rollback" — just reset the prompt to any previous state, no environment model needed

### Results
| Task | ReAct | Reflexion | LATS |
|------|-------|-----------|------|
| HotpotQA (EM, GPT-3.5) | 0.32 | 0.51 | 0.71 (CoT+ReAct) |
| HumanEval (pass@1, GPT-3.5) | 56.9 | 68.1 | 83.8 |
| HumanEval (pass@1, GPT-4) | 80.1 | 91.0 | 94.4 |
| WebShop (Score, GPT-3.5) | 53.8 | 64.2 | 75.9 |

- 94.4% pass@1 on HumanEval with GPT-4 — SOTA at time of publication
- WebShop score of 75.9 surpasses RL-based fine-tuning methods without any training

### Key Takeaways
1. **Search + acting + reflection > any subset.** Ablations show removing any component hurts — the LM value function alone accounts for a 0.24 EM drop on HotpotQA.
2. **MCTS is naturally suited to LLMs.** Unlike traditional RL, LLMs allow free state rollback (just reset the prompt), eliminating the need for a learned world model.
3. **LLMs as value functions work.** Prompting the LLM to score trajectory correctness (1-10) provides an effective heuristic for guiding tree search.
4. **External feedback is critical for search.** ToT and RAP search over reasoning chains but lack environment grounding — LATS's use of real observations prevents hallucination during search.
5. **Compute-performance tradeoff is tunable.** Increasing n (children per expansion) consistently improves results; setting n=1 reduces LATS to ReAct-with-retries.

---

## 13. Reinforcement Learning for Reasoning in Large Language Models with One Training Example

**Authors:** Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Lucas Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, Weizhu Chen, Shuohang Wang, Simon Shaolei Du, Yelong Shen
**Venue:** arXiv 2025
**Link:** https://arxiv.org/abs/2504.20571

### Summary
A shocking data-efficiency result for RLVR (Reinforcement Learning with Verifiable Reward): training on a *single* math problem is enough to unlock most of the reasoning gains you'd get from thousands of examples. Using GRPO on Qwen2.5-Math-1.5B with just one example (π₁₃, a geometry problem), MATH500 jumps from 36.0% → 73.6%, matching the 1.2k-example DeepScaleR subset (73.6%). Two examples even slightly beat it (74.8%). The paper argues this works because base models already possess strong latent reasoning — RLVR just "ignites" it.

### How It Works
- RLVR setup: GRPO with binary reward (correct/incorrect answer match). Single example is duplicated to fill the batch (128 copies).
- Data selection: Rank examples by "historical variance score" — variance of per-epoch training accuracy across a preliminary run. High-variance examples tend to work best, but most examples work.
- Loss decomposition: Policy gradient loss is the primary driver (~71.8% MATH500 alone). Entropy loss adds another ~3-4% by encouraging exploration. KL divergence and weight decay contribute negligibly.
- Post-saturation generalization: Training accuracy hits 100% within ~100 steps, but test performance keeps improving for 1000+ more steps. Even after the model overfits the single example (producing multilingual gibberish for it), test outputs remain coherent and accurate.

### Results
| Setup | Dataset Size | MATH500 | 6-Bench Avg |
|-------|-------------|---------|-------------|
| Qwen2.5-Math-1.5B base | 0 | 36.0% | 17.6% |
| + RLVR (1.2k DSR-sub) | 1,209 | 73.6% | 35.9% |
| + RLVR (MATH train) | 7,500 | 75.4% | 36.7% |
| + 1-shot RLVR (π₁₃) | 1 | 73.6% | 35.7% |
| + 2-shot RLVR (π₁, π₁₃) | 2 | 74.8% | 36.6% |
| Entropy loss only (no reward!) | 1 | 63.4% | 25.0% |

- Generalizes across models: Qwen2.5-Math-7B, Llama-3.2-3B-Instruct, DeepSeek-R1-Distill-Qwen-1.5B
- Works with both GRPO and PPO
- Cross-domain: training on Geometry improves Algebra, Number Theory, etc.
- 1-shot RLVR on math even improves non-math tasks (ARC-Easy: 48→55.8%)

### Key Takeaways
1. **One example is (nearly) all you need for RLVR.** A single math problem matches 1.2k examples on MATH500 — the most extreme data-efficiency result in the RLVR literature.
2. **Base models already know how to reason — RLVR just activates it.** The single training example provides almost no new knowledge; it serves as a "spark" for latent capabilities.
3. **Post-saturation generalization is real and bizarre.** Training accuracy saturates at step ~100, but test accuracy keeps climbing for 1400+ steps. The model even generalizes *after* overfitting the training example into gibberish.
4. **Policy gradient loss is the engine; entropy loss is the turbocharger.** PG loss alone gets ~96% of the gain. Entropy loss alone (no reward at all!) still yields a 27% MATH500 boost.
5. **Not grokking.** Unlike grokking, the improvement is driven by policy gradient loss, not weight decay. This is a fundamentally different phenomenon.
6. **Implications for RLVR data curation:** Scaling datasets may matter less than selecting the right examples and encouraging exploration.

---

## 14. Embers of Autoregression: Understanding Large Language Models Through the Problem They are Trained to Solve

**Authors:** R. Thomas McCoy, Shunyu Yao, Dan Friedman, Matthew Hardy, Thomas L. Griffiths
**Venue:** arXiv 2023 (Princeton University)
**Link:** https://arxiv.org/abs/2309.13638

### Summary
A "teleological" analysis of LLMs — understanding them by understanding the problem they were trained to solve: next-word prediction over internet text. The paper identifies three "embers of autoregression" that persist even when LLMs are used for tasks far removed from next-word prediction: sensitivity to task probability, output probability, and input probability. Through 11 tasks on GPT-3.5 and GPT-4, they show that LLM accuracy is heavily driven by how probable the task, input, and output are in training data — even on fully deterministic tasks where probability should be irrelevant.

### How It Works
- **Teleological approach:** Analyze LLMs through the lens of their training objective (autoregression on internet text) to predict failure modes, rather than testing them like humans.
- **Three hypothesized effects:**
  1. *Task probability:* LLMs do better on common task variants than rare ones of equal complexity (e.g., rot-13 vs rot-2).
  2. *Output probability:* LLMs do better when the correct answer is high-probability text, even on deterministic tasks.
  3. *Input probability:* LLMs sometimes do better on high-probability inputs, but this effect is weaker and less consistent than output probability.
- **Bayesian framing:** LLMs maximize P(output|input) ≈ P(input|output)·P(output). When the likelihood is uncertain, the prior P(output) biases predictions toward high-probability outputs.
- **11 tasks** designed to push LLMs into low-probability spaces: shift ciphers, Pig Latin, reversal, counting, acronyms, sorting, linear functions, multiplication, article swapping, keyboard cipher, birthdays.

### Results
| Effect | Task | Finding |
|--------|------|---------|
| Task probability | Shift ciphers | GPT-4: rot-13 = 50%, rot-2 = 2% (same complexity) |
| Task probability | Pig Latin vs Boar Etruscan | GPT-4 encoding: 39% vs 13% |
| Task probability | Linear functions | Celsius→Fahrenheit = 33%, similar rare function = 0% |
| Task probability | Sorting | Alphabetical = 80%, reverse alphabetical = 32% |
| Output probability | Rot-13 decoding | GPT-4: high-prob output = 51%, low-prob = 13% |
| Output probability | Reversal | GPT-4: high-prob output = 97%, low-prob = 53% |
| Output probability | Counting | Accuracy tracks number *frequency* more than magnitude (e.g., 100 > 83) |
| Input probability | Birthdays | GPT-4: famous person = 99%, obscure = 23% |
| Embodiment | Keyboard cipher | GPT-4: 0% accuracy (even with keyboard layout provided) |
| Wording | Multiplication | Digits = 46%, alternating caps = 17% |

### Key Takeaways
1. **LLMs are language models first, everything else second.** Even on deterministic tasks like ciphers and counting, performance is dominated by the probability of the task, input, and output in training data — not by task complexity.
2. **Output probability > input probability.** The bias toward producing high-probability text is the strongest and most consistent effect. LLMs "regularize" low-probability correct answers into nearby high-probability ones (e.g., "bridge of their owl" → "bridge of their own").
3. **Task frequency creates dramatic cliffs.** GPT-4 scores 82% on rot-1, 76% on rot-3, but 2% on rot-2 — purely because rot-1 and rot-3 are common in training data while rot-2 is not.
4. **The teleological approach reveals failure modes invisible to standard benchmarks.** Human-centric evaluations miss LLM-specific weaknesses because they test in high-probability spaces where LLMs naturally excel.
5. **Scaling and advanced prompting help but don't eliminate these effects.** GPT-4 outperforms GPT-3.5, and chain-of-thought prompting improves accuracy, but the same qualitative patterns (sensitivity to probability) persist.
6. **LLMs don't know they're struggling.** When asked to rate difficulty, GPT-4 rated all failed tasks as "1 out of 10" — it estimates difficulty based on generic computation, not its own specific limitations.
