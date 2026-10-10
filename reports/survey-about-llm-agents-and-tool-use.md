# Survey on LLM Agents and Tool Use: Architectures, Benchmarks, and Security

## TL;DR
Large Language Model (LLM) agents coupled with external tools represent a transformative shift from static text generation to dynamic, autonomous problem-solving. This survey synthesizes recent advancements across agent architectures, reasoning and planning frameworks, execution and retrieval mechanisms, rigorous benchmark suites, and critical security vulnerabilities. Key developments include neuro-symbolic and graded transformer integrations [1][2], hypergraph-based tool schema planning [3], budget-aware execution-centric retrieval [4][5], and rigorous multi-agent workflows [6][7]. However, significant challenges persist around prompt injection [8], unvetted skill supply chains [9][10], and robust real-time monitoring [11].

## Background
The evolution of LLM-based agents stems from the inherent limitations of static autoregressive language models in executing real-world, multi-step computational tasks [2]. While vanilla LLMs excel at fluent generation, they lack persistent memory, real-time environmental grounding, and direct access to external APIs or computational engines [2]. To bridge this gap, foundational paradigms such as MRKL systems established modular neuro-symbolic architectures combining language models with discrete reasoning and expert modules [2]. 

Contemporary agent frameworks extend these ideas by profiling language models into distinct roles—such as policy models, dynamic evaluators, and feedback coordinators—that operate iteratively over reasoning-acting loops (e.g., ReAct, Reflexion, LATS) [12][13]. These systems interleave generation with tool execution, using external environments not merely as passive output sinks but as active sources of state observation and verification feedback [12].

## Architectural Foundations of LLM Agents
Modern LLM agent architectures rely on control structures that integrate planning, memory, reflection, and tool invocation [13]. Planning mechanisms are broadly categorized into task decomposition, multi-plan selection, external module-aided planning, reflection-based refinement, and memory-augmented planning [13]. To overcome the combinatorial explosion and semantic-functional gaps in massive API ecosystems, recent architectures introduce advanced schema planning, such as directed Tool-Schema Hypergraphs in HyperAgent [3] and graded internal symbolic representations within transformers [1].

Furthermore, tool invocation mechanisms have evolved from naive token-level triggers to sophisticated reasoning-acting strategies that manage inference cycles and confidence thresholds [12]. Single-agent frameworks are increasingly complemented by multi-agent coordination systems, such as Magentic-One, where a lead Orchestrator plans, tracks progress, and delegates tasks across specialized worker agents [7]. Controlled evaluations indicate that while multi-agent coordination enhances modularity, runtime-generated and protocol-aligned workflows often outperform static multi-agent topologies [6].

## Tool Retrieval and Execution Environments
Effectively discovering and executing tools within massive API spaces remains a central engineering challenge. Recent literature formalizes tool retrieval as a resource-constrained sequential decision-making problem. For example, Lookahead-R implements budget-aware tool retrieval via execution-centric planning using lightweight surrogate world models to jointly predict success and cost [4]. Similarly, the "execution prior" paradigm leverages short, ordered state-path tool menus to surface prerequisite tools before final actions [5].

For execution environments, production-grade agent software development kits (e.g., OpenHands) provide composable, secure, and extensible foundations equipped with sandboxed execution environments and flexible interaction harnesses [14]. In software engineering domains, reproducible post-training frameworks like SWE-Master [15] and standardized harnesses evaluated via benchmarks like Claw-SWE-Bench [16] ensure that code generation and shell execution occur under rigorously controlled conditions.

## Benchmarking and Cognitive Depth
Evaluating LLM agents requires moving beyond simple end-to-end task success rates toward granular, trajectory-level assessments of cognitive depth [17]. Comprehensive benchmark suites such as SWE-bench (and continuously updated variants like SWE-bench-Live) evaluate software issue resolution on real GitHub repositories [18][19]. Other benchmarks target distinct functional modalities: GAIA assesses general assistant capabilities across multi-difficulty browsing and code execution tasks, ToolBench evaluates API composition, and WebArena tests self-contained browser navigation in isolated website environments [19].

Additionally, frameworks like AgencyBench test autonomous agents across real-world scenarios within 1M-token contexts [20], while operational criteria for "agentic cognitive depth" evaluate trajectories across context sensitivity, temporal continuity, multimodal coordination, adaptive interaction, and metacognitive monitoring [17]. To establish reliable agent provenance and protect intellectual property, semantic behavioral watermarking embeds robust markers directly into high-level agent action choices without modifying output tokens [21].

## Security, Threat Surfaces, and Vulnerabilities
As agents gain autonomous execution capabilities and direct system access, security emerges as a paramount concern. Lifecycle-based threat analyses synthesize agent security around information flow, delegated authority, and persistent state corruption [22]. Prompt injection and tool-mediated control-flow hijacking dominate reported vulnerabilities, frequently leading to unauthorized data exfiltration or unintended actions [22].

Empirical evaluations reveal severe vulnerabilities in agent ecosystems. For instance, Skill-Inject demonstrates that frontier LLM agents exhibit up to 80% vulnerability to malicious instructions embedded within unvetted skill files (`SKILL.md` and scripts) [10]. Because agent skills operate as an unvetted software supply chain lacking formal registries or version provenance [9], automated supply-chain tracing and browser-level sandboxing (such as ceLLMate, which restricts ambient authority at the HTTP layer) are critical mitigation strategies [23].

## Runtime Defenses and Real-Time Monitoring
Mitigating active threats requires robust runtime defenses and intervention mechanisms. Current prompt injection defenses face difficult trade-offs: text-level filters lack resilience against sophisticated payload manipulations, whereas execution-level monitors can incur up to 3x computational latency or trigger excessive early task refusals [8]. 

To address these limitations, advanced systems employ automated red-teaming engines like ASPIRE, which maintain evolving Agent Security Behavior Graphs to discover latent vulnerabilities [24]. Furthermore, real-time monitoring approaches such as OnTrack utilize streaming structure-aware optimal transport to compare agent steps and dependency graphs in real-time, enabling proactive intervention without the prohibitive computational overhead of post-hoc verification [11].

## Trends and Open Problems
The research landscape of LLM agents and tool use points toward several vital future directions and open problems:
1. **Internalized vs. Externalized Tools:** Investigating the transition from discrete API calls to graded internal symbolic representations and morphisms within transformer hidden states [1].
2. **Execution-Aware Retrieval:** Scaling tool discovery in massive API libraries through budget-aware planning and state-path execution priors rather than static semantic matching [4][5].
3. **Supply Chain Security:** Establishing cryptographic provenance, verified registries, and robust sandboxing for agent skill files and plugin ecosystems to prevent widespread injection attacks [9][10][23].
4. **Real-Time Safety Interventions:** Moving beyond expensive post-hoc evaluations to streaming trajectory-level monitoring and optimal transport verification that balance security with low latency [11].

## References
[1] Internalizing Tools as Morphisms in Graded Transformers. arxiv. https://arxiv.org/abs/2511.17840 (2025-11-21)
[2] MRKL Systems: A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning. arxiv. https://arxiv.org/abs/2205.00445 (2022-05-01)
[3] HyperAgent: Planning and Acting over Tool-Schema Hypergraphs for Tool-Use LLM Agents. hf-search. https://huggingface.co/papers/2608.02650 (2026-07-31)
[4] Lookahead-R: Budget-Aware Tool Retrieval via Execution-Centric Planning. arxiv. https://arxiv.org/abs/2609.35811 (2026-09-20)
[5] The Menu Is an Execution Prior: State-Path Tool Menus for Online Agents. arxiv. https://arxiv.org/abs/2609.09395 (2026-09-08)
[6] Do More Agents Help? Controlled and Protocol-Aligned Evaluation of LLM Agent Workflows. hf-search. https://huggingface.co/papers/2606.05670 (2026-06-04)
[7] Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks. arxiv. https://arxiv.org/abs/2411.04468 (2024-11-07)
[8] The Landscape of Prompt Injection Threats in LLM Agents: From Taxonomy to Analysis. arxiv. https://arxiv.org/abs/2602.10453 (2026-02-11)
[9] Skill Constellations: Tracing the Supply Chain of Agent Skills on GitHub. hf-daily. https://huggingface.co/papers/2610.11169 (2026-10-08)
[10] Skill-Inject: Measuring Agent Vulnerability to Skill File Attacks. hf-search. https://huggingface.co/papers/2602.20156 (2026-02-23)
[11] OnTrack: Real-Time Monitoring and Intervention in LLM Agent Trajectories via Streaming Structure-Aware Optimal Transport. arxiv. https://arxiv.org/abs/2610.12375 (2026-10-08)
[12] A Review of Prominent Paradigms for LLM-Based Agents: Tool Use (Including RAG), Planning, and Feedback Learning. web. https://arxiv.org/abs/2406.05804 (2024-06-09)
[13] The Landscape of Emerging AI Agent Architectures. web. https://arxiv.org/abs/2404.11584 (2024-04-17)
[14] The OpenHands Software Agent SDK: A Composable and Extensible Foundation for Production Agents. hf-search. https://huggingface.co/papers/2511.03690 (2025-11-05)
[15] SWE-Master: Unleashing the Potential of Software Engineering Agents via Post-Training. hf-search. https://huggingface.co/papers/2602.03411 (2026-02-03)
[16] Claw-SWE-Bench: A Benchmark for Evaluating OpenClaw-style Agent Harnesses on Coding Tasks. hf-search. https://huggingface.co/papers/2606.12344 (2026-06-10)
[17] Agentic Cognitive Depth: Operational Criteria for Evaluating LLM Agents. arxiv. https://arxiv.org/abs/2610.04168 (2026-10-03)
[18] SWE-bench Goes Live!. hf-search. https://huggingface.co/papers/2505.23419 (2025-05-29)
[19] Benchmark Suites: SWE-Bench, GAIA, ToolBench — AI Agents. web. https://www.coddykit.com/courses/ai_agents_basic/benchmark-suites-swe-bench-gaia-toolbench-4256074 (n.d.)
[20] AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts. hf-search. https://huggingface.co/papers/2601.11044 (2026-01-16)
[21] Semantic Behavioral Watermarking: Paraphrase-Robust and Forgery-Resistant Provenance for LLM Agents. arxiv. https://arxiv.org/abs/2610.08668 (2026-10-06)
[22] Toward Secure LLM Agents: Threat Surfaces, Attacks, Defenses, and Evaluation. arxiv. https://arxiv.org/abs/2606.10749 (2026-06-10)
[23] ceLLMate: Sandboxing Browser AI Agents. hf-search. https://huggingface.co/papers/2512.12594 (2026-03-04)
[24] ASPIRE: Agentic Safety & Prompt Injection Red-teaming Engine. arxiv. https://arxiv.org/abs/2610.08951 (2026-10-06)
