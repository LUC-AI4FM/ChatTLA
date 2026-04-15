Can LLMs Write Correct TLA+ Specifications?
Evaluating Natural-Language-to-TLA+ Generation
Anonymous Authors
Keywords:
Large Language Models, Temporal Logic of Actions, TLA+, Formal Specification, Verification,
Code Generation, Natural Language Processing
Abstract:
TLA+ has supported industrial verification at companies such as Amazon and Microsoft, yet writ-
ing correct TLA+ specifications from natural language still requires time and expertise, which
limits adoption. LLMs show promise, but no prior study measures whether they produce semanti-
cally correct TLA+ specifications from natural language. This paper presents the first systematic
evaluation of LLM-based TLA+ specification synthesis from natural language. Our study evaluates
30 LLMs across eight families on a curated dataset of 205 TLA+ specifications: 25 open-weight
models across four prompting strategies (2,600 runs) and 5 proprietary models under few-shot
prompting (130 runs), all validated by the SANY parser and TLC model checker. LLMs achieve
up to 26.6% syntactic correctness but only 8.6% semantic correctness, with successes exclusive
to progressive prompting. Results show that model size does not predict quality, e.g., DeepSeek
r1:8b outperforms its 70B variant across all strategies, which suggests the importance of reason-
ing alignment for formal languages. Code-specialized models consistently underperform due to
negative transfer from mainstream language training. We identify five recurring hallucination cat-
egories, all traceable to specific training data biases. These results suggest that current LLMs
do not generate reliable TLA+ specifications without expert oversight. We release the evaluation
framework, code, and dataset to support reproducibility and future research.

> *Note (2026-04-14): the 8.6% semantic-correctness ceiling in this paper measures **off-the-shelf** LLMs under prompting alone. A companion fine-tuning study (`docs/chattla_finetuning_paper.tex`) reports 30% pass on a harder 30-problem Diamond holdout (9/30) using repair-based GRPO on `gpt-oss:20b`. The evaluation protocols differ — Diamond adds a mutation-sensitivity clause that the 8.6% figure does not — so the numbers are not directly comparable, but they are on the same language against the same verifier and suggest the off-the-shelf ceiling is not the ceiling of the approach.*
1 Introduction
Formal specification is important to build reli-
able distributed and concurrent systems (Lam-
port, 2002). Writing correct formal system spec-
ifications from natural language remains diffi-
cult and time-consuming. Large language models
(LLMs) have made progress in code generation,
but their ability to produce semantically correct
formal specifications from natural language has
not been systematically evaluated. This work
addresses TLA+ specification (synthesis) from
natural language with LLMs. TLA+ is widely
adopted in industry, e.g., Amazon Web Services
has used it since 2011 to verify DynamoDB,
S3, and EBS (Newcombe et al., 2015), and Mi-
crosoft Azure applied it to Cosmos DB (Cirstea
et al., 2024). TLA+ combines temporal logic,
first-order logic, and set theory, which makes it
technically difficult for LLMs to generate cor-
rect specifications. TLA+ is also a specialized
formal language with a public corpus of only a
few hundred modules, compared with millions of
C, Python, and Java samples in standard LLM
pre-training data. This data imbalance makes
TLA+ syntax and semantics hard for LLMs to
learn and leaves natural-language-to-TLA+ gen-
eration under-studied.
To address this gap, we curate a benchmark of
205 TLA+ specifications from the TLA+ Founda-
tion (TLA+ Community, 2025), each paired with
natural-language comments and TLC configura-
tions, with train, validation, and test splits. We
then introduce a reproducible evaluation frame-
work for LLM-based natural-language-to-TLA+
generation. We benchmark 30 models from eight
families, i.e., DeepSeek, LLaMA, Qwen, QwQ,
GPT-OSS, code-specialized (e.g., CodeLLaMA
and Granite), instruction-tuned (e.g., Mistral,
Phi, Gemma, and Starling-LM), and proprietary
APIs (e.g., OpenAI GPT and Anthropic Claude)
on the curated dataset. The core evaluation uses
25 open-weight models across four prompt strate-
gies (2,600 runs). We also evaluate five propri-
etary models with few-shot (k = 3) only (130
runs). All outputs from all runs are validated
by the SANY parser and TLC checker.
Key Findings. (1) LLMs achieve up to 26.6%
syntactic correctness on TLA+ (SANY) but only
8.6% semantic correctness (TLC). Semantic suc-
cess is exclusive to progressive prompts. (2)
Model size does not predict quality. The 8B-
parameter DeepSeek r1 outperforms its 70B sib-
ling across all strategies. Reasoning-trace fine-
tuning is a stronger predictor of quality than
model size. (3) Code-specialized models consis-
tently underperform general-purpose ones due to
negative transfer (due to programming language
bias) from mainstream programming logic and
language training. (4) We identify five recur-
ring hallucination categories: Unicode operator
substitution, cross-language syntax injection, rea-
soning and formatting leakage, generation length
miscalibration, and structural errors. These er-
rors map to biases in current open-source training
data (i.e., code, formal math, and reasoning sam-
ples) and point to two mitigation insights, e.g.,
obtaining high-quality datasets for specifications
and grammar-constrained generation.
Contributions. Our contributions are:
• We curate a dataset for natural-language-to-
TLA+ specification synthesis, consisting of 205
TLA+ specifications paired with natural lan-
guage comments and TLC configurations.
• We provide a systematic evaluation of LLM-
based TLA+ specification generation/synthesis
from natural language, testing 30 models from
eight families across four prompting strategies
with syntactic and semantic validation.
• We establish quantified baselines for TLA+
synthesis and highlight the gap between cur-
rent LLM capabilities and reliable specification
synthesis. We provide a taxonomy of five sys-
tematic hallucination categories in TLA+ gen-
eration/synthesis and offer directions for im-
provement. We release code, dataset, models,
and results to support future work.
2 Background & Related Work
TLA+ and Formal Specification. Distributed
and concurrent systems are difficult to design and
verify. Small errors in the system’s logic can cre-
ate bugs that appear only under rare timing con-
ditions, so standard tests often cannot find them.
Formal verification solves this problem by the ap-
plication of mathematical reasoning to prove that
a system behaves correctly in every possible sce-
nario. TLA+ (Temporal Logic of Actions) (Lam-
port, 2003) was created to describe concurrent
and distributed systems behavior through math-
ematical formulas that combine temporal logic,
first-order logic and set theory.
A TLA+ specification has several key compo-
nents. The initial predicate Init defines all valid
start states and the next-state relation Next de-
scribes how the system moves from one state to
another through atomic actions. The expression
Init ∧  [Next]v defines the core behavior of the
system. The symbol  represents “always.” Sys-
tem properties are expressed as invariants (safety
properties that hold in every reachable state) and
liveness properties (behaviors that must eventu-
ally occur). The TLA+ toolchain provides two
verification methods: the TLC model checker ex-
amines every reachable state in a finite version
of a specification (Lamport, 2003), while TLAPS
allows machine-checked deductive reasoning for
infinite-state specifications (Cirstea et al., 2024).
Although TLA+ is effective at the detection of
design flaws, the creation of correct specifications
requires specialized expertise. Engineers must
translate informal descriptions with ambiguities
and hidden assumptions into precise mathemati-
cal expressions that resolve questions about fail-
ure modes, concurrency and consistency. Small
mistakes such as an incorrect quantifier bound or
a missing fairness condition can make a specifica-
tion compile but fail to capture intended behav-
ior. This expertise barrier limits TLA+ adoption.
LLM-based Generation. LLMs trained on
large collections of source code can generate
code from text. Models such as GPT-4, Gem-
ini, Claude and Code Llama can solve standard
benchmarks like HumanEval (Austin et al., 2021).
But LLMs still make many mistakes: they some-
times create APIs or library functions that do
not exist, produce code that does not follow the
rules in the specification and struggle with prob-
lems that require several steps of reasoning (Ye
et al., 2023). The generation of formal specifica-
tions is even harder than code creation. A nor-
mal program can be checked by execution, but a
formal specification must follow strict mathemat-
ical rules. A TLA+ specification may be written
correctly with proper operators and syntax but
still describe the wrong system behavior by leav-
ing out a fairness rule, defining a condition that
is too weak, or leaving important variables unde-
fined. These problems cannot be found by parsers
or type checkers and show up only during formal
verification or detailed expert review.
LLM hallucination, where a model confidently
produces incorrect output, has been classified for
software tasks into task requirement conflicts, fac-
tual knowledge conflicts and project context con-
flicts (Zhang and Zhang, 2025). Package halluci-
nation creates supply chain risks (Spracklen et al.,
2025). For formal languages like TLA+, halluci-
nated errors may only surface during verification.
GenAI for TLA+. The TLA+ Foundation
launched the GenAI-accelerated TLA+ Challenge
in 2025 to investigate generative AI applications
for formal methods (TLA+ Foundation, 2025).
Specula, the winner, generates TLA+ specifica-
tions from code that already exists (Cheng et al.,
2025). The system converts source code into
intermediate TLA+ representations, fixes errors
through Retrieval-Augmented Generation and it-
eratively corrects runtime errors identified by
TLC. The creation of the Raft specification took
about 1.5 hours of manual effort. The second-
place entry tested whether LLMs can be guided to
produce valid TLA+ syntax through token-level
rules such as Greibach Backus-Naur Form and
the Guidance framework (Helwer, 2025). This
method focuses primarily on the production of
syntactically correct specifications while seman-
tic correctness is not addressed. Gregory Terzian
studied the opposite approach: the use of com-
pleted TLA+ specifications to guide LLMs in the
generation of Rust code (Terzian, 2025).
SysMoBench (Cheng et al., 2026) is the first
benchmark for LLMs on complex distributed sys-
tems in TLA+, but it compares generated spec-
ifications to source code implementations. Our
work evaluates generation from natural language
only. Previous specification synthesis work fo-
cused on invariant discovery from program traces:
DistAI infers inductive invariants for distributed
protocols (Yao et al., 2021) and SpecGen uses
LLMs to generate function-level Java specifica-
tions through iterative verification feedback (Ma
et al., 2025). Neither generates system-level spec-
ifications like TLA+.
Gap in Prior Work and Our Contribution.
Existing GenAI research on TLA+ focuses on the
creation of specifications from code, the enforce-
ment of syntax through token constraints and the
use of specifications to guide code generation. No
systematic study has tested whether LLMs can
produce full and semantically correct TLA+ spec-
ifications directly from natural language, what er-
ror patterns emerge across model architectures, or
how prompting strategies affect generation qual-
Table 1: Summary of data availability across 205
TLA+ specifications in our dataset.
Category Count Percentage
Both CFG & Comments 92 44.9%
CFG only 14 6.8%
Comments only 95 46.3%
Neither 4 2.0%
Has CFG 106 51.7%
Has Comments 187 91.2%
ity. This work provides the first systematic evalu-
ation of 30 LLMs across four prompting strategies
to measure both syntactic and semantic correct-
ness of TLA+ specifications generated from natu-
ral language. We use the TLA+ Examples repos-
itory, which contains both formal specifications
and natural language comments, to test which
model architectures and prompting approaches
produce correct specifications and what types of
errors and hallucinations occur.
3 Methods
Research Questions. This research addresses
four questions. RQ1: Can LLMs generate TLA+
specifications that are both syntactically correct
and semantically faithful? RQ2: What er-
ror types do different LLM backends produce?
RQ3: How do quality metrics correlate with
model architecture, size, and fine-tuning? RQ4:
What failure modes and hallucinations recur in
LLM-generated specifications? Formal validation
(SANY and TLC) answers RQ1, RQ2, and RQ4.
Textual similarity metrics answer RQ3. No public
dataset supports systematic evaluation of LLM-
generated TLA+ specifications across these four
dimensions. We curate a benchmark of 205 spec-
ifications to enable this evaluation.
Dataset. We use the TLA+ Examples
repository maintained by the TLA+ Founda-
tion (TLA+ Community, 2025), the primary
source of community-written formal specifica-
tions. The repository contains 205 TLA+ spec-
ifications across 98 projects: distributed con-
sensus protocols (Paxos, Raft, Byzantine agree-
ment), concurrency problems (dining philoso-
phers, readers-writers), and algorithmic puzzles
(Towers of Hanoi, N-queens). Each specification
includes a .tla file, extracted natural language
comments used as LLM input, and an optional
TLC configuration file. We split the dataset into
training (143 specifications, 70%), validation (31
specifications, 15%), and test (31 specifications,
15%) sets stratified by project. The training set
Table 2: Token usage by model family. R-Tok: rea-
soning tokens; T-Tok: total tokens; R%: reasoning
percentage; Ctx: context window size.
Family Model R-Tok T-Tok R% Ctx
Proprietary Models
GPT gpt-5 3.8K 24K 15.8 128K
gpt-4o 338 19K 1.8 128K
Claude sonnet-4-5 338 28K 1.2 200K
haiku-4-5 338 17K 2.0 200K
opus-4-1 338 31K 1.1 200K
Open-Weight Models
DeepSeek r1:70b 364 12K 3.0 131K
r1:32b 338 12K 2.8 131K
r1:14b 338 9.5K 3.5 131K
r1:8b 390 6.7K 5.8 131K
r1:7b 364 25K 1.4 131K
r1:1.5b 390 1141.7K 0.03 131K
coder:6.7b 390 754K 0.05 4K
coder:latest 286 289K 0.1 128K
LLaMA 3.3:70b 338 20K 1.7 128K
3.3:latest 338 20K 1.7 128K
3.1:8b 416 17K 2.4 128K
3:latest 338 16K 2.1 8K
3.2:1b 390 771K 0.05 131K
Qwen 3:235b 520 18K 2.9 32K
3:30b 520 17K 3.1 32K
3:4b 390 4.1K 9.6
QwQ latest 390 110K 0.4 32K
Phi 4-mini:latest 416 1.3K 31.7 128K
Granite 3.3:latest 390 42K 0.9 131K
GPT-OSS 20b 520 20K 2.6 128K
120b 390 19K 2.1 128K
CodeLLaMA latest 416 56K 0.8 16K
Mistral latest 390 61K 0.6 32K
Gemma 2b 520 29K 1.8 8K
Starling-LM latest 364 38K 1.0 8K
provides few-shot examples. The validation set
is used for model hyperparameters and prompt
tuning. The test set originally contains 31 speci-
fications. Five were excluded because they lacked
either natural language comments or a TLC con-
figuration file, both required for evaluation. This
yields 26 usable test specifications. Table 1 shows
the distribution across splits.
LLM Models. We test 30 LLMs from eight
model families (Table 2). The core evaluation
uses 25 open-weight models across all four strate-
gies via the Ollama runtime (Ollama Contrib-
utors, 2024): ❶ DeepSeek (R1 series: 1.5B–
70B; Coder) (Guo et al., 2025); ❷ LLaMA
(3.x: 1B–70B) (Grattafiori et al., 2024);
❸ Code-specialized (CodeLLaMA, DeepSeek-
Coder, Granite); ❹ Reasoning-focused (QwQ,
Qwen3 (Yang et al., 2025)); ❺ Instruction-tuned
(Mistral (Jiang et al., 2023), Starling-LM, Phi-
4 (Microsoft et al., 2025), Gemma 2b (Gemma
Team et al., 2024)). We also test five proprietary
models with few-shot (k = 3) prompting: GPT-
5 (Singh et al., 2025), GPT-4o (OpenAI et al.,
2024), Claude Opus 4.1 (Anthropic, 2023), Son-
net 4.5 (Anthropic, 2025), and Haiku 4.5. We
focus the multi-strategy analysis on open-weight
models for two reasons. First, open-weight mod-
els support full reproducibility. Second, formal
specifications often describe sensitive system de-
signs better suited to local generation.
Prompting Strategies. We test four prompt-
ing strategies adapted from common LLM code-
generation settings for TLA+ specification gen-
eration/synthesis. Each approach tests different
LLM capabilities.
1 Few-Shot Prompting: This strategy tests
few-shot in-context learning with k = 3 examples
from the training set. Each example pairs natural
language comments with the corresponding com-
plete TLA+ specification. The prompt consists of
a system instruction, the few-shot examples, and
the target comments. The LLM generates the full
specification in one step.
2 Progressive Prompting: This strategy also
uses k = 3 few-shot examples but adds instruc-
tional messages that guide the model through the
specification structure. Each message focuses on
a different aspect: module declaration, state vari-
ables, operators, and temporal properties. This
structure prepares the model before it receives the
target comments.
3 Fill-in-Middle: This code completion strat-
egy splits the ground-truth specification into
three parts: prefix (first 30%), middle (40%), and
suffix (last 30%). The prompt provides the prefix
and suffix with a <FILL> marker. The LLM gen-
erates the missing middle portion. No few-shot
examples are used. The surrounding context pro-
vides sufficient syntactic and semantic informa-
tion for the model.
4 Half Completion: This strategy provides the
first 50% of the specification as a prefix along with
comments and configuration. The LLM generates
the remaining 50%. No few-shot examples are
used. The prefix provides sufficient syntactic and
semantic context for the model.
Evaluation Metrics. We use two metric groups
with different roles. Formal validation measures
correctness of the generated specification as an
executable formal artifact. Textual similarity
measures closeness to the reference sequence.
Formal Validation: Each specification passes
through two stages. ❶ SANY checks syntax with
a 30-second timeout and passes if it reports “Pars-
ing completed.” ❷ Specifications that pass SANY
are submitted to TLC with their configuration file
for model checking.
Textual Similarity: For completion strategies
(HC and FIM), a well-defined target region exists
for direct comparison. We compute BLEU (Pa-
pineni et al., 2002), ROUGE-L (Lin, 2004), edit
Table 3: Cross-method comparison of SANY, TLC
and textual similarity results across four prompting
strategies (25 open-weight models × 26 specifications
= 650 runs per strategy). Only Progressive achieves
TLC passes (8.6%). Per-model breakdowns appear
in Table 5; per-specification breakdowns in Table 10.
†TLC pass rates of 0% reflect failures (i.e., State Di-
rectory Conflicts and Module Not Found errors).
FS HC Prog. FIM
Syntax Validation (SANY)
Passed / Total 173 / 650 80 / 650 162 / 650 92 / 650
Pass Rate (%) 26.6 12.2 24.9 14.2
Best Model (%) qwen3:30b (92.3) qwq (34.6) qwen3:235b (80.8) qwq (38.5)
Specs at 0% 0 / 26 6 / 26 0 / 26 9 / 26
Model Checking (TLC)
Passed / Total 0 / 650 0 / 650 56 / 650 0 / 650
Pass Rate (%) 0.0 0.0 8.6 0.0
Best Model (%) N/A N/A r1:8b (53.8) N/A
Specs at 0% 26 / 26 26 / 26 1 / 26 26 / 26
Similarity to Ground Truth
Avg BLEU N/A 0.062±0.11 N/A 0.077±0.13
Avg ROUGE-L N/A 0.189±0.19 N/A 0.213±0.20
Avg Line Accuracy (%) N/A 10.90±11.8 N/A 12.27±13.9
Exact Matches N/A 0 / 650 N/A 1 / 650
FS: Few Shot, HC: Half Completion, Prog.: Progressive, FIM: Fill-in-Middle
distance, exact match, and line accuracy against
the ground-truth segment.
4 Results
Table 3 summarizes results across all four
prompting strategies. The headline finding is a
persistent syntax–semantics gap: i.e., even the
best strategy reaches only 26.6% SANY syntax
validity. Only Progressive prompting produces
any model-checking successes (8.6%). Within
those successes, a single 8B reasoning model ac-
counts for a quarter of all passing runs and out-
performs every larger model tested.
4.1 RQ1: Synthesis Correctness
Syntax Validation: FS achieves the highest
SANY pass rate at 26.6% (173/650) followed by
Prog. at 24.9% (162/650). The two completion
strategies lag behind: FIM passes 14.2% (92/650)
and HC passes 12.2% (80/650).
Model Checking: The clearest result is the
near-total failure of semantic validation. Three
of the four strategies (FS, HC, and FIM) achieve
0% TLC pass rates across all 650 runs each. Prog.
is the only exception with 56 of 650 runs (8.6%)
passing TLC model checking (Table 4 details two
categories of TLC failure: infrastructure failures,
i.e., environment-level issues such as unresolved
EXTENDS module paths that prevent TLC from
loading the specification, and genuine semantic
failures caused by invalid model-checking con-
figuration in the LLM output). The 56 passes
are concentrated in just three model families.
DeepSeek r1:8b is the strongest performer with
14 out of 26 TLC passes (53.8%) under Prog.,
a result that accounts for 25% of all successful
runs. Qwen3:235b and Qwen3:30b follow with
12 and 10 TLC passes, respectively, together
contributing another 39% of all Prog. successes.
GPT-OSS-20b achieves 6 and Phi-4-mini 4 TLC
passes; every other model achieves 3 or fewer.
The Qwen family’s TLC success under Prog. con-
trasts sharply with its 0 TLC passes under FS
despite its highest FS syntax rates. We examine
this pattern in RQ3. Best Models per Strat-
egy: Table 5 shows two contrasting performance
profiles. For full-generation strategies (FS and
Prog.), large general-purpose models dominate:
Qwen3:30b (23/26) and Qwen3:235b (22/26) lead
FS SANY, while Qwen3:235b (21/26) and both
LLaMA 3.3 variants (18/26) lead Prog. SANY.
For completion strategies, the picture inverts:
QwQ achieves only 3/26 on FS yet leads both
HC (9/26) and FIM (10/26). Chain-of-thought
reasoning appears to help context-sensitive com-
pletion, but hurts open-ended generation.
Proprietary Models (FS k = 3 Only): In
a companion experiment, we test five propri-
etary models with FS (k = 3) as the sole strat-
egy. The detailed results are shown in Table 6.
Among these, GPT-5 is the strongest overall: it
passes SANY on all 26 specifications (100%) and
achieves 7/26 TLC passes (26.9%), the highest
TLC rate for any model under FS. Claude Son-
net 4.5 and Haiku 4.5 pass SANY at 92% (24/26)
and 81% (21/26) respectively and each achieves
3/26 TLC passes (11.5%). Claude Opus 4.1
passes 23/26 SANY (88.5%) with 1 TLC pass.
GPT-4o passes 20/26 SANY (77%) with 1 TLC
pass. These proprietary models were not included
in the multi-strategy analysis due to reproducibil-
ity constraints and because formal specifications
often describe sensitive system designs. Even the
best frontier model (GPT-5) achieves 26.9% TLC
success under FS, comparable to the open-weight
FS ceiling. Syntax–semantics gap persists at fron-
tier scale under the same prompting conditions.
Textual Similarity: For the completion strate-
gies where ground-truth comparison is possible,
similarity to the reference is low as shown in Ta-
bles 7 and 8. HC achieves a mean BLEU of 0.062
(±0.11) and ROUGE-L of 0.189 (±0.19) while
FIM scores slightly higher with BLEU of 0.077
(±0.13) and ROUGE-L of 0.213 (±0.20). Line
accuracy averages 10.90% for HC and 12.27% for
FIM; exact matches are nearly absent (0/650 for
Table 4: TLC model checking infrastructure failure analysis across prompting strategies. Progressive shows 122
TLC failures plus 455 parsing errors (577 total failures, 56 passes, 8.6% pass rate). FS, HC and FIM each had
650 TLC failures (0% pass rate). The table decomposes TLC failures into infrastructure artifacts (Module Not
Found) and genuine semantic failures (Config File Exception: 18 FS, 5 HC, 110 Prog., 5 FIM).
Category Type FS HC Prog. FIM Description
Module Not Found Infra. 159 (24.5%) 159 (24.5%) 12 (9.8%) 156 (24.0%) Unresolved EXTENDS module paths
Config File Exception Genuine 18 (2.8%) 5 (0.8%) 110 (90.2%) 5 (0.8%) Invalid .cfg for TLC
Total 650 650 122 650
Infrastructure % 97.2% 99.2% 9.8% 99.2%
Table 5: Per-model SANY (S) and TLC (T) pass
counts across all four prompting strategies (25 open-
weight models, 26 specifications each). Detailed per-
model per-specification breakdowns are provided in
supplementary materials.
FS HC Prog. FIM
Family Model S T S T S T S T
r1:70b 0 0 3 0 3 0 3 0
r1:32b 6 0 2 0 8 1 6 0
r1:14b 1 0 2 0 2 1 2 0
DeepSeek r1:8b 22 0 1 0 15 14 3 0
r1:7b 1 0 4 0 6 1 3 0
r1:1.5b 1 0 2 0 8 3 2 0
coder:6.7b 1 0 1 0 0 0 2 0
coder:latest 0 0 5 0 3 1 5 0
3.3:70b 11 0 2 0 18 0 6 0
3.3:latest 11 0 1 0 18 0 5 0
LLaMA 3.1:8b 6 0 3 0 3 0 4 0
3:latest 3 0 3 0 2 0 5 0
3.2:1b 2 0 2 0 12 0 4 0
3:235b 22 0 3 0 21 12 5 0
Qwen 3:30b 23 0 3 0 11 10 5 0
3:4b 5 0 3 0 1 1 2 0
QwQ latest 3 0 9 0 2 0 10 0
Phi 4-
mini:latest
5 0 4 0 4 4 1 0
Granite 3.3:latest 3 0 5 0 2 0 2 0
GPT-OSS 120b 17 0 4 0 6 1 3 0
20b 18 0 3 0 9 6 5 0
CodeLLaMA latest 3 0 5 0 2 0 3 0
Mistral latest 3 0 2 0 3 0 3 0
Gemma 2b 6 0 4 0 2 1 2 0
Starling-LM latest 0 0 2 0 1 0 1 0
Average 6.920.00 3.170.00 6.382.08 3.71 0.00
FS: Few Shot, HC: Half Completion, Prog.: Progressive, FIM: Fill-in-Middle
Table 6: SANY and TLC performance for five propri-
etary LLMs under Full-Generation (FS) prompting.
GPT-5 achieves the highest TLC rate at 26.9%, con-
firming that the syntax–semantics gap persists even
at frontier model scale.
Model SANY TLC
Pass % Pass %
GPT-5 26/26 100.0 7/26 26.9
Claude Sonnet 4.5 24/26 92.3 3/26 11.5
Claude Haiku 4.5 21/26 80.8 3/26 11.5
Claude Opus 4.1 23/26 88.5 1/26 3.8
GPT-4o 20/26 76.9 1/26 3.8
HC, 1/650 for FIM). These low scores follow di-
rectly from the length miscalibration documented
in RQ4: models that generate 3–9x the expected
output length will diverge from ground-truth n-
gram patterns regardless of semantic quality.
Table 7: Per-model textual similarity metrics for
Half-Completion (HC) prompting (mean±SD across
26 specifications). Includes BLEU, ROUGE, edit dis-
tance, exact match and line accuracy.
Family Model E Acc (%) Similarity BLEU ROUGE-L
r1:70b 0 14.69±11.6 0.154±0.21 0.107±0.17 0.278±0.21
r1:32b 0 15.43±13.2 0.175±0.17 0.120±0.16 0.319±0.19
r1:14b 0 11.67±8.7 0.083±0.06 0.050±0.07 0.227±0.14
DeepSeek r1:8b 0 4.42±6.1 0.008±0.02 0.002±0.01 0.030±0.08
r1:7b 0 12.62±11.3 0.081±0.08 0.056±0.07 0.215±0.14
r1:1.5b 0 6.05±8.0 0.037±0.05 0.020±0.04 0.086±0.13
coder:6.7b 0 10.48±14.2 0.062±0.10 0.032±0.06 0.163±0.15
coder:latest 0 10.80±11.7 0.044±0.04 0.014±0.03 0.109±0.10
3.3:70b 0 17.67±13.9 0.181±0.17 0.158±0.18 0.366±0.20
3.3:latest 0 17.51±13.5 0.166±0.17 0.161±0.18 0.361±0.20
LLaMA 3.1:8b 0 9.05±10.7 0.106±0.11 0.083±0.11 0.232±0.16
3:latest 0 12.07±9.9 0.115±0.12 0.064±0.09 0.265±0.16
3.2:1b 0 6.72±8.7 0.053±0.07 0.043±0.08 0.140±0.14
3:235b 0 5.18±10.3 0.061±0.14 0.014±0.05 0.093±0.19
Qwen 3:30b 0 3.58±4.8 0.041±0.15 0.023±0.08 0.058±0.21
3:4b 0 8.24±9.2 0.127±0.15 0.067±0.15 0.217±0.21
QwQ latest 0 6.06±9.2 0.037±0.05 0.018±0.02 0.066±0.04
Phi 4-
mini:latest
0 1.54±4.2 0.015±0.03 0.003±0.01 0.034±0.06
Granite 3.3:latest 0 15.30±12.6 0.071±0.07 0.055±0.10 0.192±0.15
GPT-OSS 120b 0 18.39±14.1 0.195±0.17 0.151±0.18 0.362±0.22
20b 0 11.12±12.9 0.098±0.13 0.052±0.10 0.197±0.22
CodeLLaMA latest 0 8.84±11.2 0.057±0.11 0.038±0.09 0.121±0.16
Mistral latest 0 14.15±12.3 0.068±0.06 0.084±0.11 0.213±0.17
Gemma 2b 0 15.23±11.5 0.073±0.06 0.067±0.09 0.194±0.11
Starling-LM latest 0 15.73±12.3 0.083±0.09 0.070±0.12 0.188±0.13
Total – 0 10.90±10.6 0.088±0.10 0.062±0.09 0.189±0.15
RQ1 Summary
LLMs produce syntactically valid TLA+ in up
to 26.6% of attempts, but semantic correctness
is rare (8.6% TLC under Prog.). Only Prog.
achieves any TLC passes, led by DeepSeek r1:8b
(14/26) and Qwen3 models (10–12 TLC each).
Completion strategies show low textual similar-
ity (BLEU < 0.08 and ROUGE-L < 0.22).
4.2 RQ2: Synthesis Errors
Dominant Error Categories: Table 9 (Panel-
A) shows the distribution of SANY errors across
the four strategies. The most common error cate-
gory is Parse: Bad Module Body, which accounts
for roughly half of all failures across every strat-
egy (47.8% FS, 46.3% HC, 49.8% Prog., 52.0%
FIM). This error arises when the parser encoun-
ters invalid syntax within the module body: mal-
formed operator definitions, structurally broken
expressions, or injected non-TLA+ syntax. This
Table 8: Per-model textual similarity metrics for Fill-
in-the-Middle (FIM) prompting (mean±SD across 26
specifications). Includes BLEU, ROUGE, edit dis-
tance, exact match and line accuracy.
Family Model E Acc (%) Similarity BLEU ROUGE-L
r1:70b 1 18.59±21.2 0.151±0.19 0.159±0.21 0.348±0.22
r1:32b 0 19.02±20.0 0.195±0.18 0.145±0.17 0.371±0.19
r1:14b 0 12.19±14.2 0.152±0.18 0.076±0.13 0.270±0.20
DeepSeek r1:8b 0 6.04±10.0 0.028±0.05 0.017±0.03 0.048±0.07
r1:7b 0 8.06±9.4 0.128±0.12 0.053±0.08 0.217±0.12
r1:1.5b 0 7.89±11.0 0.064±0.09 0.027±0.05 0.131±0.16
coder:6.7b 0 13.33±17.5 0.076±0.11 0.044±0.07 0.209±0.19
coder:latest 0 9.55±12.8 0.075±0.09 0.023±0.04 0.171±0.15
3.3:70b 0 22.39±17.1 0.194±0.16 0.204±0.19 0.377±0.17
3.3:latest 0 21.70±18.4 0.185±0.17 0.195±0.19 0.367±0.18
LLaMA 3.1:8b 0 9.55±11.9 0.132±0.13 0.121±0.15 0.278±0.16
3:latest 0 10.68±11.3 0.142±0.14 0.091±0.13 0.285±0.15
3.2:1b 0 3.82±6.0 0.075±0.09 0.026±0.05 0.135±0.14
3:235b 0 9.62±13.1 0.054±0.09 0.054±0.08 0.087±0.14
Qwen 3:30b 0 5.37±6.8 0.033±0.06 0.013±0.02 0.049±0.08
3:4b 0 8.27±11.7 0.149±0.16 0.021±0.04 0.191±0.16
QwQ latest 0 14.66±19.5 0.030±0.05 0.020±0.04 0.070±0.08
Phi 4-
mini:latest
0 2.20±6.4 0.023±0.05 0.001±0.00 0.025±0.07
Granite 3.3:latest 0 14.45±13.5 0.130±0.13 0.103±0.14 0.285±0.17
GPT-OSS 120b 0 24.16±21.0 0.240±0.24 0.212±0.25 0.392±0.25
20b 0 8.46±12.7 0.065±0.10 0.033±0.06 0.089±0.15
CodeLLaMA latest 0 16.84±17.5 0.052±0.09 0.063±0.12 0.191±0.19
Mistral latest 0 16.41±16.3 0.109±0.11 0.113±0.14 0.285±0.17
Gemma 2b 0 10.09±10.6 0.090±0.10 0.038±0.07 0.219±0.14
Starling-LM latest 0 15.88±15.8 0.093±0.10 0.062±0.09 0.242±0.16
Average 0.04 12.37±13.83 0.107±0.119 0.077±0.102 0.213±0.154
Table 9: SANY error taxonomy across four prompt-
ing strategies. Panel A: distribution of error types
among failed runs. Panel B: location of the first error
in the generated file.
Panel A: Error Type Distribution
Error Type Method n %
Parse: Bad Module Body FS 228 47.8
HC 264 46.3
Prog. 243 49.8
FIM 290 52.0
Parse: Other FS 72 15.1
HC 227 39.8
Prog. 70 14.3
FIM 191 34.2
Parse: Unexpected Token FS 82 17.2
HC 29 5.1
Prog. 65 13.3
FIM 17 3.0
Panel B: First Error Location
Location Method n %
Early (first third) FS 240 50.3
HC 63 11.1
Prog. 220 45.1
FIM 81 14.5
Mid (middle third) FS 196 41.1
HC 261 45.8
Prog. 209 42.8
FIM 310 55.6
Late (final third) FS 41 8.6
HC 246 43.2
Prog. 59 12.1
FIM 167 29.9
pattern remains constant across strategies, which
indicates a core limitation in LLM understanding
of TLA+ grammar rather than prompt design.
Strategy-Specific Error Patterns: The sec-
ond most common error type differs by strategy
class. FS and Prog. produce relatively more Un-
expected Token errors (17.2% and 13.3%), which
arise when the parser encounters a token that vio-
lates the expected grammar. In contrast, HC and
FIM produce more Parse: Other errors (39.8%
and 34.2%), which capture miscellaneous parse
failures such as mismatched delimiters and incom-
plete expressions. Full-generation models must
construct correct operator syntax from scratch
and tend to fail with specific illegal tokens. Com-
pletion models receive valid syntactic context and
fail with structural mismatches deeper in the file.
Lexical Errors: Lexical errors account for less
than 10% of SANY-detected failures but re-
veal systematic cross-language contamination.
Unicode operator substitution (replacing TLA+
ASCII operators with LATEX/math equivalents)
appears in 7.1% of FS and 8.8% of Prog. failures.
Backtick errors from Markdown code-fence leak-
age appear in 5.9% of FS failures. Semicolons,
absent from TLA+ but ubiquitous in C, Java,
and Python, appear as SANY first errors in 3.8%
of FS and 5.1% of Prog. failures. A broader file
scan reveals higher ambient contamination rates
of 7.5% for FS and 12.6% for Prog. These lexical
hallucinations are discussed in full under RQ4.
Error Location: Panel B of Table 9 shows where
the first error occurs in the generated file, di-
vided into early (first third), middle, and late (fi-
nal third) segments. FS and Prog. exhibit early-
heavy error distributions: 50.3% and 45.1% of
errors occur in the first third of the file. HC
and FIM show the opposite pattern: errors con-
centrate in the middle and late portions (88.9%
and 85.5% combined). Full-generation strategies
must produce a valid TLA+ module header from
scratch and fail immediately when the preamble
is wrong. Completion strategies receive a correct
prefix and only diverge further into the body.
RQ2 Summary
“Parse: Bad Module Body” accounts for roughly
half of all failures across every strategy and
points to a structural failure mode that future
datasets should target. Full-generation strate-
gies fail early with specific illegal tokens; com-
pletion strategies fail mid-to-late with struc-
tural mismatches. Lexical contamination from
non-TLA+ languages (Unicode operators, back-
ticks, semicolons) is systematic and strategy-
independent. Grammar-constrained generation
and deterministic post-processing offer concrete
paths for future improvement.
Table 10: Specification-level SANY and TLC pass
rates averaged across 25 open-weight models for each
of the 26 test specifications. Per-specification similar-
ity metrics and detailed breakdowns are provided in
supplementary materials.
FS HC Prog. FIM
Specification S T S T S T S T
Alternate 44.0 0.0 4.0 0.0 28.0 4.0 0.0 0.0
BinarySearch 12.0 0.0 0.0 0.0 24.0 8.0 0.0 0.0
BufferedRandom
AccessFile 16.0 0.0 0.0 0.0 28.0 12.0 0.0 0.0
ClientCentric 24.0 0.0 16.0 0.0 12.0 8.0 0.0 0.0
Consensus 44.0 0.0 4.0 0.0 36.0 12.0 8.0 0.0
Environment
Controller 24.0 0.0 4.0 0.0 24.0 16.0 4.0 0.0
EWD687a_anim 36.0 0.0 0.0 0.0 28.0 12.0 4.0 0.0
EWD998ChanID
_export 20.0 0.0 0.0 0.0 16.0 8.0 0.0 0.0
Hanoi 4.0 0.0 20.0 0.0 16.0 12.0 4.0 0.0
HDiskSynod 32.0 0.0 0.0 0.0 20.0 4.0 48.0 0.0
KVsnap 28.0 0.0 16.0 0.0 24.0 12.0 0.0 0.0
Majority 20.0 0.0 8.0 0.0 16.0 4.0 8.0 0.0
MajorityProof 32.0 0.0 32.0 0.0 28.0 4.0 12.0 0.0
MCConsensus 36.0 0.0 4.0 0.0 40.0 8.0 8.0 0.0
MCDieHarder 28.0 0.0 4.0 0.0 16.0 0.0 24.0 0.0
MCEWD687a 24.0 0.0 24.0 0.0 32.0 4.0 36.0 0.0
MultiPaxos 24.0 0.0 8.0 0.0 32.0 16.0 0.0 0.0
MultiPaxos_MC 36.0 0.0 36.0 0.0 36.0 4.0 8.0 0.0
nbacc_ray97 32.0 0.0 0.0 0.0 20.0 12.0 36.0 0.0
Relation 28.0 0.0 4.0 0.0 20.0 4.0 8.0 0.0
SimpleRegular 24.0 0.0 8.0 0.0 32.0 12.0 44.0 0.0
TLAPS 20.0 0.0 12.0 0.0 24.0 4.0 4.0 0.0
TwoPhase 24.0 0.0 8.0 0.0 20.0 8.0 12.0 0.0
VoucherCancel 24.0 0.0 4.0 0.0 28.0 16.0 0.0 0.0
VoucherTransfer 32.0 0.0 4.0 0.0 24.0 4.0 0.0 0.0
YoYoNoPruning 24.0 0.0 100.0 0.0 24.0 16.0 100.0 0.0
Average 26.6 0.0 12.2 0.0 24.9 8.6 14.2 0.0
4.3 RQ3: Model Impact
Model Family Analysis: The per-model results
in Table 5 reveal that model family predicts which
strategy a model excels at more than it predicts
overall quality. The Qwen3 family dominates
syntax generation under full-generation strate-
gies but achieves no TLC passes under FS de-
spite leading the SANY rankings. High syntac-
tic fluency does not translate to semantic cor-
rectness without the structured guidance of Prog.
The DeepSeek R1 series presents the most strik-
ing result: the 8B parameter model achieves the
best TLC result in the entire study (14/26 un-
der Prog.) while the 70B variant fails to pro-
duce a single SANY pass under FS. This inver-
sion of scale is not noise: it appears consistently
across metrics and strategies (see Model Size be-
low). The R1 training regime produces reasoning
capabilities that transfer differently at different
scales. The LLaMA 3.3 family performs reliably
under Prog. for syntax (18/26 for both 70b and
latest) but, like Qwen, achieves no TLC passes.
Model Size: Larger models do not consistently
outperform smaller ones, a finding that holds
within three independent model families. Within
DeepSeek R1, the 8B model achieves 14 TLC
passes under Prog. while the 70B model achieves
none. The 8B model also leads on SANY un-
der FS (22/26 vs. 0/26). Within Qwen3, the 30B
variant leads FS SANY (23/26) over the 235B
model (22/26). Within LLaMA, the 1B model
(3.2:1b) achieves the highest Prog. SANY of any
LLaMA variant (12/26), ahead of LLaMA 3.1:8b
(3/26) and LLaMA 3:latest (2/26). For TLA+, a
scarce-corpora formal language, reasoning align-
ment and instruction-following discipline matter
more than raw parameter count.
Code Specialization: Models designed or fine-
tuned for code generation do not outperform
general-purpose models on TLA+. CodeL-
LaMA achieves 0–5 SANY passes per strategy,
DeepSeek-Coder peaks at 5/26 (HC and FIM),
and Granite reaches only 5/26 (HC). General-
purpose models like Qwen3:30b (23/26 FS) and
LLaMA 3.3:70b (18/26 Prog.) far exceed these
code specialists. Models heavily fine-tuned on
Python, C, and Java learn strong priors about
those languages’ syntax that interfere with the
stricter and more distinctive grammar of TLA+.
Per-Specification Difficulty: Table 10 shows
that specification difficulty varies from 0% to
100% SANY pass rates and that the diffi-
culty profile is strategy-specific. Simple state-
machine specifications (e.g., YoYoNoPruning)
achieve 100% SANY under both completion
strategies, while complex distributed protocols
(e.g., HDiskSynod, BinarySearch) score 0%. Un-
der Prog. with TLC, MultiPaxos and Voucher-
Cancel achieve the highest semantic pass rates
(16% each). MCDieHarder achieves 0% TLC
across every strategy.
RQ3 Summary
Standard assumptions about scale and special-
ization do not hold for TLA+. Smaller reasoning
models (DeepSeek r1:8b, LLaMA 3.2:1b) out-
perform larger siblings. Code-specialist models
consistently underperform general-purpose ones
due to negative transfer from mainstream lan-
guage training. Model family determines strat-
egy fit better than overall capability.
4.4 RQ4: Failure Modes
Beyond the error type distribution in RQ2, we
identify five categories of hallucination that recur
across models and strategies (Table 11).
Unicode Operator Substitution: LLMs fre-
quently replace the ASCII operators that TLA+
requires with their Unicode or LATEX equivalents.
Table 11: Hallucination taxonomy across prompting strategies (RQ4). Integer values = file counts; X = observed
but count not isolated; 0 = not observed; n/a = not applicable. †FIM applies clean_llm_response() to strip
markdown before validation. ‡File scan (contains ≥1 semicolon). §SANY first-error count (FS: 18/477; HC:
11/570; Prog. : 25/488; FIM: 10/558). ¶File scan shows higher rates (FS: 49/650 = 7.5%; HC: 23/650 = 3.5%;
Prog. : 82/650 = 12.6%; FIM: 22/650 = 3.4%).
Pattern Source Correct
TLA+
FS HC Prog FIM
Unicode Operator Substitution (Lexical Error)
∨ (U+2228) LATEX/Math \/ X X X X
∧ (U+2227) LATEX/Math /\ X X X X
∈ (U+2208) LATEX/Math \in X X X X
→ (U+2192) LATEX/Math -> X X X X
Cross-Language Syntax Injection
Semicolons (;) C/Java/Py invalid 18§/49¶ 11§/23¶ 25§/82¶ 10§/22¶
Backticks (‘) Markdown invalid 28 10 21 13
END keyword Pascal/SQL ==== 0 0 X 0
Reasoning/Formatting Leakage
<think> blocks CoT reasoning invalid 15 30 n/a 28
Markdown fences Output format invalid 0 0 X 0†
NL prose in body Task confusion invalid 0 0 X 0
Structural Errors
Missing ==== Incomplete gen ==== X X 243+ X
Missing MODULE No preamble - MODULE
X -
0 0 X 0
Duplicate MODULE Repeated start single
header
0 0 X 0
Common substitutions include ∨ (U+2228) for
\/, ∧ (U+2227) for /\, ∈ (U+2208) for \in, and
→ (U+2192) for ->. These substitutions appear
across all four strategies and all model families.
SANY rejects all Unicode operators, which makes
this a consistent source of lexical errors.
Cross-Language Syntax Injection: Models
inject syntax from mainstream languages into
TLA+ output. Semicolons are the most common
example: Prog. shows the highest contamination
rate at 12.6% of files (82/650) followed by FS at
7.5% (49/650), while HC and FIM are lower at
approximately 3.5% each. QwQ is the worst of-
fender with 35% semicolon contamination in FS
and 27% in Prog. Granite and Mistral each reach
31% in Prog. Backtick injection from Markdown
code fences is most common in FS (28 files), with
DeepSeek r1:7b particularly prone (12 out of 26
files). Prog. also shows isolated instances of END
keyword injection from Pascal/SQL syntax.
Reasoning and Formatting Leakage: Mod-
els with chain-of-thought capabilities leak their
internal reasoning into the generated specifica-
tion. QwQ is the worst offender: it injects
<think> blocks into 100% of its HC and FIM
outputs (26/26 files each) and 46.2% of FS out-
puts DeepSeek R1 models (r1:14b and r1:32b)
and Qwen3:235b show lower contamination rates
(3.8–7.7%). The overall contamination rates
are 2.3% for FS, 4.6% for HC, and 4.3% for
FIM. Prog. is not affected because its multi-step
pipeline structure handles reasoning differently.
Beyond thinking blocks, Prog. outputs occasion-
ally contain Markdown fences and natural lan-
guage prose within the TLA+ module body.
Generation Length Miscalibration: Models
exhibit poor calibration between output length
and the expected ground-truth length On aver-
age, HC outputs are 3.46× the length of the
ground-truth target segment and FIM outputs
are 2.24×. The extremes are stark: DeepSeek
coder:latest generates 9.15× the expected HC
length, LLaMA 3.2:1b generates 9.84×, and QwQ
generates 7.71× for HC and 7.75× for FIM. At
the opposite extreme, Qwen3:30b produces near-
empty output for HC (0.00×) and Qwen3:235b
produces only 0.09× the expected length. Only
a few models achieve near-perfect calibration:
LLaMA 3.3:70b (1.16× HC and 1.03× FIM) and
DeepSeek r1:7b (0.97× HC).
Structural Errors: Prog. produces distinc-
tive structural hallucinations due to its multi-
step generation process. The most common is
the missing ==== terminator: over 243 Prog.-
generated files lack the four-equals-sign module
terminator that TLA+ requires. Models also oc-
casionally produce duplicate MODULE headers or
omit the MODULE header entirely when the focus
is on body content from a particular step. These
structural errors are largely absent from the other
strategies where generation occurs in one pass.
RQ4 Summary
Five systematic hallucination categories recur
across all models: (1) Unicode operator sub-
stitution; (2) cross-language syntax injection
(semicolons in 12.6% of files, backticks in 5.9%);
(3) reasoning token leakage (<think> blocks
in QwQ); (4) generation length miscalibra-
tion (0.00–9.84× ground truth); and (5) struc-
tural errors (243+ missing ==== terminators).
Each category is traceable to a specific training
data bias and addressable by better datasets,
grammar-specific decoding/post-processing.
5 Discussion
This section discusses the results and practical
implications for TLA+ LLM-based synthesis.
The Syntax-Semantics Gap Is Structural,
Not Incidental: The wide gap between syntac-
tic validity (up to 26.6% SANY) and semantic
correctness (8.6% TLC) persists across all four
strategies and all model scales. It reflects a struc-
tural property: LLMs acquire surface-level token
patterns of TLA+ syntax without internalizing
the modal and set-theoretic semantics those to-
kens encode. A TLA+ module that parses is only
a skeleton. Hahn et al. show that LLM perfor-
mance degrades as target formalism complexity
increases from regular expressions through first-
order logic to LTL (Hahn et al., 2022); TLA+,
combining TLA with set theory, sits at the harder
end of that spectrum. Ferrari and Spoletini frame
LLMs and formal methods as mutually reinforc-
ing: LLMs accelerate specification drafting while
formal tools provide correctness guarantees (Fer-
rari and Spoletini, 2025). Our results illustrate
why both directions are needed. The Qwen3
family makes the gap most concrete: Qwen3:30b
achieves the highest FS SANY rate (23/26) yet
produces zero TLC passes under the same strat-
egy. The results establish that closing the syntax-
semantics gap is the primary challenge for future
TLA+ work. Fine-tuning on the 205 TLA+ Foun-
dation specifications could improve semantic un-
derstanding, and iterative refinement with TLC
feedback could help toward valid synthesis.
Progressive Prompting as Cognitive Scaf-
folding: Progressive prompting is the only strat-
egy that achieves TLC passes (8.6%) and its
SANY rate (24.9%) is competitive with FS
(26.6%), making it the dominant choice over-
all. Its advantage lies in decomposing the genera-
tion task into sequential concerns: module struc-
ture, variable declarations, initial predicate, next-
state relation, temporal property. This matches
the insight behind chain-of-thought prompting,
where externalising intermediate reasoning steps
improves performance on tasks requiring multi-
hop inference (Wei et al., 2022; Li et al., 2023).
Correct TLA+ synthesis must simultaneously sat-
isfy module syntax, type consistency, behavioral
semantics and temporal logic. Kogler et al. in-
dependently found that iterative LLM prompt-
ing with post-processing achieves syntactic valid-
ity in formal railway specifications (Kogler et al.,
2024), and the Specula system’s architecture con-
firms that multi-component pipelines substan-
tially outperform single-pass generation (Cheng
et al., 2025). The cost of progressive prompt-
ing is a new class of structural errors (duplicate
headers, missing terminators) that do not arise
in single-pass strategies. This observation sug-
gests that future systems should pair progressive
decomposition with a lightweight global consis-
tency check. The baseline 8.6% TLC success un-
der progressive prompting sets a foundation that
future work with fine-tuned models and improved
datasets could substantially exceed through pro-
gressive decomposition and semantic feedback.
Model Scale and the Scarce-corpora For-
mal Language Problem.: The inversion of
scale, where DeepSeek r1:8b outperforms r1:70b
and Qwen3:30b matches Qwen3:235b, recurs
across families and strategies and is not noise.
TLA+ is a scarce-corpora formal language: its
entire public corpus amounts to hundreds of
modules, dwarfed by billions of tokens of C,
Python and Java. Larger models have propor-
tionally more capacity devoted to high-resource
languages, which manifests as stronger prior pulls
toward mainstream syntax. Smaller reasoning-
distilled models in the DeepSeek R1 series were
trained with reinforcement learning on step-by-
step symbolic reasoning traces (Guo et al., 2025);
at 8B parameters the model reasons over TLA+
structure without the noise of the larger conflict-
ing training pool. Mirzadeh et al. show that
LLMs replicate reasoning steps from training data
rather than performing genuine logical reason-
ing (Mirzadeh et al., 2025). A model whose rea-
soning depends on training distribution will be
most affected when that distribution is sparse,
as it is for TLA+. For practitioners, this base-
line guidance means that the largest available
model is not necessarily the right choice: smaller,
reasoning-oriented models may be more reliable
and substantially cheaper.
Negative Transfer from Code Special-
ization: Code-specialist models (CodeLLaMA,
DeepSeek-Coder, Granite) consistently under-
perform general-purpose counterparts across all
strategies. The mechanism is negative trans-
fer: fine-tuning on C, Java and Python trains
the model to associate token sequences (semi-
colons, curly braces, return) with correct code,
and these associations are activated when gener-
ating TLA+. This directly accounts for two of
our five hallucination categories: semicolon in-
jection and backtick fence insertion. Beg et al.
identify the mismatch between target formal lan-
guages and pre-training distributions as a prin-
cipal challenge for LLM-based formal require-
ments (Beg et al., 2025). When selecting a model
for TLA+ generation, general-purpose reasoning-
oriented models retain the linguistic flexibility
needed for TLA+’s unusual operator vocabulary
and set-theoretic structure.
Hallucinations Are Systematic and Mitiga-
ble: All five hallucination categories are trace-
able to specific training distributions rather than
to the TLA+ generation task itself. Unicode op-
erator substitution arises because documentation
renders operators in mathematical Unicode (∈,
∧) while SANY requires ASCII (\in, /\). Semi-
colon and backtick injection arise from C-family
and Markdown-dominated pre-training corpora.
Reasoning token leakage is a side effect of chain-
of-thought fine-tuning. Generation length mis-
calibration (3.46× for HC, 2.24× for FIM, with
outliers at 9.84×) directly explains the near-zero
textual similarity scores. Phantom operator in-
vention parallels the package hallucination phe-
nomenon (Spracklen et al., 2025): models gen-
erate identifiers statistically consistent with con-
text that do not exist in the target environ-
ment. The systematic and predictable nature of
these errors is itself an opportunity: grammar-
constrained decoding (Helwer, 2025) can enforce
TLA+ syntax at the token level, determinis-
tic post-processing (Unicode normalization, semi-
colon stripping, fence removal) can eliminate a
large fraction of SANY failures before the parser
is invoked, and retrieval-augmented generation
over verified TLA+ specifications can ground op-
erator usage in concrete examples.
Implications for Practice: Progressive
prompting achieves semantically valid speci-
fications. Reasoning-oriented is favored over
general-purpose models and code specialists.
Deterministic post-processing (with augmented
retrieval) before SANY could help with TLA+
synthesis, e.g., Unicode normalization, fence
removal, and reasoning token stripping to elimi-
nate the majority of preventable syntax failures
at zero cost. SANY pass rate can be only a base,
as semantic validity requires model checking.
Limitations: We run each model-specification
pair once per strategy, so stochastic variation
is not fully captured; we address this through
scale (650 runs per strategy, 2,600 core runs to-
tal) with temperature and sampling parameters
held constant across all models. Our 26 test
specifications cover distributed protocols, concur-
rency problems, and algorithmic puzzles from the
TLA+ community repository, but may not repre-
sent the full range of industrial use cases, as pro-
prietary specifications from organizations such as
Amazon and Microsoft are not publicly available.
This study focuses on TLA+, and the results may
not transfer directly to other formal languages.
Finally, the natural language inputs vary in com-
ment density across specifications, and TLC vali-
dation is bounded to finite-state instances, so er-
rors that appear only with larger parameter val-
ues remain undetected.
6 Conclusion
We present the first systematic evaluation of
LLM-based TLA+ specification synthesis from
natural language. Across 30 models, eight fam-
ilies, 26 test specifications, and more than 2,700
runs, results show a clear syntax-semantics gap.
LLMs reach 26.6% SANY validity, yet only 8.6%
pass TLC, and only under progressive prompt-
ing. DeepSeek r1:8b outperforms its 70B counter-
part. Code-specialized models also trail general-
purpose models. Through our experiments, we
identify five recurring hallucination categories:
Unicode operator substitution, cross-language
syntax injection, reasoning leakage, length mis-
calibration, and structural errors. These results
show that current LLMs do not yet produce re-
liable TLA+ specifications rigorous review by a
human. Future work should target three chal-
lenges in this area: (1) iterative repair/feedback
loop with SANY and TLC to address the syntax-
semantics gap; (2) multi-step prompting with
global consistency checks to mitigate lexical and
structural errors in progressive generation; and
(3) fine-tuning on TLA+ data with grammar-
constrained decoding to reduce negative transfer
and systematic hallucinations.
REFERENCES
Anthropic (2023). Introducing claude: A next-
generation ai assistant. Anthropic Blog.
Anthropic (2025). Claude sonnet 4.5: Anthropic’s
most capable model for coding, agents, and com-
puter use. Anthropic Blog.
Austin, J., Odena, A., Nye, M., Bosma, M.,
Michalewski, H., Dohan, D., Jiang, E., Cai, C.,
Terry, M., Le, Q., et al. (2021). Program synthe-
sis with large language models. arXiv preprint
arXiv:2108.07732.
Beg, A., O’Donoghue, D., and Monahan, R. (2025).
Leveraging llms for formal software requirements
– challenges and prospects.
Cheng, Q., Tang, R., Ma, E., Hackett, F., He, P., Su,
Y., Beschastnikh, I., Huang, Y., Ma, X., and
Xu, T. (2026). Sysmobench: Evaluating ai on
formally modeling complex real-world systems.
Cheng, Q., Xu, T., and Huang, Y. (2025). Spec-
ula: First place, genai-accelerated tla+ challenge
(code → spec). TLA+ Foundation.
Cirstea, H., Kuppe, M. A., Loillier, B., and Merz,
S. (2024). Validating traces of distributed pro-
grams against tla+ specifications.
Ferrari, A. and Spoletini, P. (2025). Formal require-
ments engineering and large language models:
A two-way roadmap. Information and Software
Technology, 181:107697.
Gemma Team, Mesnard, T., Hardin, C., Dadashi, R.,
Bhupatiraju, S., et al. (2024). Gemma: Open
models based on gemini research and technology.
Grattafiori, A. et al. (2024). The llama 3 herd of
models.
Guo, D., Yang, D., Zhang, H., Song, J., Wang, P.,
Zhu, Q., Xu, R., Zhang, R., Ma, S., Bi, X.,
et al. (2025). Deepseek-r1 incentivizes reasoning
in llms through reinforcement learning. Nature,
645:633–638.
Hahn, C., Schmitt, F., Tillman, J. J., Metzger, N.,
Siber, J., and Finkbeiner, B. (2022). Formal
specifications from natural language.
Helwer, A. (2025). Second place, genai-accelerated
tla+ challenge. TLA+ Foundation.
Jiang, A. Q. et al. (2023). Mistral 7b.
Kogler, P., Falkner, A., and Sperl, S. (2024). Re-
liable generation of formal specifications using
large language models. In SE 2024 - Compan-
ion, pages 141–153. Gesellschaft für Informatik
e.V.
Lamport, L. (2002). Specifying Systems: The TLA+
Language and Tools for Hardware and Software
Engineers. Addison-Wesley Longman Publishing
Co., Inc.
Lamport, L. (2003). Specifying Systems: The TLA+
Language and Tools for Hardware and Software
Engineers. Addison-Wesley.
Li, J., Li, G., Li, Y., and Jin, Z. (2023). Structured
chain-of-thought prompting for code generation.
Lin, C.-Y. (2004). ROUGE: A package for automatic
evaluation of summaries. In Text Summariza-
tion Branches Out, pages 74–81. Association for
Computational Linguistics.
Ma, L., Liu, S., Li, Y., Xie, X., and Bu, L. (2025).
Specgen: Automated generation of formal pro-
gram specifications via large language models.
Microsoft et al. (2025). Phi-4-mini technical re-
port: Compact yet powerful multimodal lan-
guage models via mixture-of-loras.
Mirzadeh, I., Alizadeh, K., Shahrokhi, H., Tuzel,
O., Bengio, S., and Farajtabar, M. (2025).
Gsm-symbolic: Understanding the limitations of
mathematical reasoning in large language mod-
els.
Newcombe, C., Rath, T., Zhang, F., Munteanu, B.,
Brooker, M., and Deardeuff, M. (2015). How
amazon web services uses formal methods. Com-
munications of the ACM.
Ollama Contributors (2024). Ollama: Open source
runtime for local large language models. GitHub
repository.
OpenAI et al. (2024). Gpt-4 technical report.
Papineni, K., Roukos, S., Ward, T., and Zhu, W.-J.
(2002). Bleu: a method for automatic evalua-
tion of machine translation. In Proceedings of
the 40th Annual Meeting of the Association for
Computational Linguistics, pages 311–318. As-
sociation for Computational Linguistics.
Singh, A. et al. (2025). Openai gpt-5 system card.
Spracklen, J., Wijewickrama, R., Sakib, A. H. M. N.,
Maiti, A., Viswanath, B., and Jadliwala, M.
(2025). We have a package for you! a com-
prehensive analysis of package hallucinations by
code generating llms. In Proceedings of the 34th
USENIX Security Symposium.
Terzian, G. (2025). Third place, genai-accelerated
tla+ challenge. TLA+ Foundation.
TLA+ Community (2025). Tla+ examples: A collec-
tion of tla+ specifications and models. GitHub
repository.
TLA+ Foundation (2025). Genai-accelerated tla+
challenge announcement. TLA+ Foundation.
Wei, J., Wang, X., Schuurmans, D., Bosma, M.,
Ichter, B., Xia, F., Chi, E. H., Le, Q. V., and
Zhou, D. (2022). Chain-of-thought prompting
elicits reasoning in large language models.
Yang, A. et al. (2025). Qwen3 technical report.
Yao, J., Tao, R., Gu, R., Nieh, J., Jana, S., and
Ryan, G. (2021). DistAI: Data-Driven auto-
mated invariant learning for distributed proto-
cols. In 15th USENIX Symposium on Operating
Systems Design and Implementation (OSDI 21),
pages 405–421. USENIX Association.
Ye, H., Liu, T., Zhang, A., Hua, W., and Jia, W.
(2023). Cognitive mirage: A review of halluci-
nations in large language models.
Zhang, W. and Zhang, J. (2025). Hallucination mit-
igation for retrieval-augmented large language
models: A review. Mathematics, 13(5):856.