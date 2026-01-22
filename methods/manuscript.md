# Evaluating Input Representation Methods for EHR Foundation Models: A Systematic Benchmark

## Abstract

Electronic Health Record (EHR) foundation models require principled methods for representing continuous physiological measurements within token sequences consumed by transformer architectures. Population-based quantile binning (e.g., deciles) has become a de facto choice in prior pipelines (e.g., ETHOS-ARES [1]), but it imposes an arbitrary categorical structure on intrinsically continuous measurements and may obscure clinically meaningful thresholds defined by reference intervals [5]. We present a systematic benchmark that isolates *input representation* as the primary experimental variable along three axes: (1) quantization granularity and clinical anchoring of bin boundaries using laboratory reference ranges, (2) representation mechanics that either treat binned values as discrete tokens or preserve numeric precision via soft discretization (convex combinations of bin embeddings [14]) and learned continuous encoders inspired by xVal-style numeric embeddings [7], and (3) **vocabulary semantics as a controlled intervention** that swaps only the *code namespace* (native identifiers vs. standardized concepts; CLIF) while holding the underlying event rows, timestamps, and numeric values fixed, with explicit negative controls (randomized mappings and frequency-matched collapses). We evaluate these design choices on MIMIC-IV v3.1 [13] across four prediction tasks (same-admission mortality, long length of stay, ICU admission after 24 hours, invasive mechanical ventilation after 24 hours), reporting discrimination (AUROC/AUPRC), calibration (Brier score/ECE), fairness (subgroup AUROC gaps), and efficiency (token counts, FLOPs, inference latency) in the style of clinically meaningful evaluation frameworks (FoMoH [10]) and benchmark practice (EHRSHOT [11]). Across 100 training runs using a scaled-down LLaMA 3.2 configuration (config overrides: `hidden_size=1024`, `intermediate_size=2048`, `num_hidden_layers=8`, `num_attention_heads=8`; **≈87M parameters**, depending on vocab size; RoPE [12]) and five random seeds per configuration, we will report mean±SD performance and paired statistical comparisons across representation variants. [PLACEHOLDER: Key quantitative findings with specific AUROC/AUPRC and calibration changes]. This benchmark is intended to yield empirically grounded guidelines for representation design in clinical foundation models.

---

## 1. Introduction

### Paragraph 1: Field Objectives (F = Clinical AI / Healthcare Foundation Models)

Artificial intelligence applied to healthcare aims to transform clinical decision-making through automated pattern recognition in complex medical data. The overarching objective of clinical AI is to improve patient outcomes by enabling earlier disease detection, more accurate prognosis, and personalized treatment recommendations. Healthcare systems globally face mounting challenges: aging populations, physician shortages, and information overload from increasingly detailed electronic documentation. Foundation models—large neural networks pretrained on extensive unlabeled data—offer a promising paradigm for addressing these challenges by learning generalizable representations that transfer across diverse clinical tasks. The stakeholders who stand to benefit include clinicians seeking decision support, hospital administrators optimizing resource allocation, and patients receiving more timely and accurate care. However, realizing this potential requires principled methods for encoding the heterogeneous, multimodal nature of clinical data into representations suitable for modern neural architectures.

### Paragraph 2: Evaluation Framework (E = Clinical Prediction Performance)

Clinical prediction tasks provide the primary evaluation framework for assessing EHR foundation models. These tasks encompass risk stratification for adverse outcomes (mortality, readmission), early warning for acute deterioration (sepsis, respiratory failure), and resource utilization forecasting (length of stay, ICU admission). The appropriateness of prediction tasks as an evaluation framework stems from their direct clinical relevance: accurate predictions enable proactive interventions, efficient bed management, and informed shared decision-making. Evaluation practice has evolved from discrimination-only reporting (AUROC) to multi-axis assessment incorporating calibration (e.g., Brier score, ECE) and subgroup reliability via performance gap analyses across demographic strata [10,20]. Standardized benchmarks including EHRSHOT [11] and FoMoH [10] have emerged to enable systematic comparison across models. We adopt this multi-dimensional evaluation approach, recognizing that discrimination alone is insufficient for clinical deployment where probabilistic predictions inform high-stakes decisions.

### Paragraph 3: Methodological Landscape (D = EHR Foundation Model Architectures)

The application of foundation models to longitudinal EHR data has progressed through several methodological paradigms. Early approaches adapted BERT-style masked language modeling to diagnosis code sequences (BEHRT [2], Med-BERT [3]), demonstrating that self-supervised pretraining yields transferable clinical representations. Subsequent work incorporated richer event types—laboratory results, medications, procedures—requiring decisions about how to tokenize continuous measurements alongside discrete codes. The ETHOS-ARES framework [1] represents the current state-of-the-art, employing a 14-stage preprocessing pipeline that discretizes laboratory values via decile quantization before autoregressive language modeling. Alternative architectures have explored bidirectional attention (CORE-BEHRT [4]) and time-aware embedding strategies, among other extensions. Domain-specific pretraining consistently outperforms general-purpose language models on clinical benchmarks, as demonstrated by Lang1 [16] achieving superior performance over GPT-4 and Llama-70B on hospital operations tasks despite having fewer parameters. The common thread across these approaches is the need to transform raw clinical data into discrete token sequences—a transformation whose optimal design remains understudied.

### Paragraph 4: Specific Method Analysis (C = Discretization Methods)

Among the methodological choices in EHR foundation models, the discretization of continuous laboratory values into categorical tokens presents fundamental tradeoffs. Population-based quantile binning (deciles, percentiles) ensures uniform token frequency but ignores clinical semantics: a glucose of 69 mg/dL (hypoglycemic) and 71 mg/dL (normal) may occupy adjacent bins with no representation of their distinct physiological significance. Clinically-anchored binning using reference ranges ([L, U]) partitions values into below-normal, normal, and above-normal regions, preserving diagnostic relevance but potentially creating uneven bin occupancy. Finer granularity (centiles vs. deciles) increases vocabulary size and sequence length while potentially capturing subtle physiological variations. Coarser granularity reduces computational cost but may conflate clinically distinct values. The performance of these discretization variants (C', C'', C''') has been evaluated primarily in traditional machine learning contexts [6], with limited systematic comparison within the foundation model paradigm where representation learning during pretraining may differentially leverage various binning schemes.

### Paragraph 5: Enhancement Principle (B = Continuous Value Encoding)

Techniques for preserving continuous numeric information within transformer architectures offer a potential enhancement over hard discretization. The theoretical motivation is information-theoretic: discretization inherently discards precision, mapping a continuous range to a finite set of bins. xVal [7] demonstrates that continuous number encoding—representing numeric values through learned magnitude embeddings rather than categorical bins—enables strong out-of-distribution generalization on scientific datasets. The mechanism is straightforward: rather than mapping a value to one of K discrete tokens, the value directly modulates an embedding vector, preserving arbitrary precision. Convex combinations of embeddings (ConSE [14]) provide an intermediate approach: values falling between bin boundaries receive interpolated embeddings that smoothly transition across the categorical representation. These continuous encoding techniques address a fundamental limitation of discrete tokenization while maintaining compatibility with standard transformer architectures. Their effectiveness in the clinical domain—where laboratory values carry diagnostic significance and distributional properties differ from scientific datasets—requires empirical validation.

**Implementation-critical note (fairness / correctness)**: xVal-style encoders require values to be kept within a bounded dynamic range (xVal discusses preprocessing to keep values in \([-5,5]\)). In EHRs, raw magnitudes vary drastically across codes (e.g., creatinine vs. heart rate vs. infusion rates), so our pipeline uses **per-code robust scaling** computed from raw training values. For each numeric code \(c\), we compute \(\mu_c := \mathrm{median}(v_c)\) and \(\sigma_c := \mathrm{IQR}(v_c)/1.35\) on the training split and clip \(z_c = (v-\mu_c)/\sigma_c\) to \(\pm 5\). These statistics are exported as `numeric_stats.json` during tokenization and are consumed by the continuous encoder during Exp2/Exp3 training.

Crucially, we do **not** derive \((\mu_c,\sigma_c)\) from quantization breakpoints stored in `vocab.aux`, because those breakpoints can be **clinically anchored** (e.g., 5-10-5) and would otherwise couple continuous normalization to the discretization regime under test.

### Paragraph 6: Research Focus Definition (A = Input Representation for EHR Foundation Models)

We define our research focus as the systematic evaluation of input representation methods at the interface between raw clinical data and transformer-based foundation models. This focus encompasses three interrelated decisions: (1) how to discretize continuous laboratory values (granularity, clinical anchoring), (2) whether to employ discrete tokens or continuous encodings (representation mechanics), and (3) whether to use institution-specific codes or standardized vocabularies (semantic mapping). These decisions are made upstream of model architecture and training procedure, yet fundamentally shape what information is available for representation learning. The coherence of this research area stems from the observation that all EHR foundation models must address these input representation choices, typically through ad-hoc decisions inherited from prior work rather than principled optimization. Our positioning relative to adjacent research—model architecture, training objectives, downstream task formulation—is explicitly agnostic: we fix these factors to isolate representation effects.

### Paragraph 7: Prior Work in A (Input Representation Methods)

Prior work on input representation for EHR foundation models has explored various approaches without systematic comparison. ETHOS-ARES [1] employs decile quantization with separate tokens for laboratory codes and quantized values, a pragmatic design aligned with the discrete token interface of standard language models. BEHRT [2] and Med-BERT [3] focus primarily on discrete clinical codes, limiting direct analysis of continuous value representations. EHRSHOT [11] provides standardized downstream tasks for few-shot evaluation, but does not by itself resolve upstream representation choices. MedRep [8] addresses vocabulary standardization through concept mapping (e.g., OMOP alignment) but does not systematically vary discretization granularity or continuous encoding mechanics. Burkhart et al. [9] analyze representation dynamics and transferability of EHR foundation models across institutions, highlighting that institution-specific vocabularies can limit generalization and motivating controlled vocabulary-semantic comparisons. These works define a broad design space; our contribution is to systematically map key axes of this space through controlled experiments that hold model architecture and training objectives fixed.

### Paragraph 8: Critical Assessment

The existing approaches share common strengths: they successfully adapt transformer architectures to longitudinal EHR data, achieve competitive performance on standard benchmarks, and demonstrate transfer learning benefits. However, several limitations persist. First, discretization choices are typically inherited rather than optimized, with decile binning adopted by convention. Second, clinical semantics—the diagnostic significance of laboratory values relative to reference ranges—are ignored in population-based quantization. Third, the information loss from discretization is unquantified, leaving unclear whether finer granularity or continuous encoding would improve downstream performance. Fourth, vocabulary choices vary across studies, complicating cross-model comparison and cross-site deployment. Fifth, the interaction between representation choices and other factors (model scale, task type, data volume) remains unexplored.

### Paragraph 9: Perspective-Based Critique (X = Clinical Semantic Preservation)

Through the lens of clinical semantic preservation, existing input representation methods exhibit specific deficiencies. Laboratory reference ranges—the cornerstone of clinical interpretation—are entirely ignored by population-based quantization. A creatinine value of 1.1 mg/dL may be normal for a muscular adult male but concerning for an elderly female; yet both are mapped to the same quantile bin. The boundary between bins 5 and 6 in decile quantization has no clinical significance, potentially splitting clinically similar values while grouping clinically distinct ones. This semantic mismatch is particularly problematic for foundation models, where pretrained representations are expected to capture meaningful structure: if the tokenization obscures clinical meaning, the model cannot learn it. Furthermore, the choice of discrete tokens forces an artificial categorical structure onto inherently continuous measurements, with the distance between adjacent bins undefined in the embedding space. These limitations suggest that clinically-informed discretization and continuous encoding methods could yield representations more aligned with medical semantics.

### Paragraph 10: Proposed Contribution (Z = Systematic Evaluation Framework)

We propose a systematic evaluation framework for input representation methods in EHR foundation models. Our contribution is threefold: (1) **Experiment 1** evaluates quantization granularity (decile through centile) with clinically-anchored variants using laboratory reference ranges, testing whether higher resolution and semantic awareness improve prediction; (2) **Experiment 2** compares discrete tokenization against soft discretization (convex combinations) and learned continuous encoders, testing whether preserving numeric precision benefits clinical tasks; (3) **Experiment 3** contrasts institution-specific vocabularies against standardized CLIF concepts, testing whether semantic aggregation improves or harms representation quality. Across 100 training runs with a scaled-down LLaMA 3.2 architecture on MIMIC-IV v3.1, we provide the first systematic mapping of the input representation design space for clinical foundation models. [PLACEHOLDER: Preview of key findings]. The remainder of this paper details related work (§2), our methodology (§3), experimental setup (§4), results (§5), and implications (§6).

---

## 2. Related Work

### 2.1. EHR Foundation Models (D)

The development of EHR foundation models represents a convergence of advances in transformer architectures and electronic health record digitization. BEHRT [2] pioneered the application of BERT-style pretraining to diagnosis code sequences, demonstrating that masked language modeling objectives transfer effectively to clinical prediction tasks. Med-BERT [3] extended this approach to structured EHR data encompassing diagnoses, procedures, and medications, establishing that domain-specific pretraining outperforms general-purpose language models. CORE-BEHRT [4] addressed reproducibility concerns through rigorous optimization and evaluation protocols.

The ETHOS-ARES framework [1] represents the current methodological frontier, employing autoregressive language modeling over Patient Health Timelines that include laboratory results, diagnoses, procedures, medications, and temporal markers. The system achieves state-of-the-art performance on clinical risk prediction tasks including hospital admission, ICU admission, and prolonged length of stay. Critically, ETHOS tokenizes continuous laboratory values via decile quantization—a design choice we systematically evaluate. We leverage ETHOS solely for MEDS data extraction, with all tokenization, training, and evaluation performed in our independent `fms-ehrs` framework.

Recent work has highlighted the importance of domain-specific pretraining for operational clinical tasks. Lang1 [16], pretrained on 80 billion tokens of EHR and web text, outperforms larger generalist models (GPT-4, Llama-70B) on hospital operations tasks including readmission and insurance denial prediction, demonstrating that in-domain pretraining efficiency exceeds raw scale. MOTOR [17] extends foundation model approaches to time-to-event prediction, parameterizing piecewise exponential hazard functions within a transformer backbone.

### 2.2. Discretization Methods (C)

The discretization of continuous variables has a long history in machine learning, with methods categorized as supervised (entropy-based, class-attribute interdependence) or unsupervised (equal-width, equal-frequency, clustering-based) [6]. In clinical contexts, equal-frequency quantile binning has emerged as the default approach, ensuring balanced class distributions across bins.

However, clinical laboratory medicine emphasizes reference intervals—the range of values observed in healthy populations—as the primary interpretive framework [5]. Values below or above reference bounds carry distinct diagnostic significance that population-based quantization ignores. Our clinically-anchored approach partitions the value space into below-normal, within-normal, and above-normal regions before applying quantile binning within each region, preserving this diagnostic structure.

The granularity-performance tradeoff has received limited attention in the foundation model context. Finer bins (percentiles) increase vocabulary size and sequence length but may capture subtle physiological variations; coarser bins (deciles) reduce computational cost but conflate clinically distinct values. We systematically evaluate this tradeoff across four granularity levels.

### 2.3. Continuous Value Encoding (A)

Several approaches have been proposed to preserve continuous numeric information within transformer architectures. xVal [7] introduces a number encoding scheme where numeric values are represented as single tokens with learned magnitude embeddings, preserving full precision through separate encoding of mantissa and exponent. Evaluation on scientific datasets demonstrates strong out-of-distribution generalization compared to digit-based tokenization.

Norouzi et al. [14] introduced ConSE (Convex Combination of Semantic Embeddings) for zero-shot image classification, representing inputs as probability-weighted combinations of category embeddings. We adapt this principle to laboratory value representation: values falling between bin boundaries receive interpolated embeddings based on their relative position, smoothly transitioning across the discrete representation while maintaining the computational structure of categorical embeddings.

The Continuous Autoregressive Language Model (CALM) [18] replaces discrete next-token prediction with continuous next-vector prediction using variational autoencoders, demonstrating that autoregressive modeling extends naturally to continuous outputs. While CALM targets text generation, its principles inform continuous value encoding for EHR data where preserving numeric precision may benefit downstream prediction.

### 2.4. Temporal Encoding in Transformers

Temporal structure is fundamental to longitudinal EHR data, where the timing of clinical events carries diagnostic significance. Standard transformer positional encodings (sinusoidal, learned) encode sequence position rather than absolute or relative time. ETHOS-ARES [1] addresses this through discrete time spacing tokens (e.g., `T_5m-15m`, `T_1h-2h`) injected between events to encode elapsed time.

Time2Vec [15] provides a learned continuous representation of time through periodic and linear components: $t2v(\tau) = [w_0 \tau + \phi_0, \sin(w_1 \tau + \phi_1), \ldots]$. This encoding captures periodic patterns (circadian rhythms, weekly cycles) alongside linear trends. JETS [19] applies similar principles to irregular multivariate time series from wearables, using Mamba-based encoders to handle variable sampling intervals.

Rotary Position Embeddings (RoPE), native to LLaMA architectures [12], encode relative positions through rotation matrices applied to query and key vectors. RoPE and continuous temporal encoding operate on orthogonal embedding dimensions—RoPE handles sequence position while Time2Vec handles absolute timestamps—enabling their composition.

### 2.5. Positioning Statement

Our work differs from prior literature in several key respects. Unlike ETHOS-ARES [1], we systematically vary discretization granularity and evaluate clinically-anchored alternatives. Unlike xVal [7] and CALM [18], we focus on clinical rather than scientific data, where reference ranges provide domain-specific structure. Unlike MedRep [8], we evaluate vocabulary choices alongside discretization and encoding methods. Our contribution is the first systematic benchmark mapping the input representation design space for EHR foundation models, providing empirical guidelines that complement architectural and training innovations.

---

## 3. Methods

### 3.1. Problem Formulation

Let $\mathcal{P} = \{p_1, \ldots, p_N\}$ denote a set of patients, where each patient $p_i$ is associated with a longitudinal health timeline $\mathcal{T}_i = \{(e_1, t_1), \ldots, (e_{L_i}, t_{L_i})\}$ consisting of clinical events $e_j$ occurring at timestamps $t_j$. Events include laboratory measurements $(c, v)$ with code $c$ and numeric value $v$, as well as other time-stamped clinical events (e.g., admissions/discharges, medication administrations, transfers, ICU vitals/inputs/outputs/procedures) represented as categorical codes and, when applicable, aligned scalar values in `numeric_value` under the MEDS schema.

The input representation problem is to define a mapping $\phi: \mathcal{T}_i \rightarrow \mathcal{S}_i$ from raw timelines to token sequences $\mathcal{S}_i = (s_1, \ldots, s_M)$ suitable for transformer processing. This mapping encompasses:

1. **Discretization**: For continuous values $v$, define bin boundaries $\{b_0, b_1, \ldots, b_K\}$ and map $v \mapsto Q_k$ where $b_{k-1} \leq v < b_k$
2. **Encoding**: Produce embeddings $\mathbf{e}_j \in \mathbb{R}^d$ for each token, either through discrete embedding lookup or continuous projection
3. **Temporal encoding**: Incorporate timestamp information through discrete tokens or continuous representations

The downstream task is clinical prediction: given a timeline prefix $\mathcal{T}_i^{<t}$, predict binary outcomes $y_i \in \{0, 1\}$ for mortality, length of stay, ICU admission, and mechanical ventilation.

### 3.2. Preliminaries

**MEDS Format**: We represent longitudinal EHR data in the Medical Event Data Standard (MEDS) schema, where each event is a row with minimally required columns `subject_id`, `time`, and `code`, and optional value columns including `numeric_value` (for scalar measurements) and metadata such as `ref_range_lower` / `ref_range_upper` (for laboratory reference intervals). This schema supports mixed discrete and continuous event streams in a single chronologically ordered timeline.

**Data Pipeline**: We use the MEDS extraction pipeline adapted from **ETHOS-ARES** [1] for MIMIC-IV → MEDS conversion. The original pipeline is available at https://github.com/ipolharvard/ethos-ares under the MIT License (Copyright © 2024 Paweł Renc). We have adapted this pipeline to our repository with:
- Modified split fractions (70/10/20 vs. original 90/10)
- Custom event configuration using `storetime` semantics (when available) to better approximate information availability and reduce temporal look-ahead
- Extended MEDS fields (`ref_range_lower`, `ref_range_upper`) for clinical anchoring

All downstream tokenization, training, and evaluation rely entirely on our independent `fms-ehrs` codebase. This separation ensures clean modularity: MEDS extraction (adapted from ETHOS-ARES) produces standardized parquet files, while `fms-ehrs` handles all representation experiments without upstream dependencies.

**Tokenization (fms-ehrs)**: We tokenize MEDS timelines using our `fms-ehrs` framework. Tokenization is YAML-configurable and supports: (i) quantization strategies (deciles/ventiles/trentiles/centiles), (ii) clinically anchored binning using reference intervals [5], (iii) optional fused code–value tokens (to reduce sequence length), (iv) time spacing tokens (as in ETHOS-ARES [1]) and Time2Vec temporal features [15], and (v) aligned auxiliary arrays for continuous-value representations (see §4.4). In a concrete validation run on MEDS-formatted MIMIC-IV v3.1 (deciles, non-fused, time-spacing tokens, `max_padded_len=1024`, with 24h-cut artifacts), tokenization produced a vocabulary of **19,363** tokens and **297,817** hospitalization timelines in the train split after the ≥24h stay filter; full commands, timestamps, and readouts are recorded in `methods/data-columns.md`.

**LLaMA Architecture**: Our base model is a decoder-only transformer with rotary position embeddings, using a scaled-down configuration (config overrides: `hidden_size=1024`, `intermediate_size=2048`, `num_hidden_layers=8`, `num_attention_heads=8`; **≈87M parameters**, depending on vocab size) for computational tractability across 100 training runs.

**Data Inclusion Criteria and Label Leakage Prevention**: A central requirement for predictive evaluation is that model inputs reflect information that would be available at the prediction time. Several administrative/billing code tables in MIMIC-IV are known to be assigned or finalized at or after discharge (e.g., ICD diagnosis/procedure codes, DRG codes), making them high-risk sources of temporal leakage for within-admission prediction. Accordingly, we exclude four MIMIC-IV tables whose primary purpose is post hoc billing/administration and may encode outcome-related information not available during the admission [13]. This design choice improves **internal validity** (reduces leakage-driven overestimation of performance) and improves **ecological validity** for deployment-time prediction [10,20].

| Excluded Table | Code Type | Reason |
|----------------|-----------|--------|
| `hosp/diagnoses_icd` | ICD-9/ICD-10 | Assigned by coders at discharge |
| `hosp/procedures_icd` | ICD-9/ICD-10 | Billed retrospectively |
| `hosp/hcpcsevents` | CPT/HCPCS | Billing codes for services |
| `hosp/drgcodes` | DRG | Reimbursement codes |

We include real-time clinical data from `hosp/admissions`, `hosp/labevents`, `hosp/emar`, `hosp/omr`, `hosp/patients`, `hosp/transfers`, and the ICU module tables (`icustays`, `chartevents`, `inputevents`, `outputevents`, `procedureevents`). Notably, `icu/procedureevents` uses internal MIMIC `itemid` identifiers—not billing codes—representing real-time clinical documentation. For timestamp semantics, we order all events by **storage time** (`storetime`) instead of event occurrence time (`charttime`, `starttime`, `endtime`). This ensures event ordering reflects information availability in the EHR system, preventing look-ahead bias where a measurement physically occurred before it was actionable to clinicians.

**Included event types and columns**: The primary focus of Experiments 1–2 is on laboratory values (`hosp/labevents.valuenum`) and their reference intervals (`ref_range_lower`, `ref_range_upper`) [5]. Additional event streams (medications, transfers, ICU vitals/inputs/outputs/procedures) provide clinical context and enable downstream outcome labeling under consistent timestamp semantics. A complete table→column mapping used in this benchmark is provided in `methods/data-columns.md`.

### 3.3. Representation Models, Hypotheses, and Falsifiable Consequences

We frame input representation as a *model class* choice that induces measurable differences in (i) information preserved from raw clinical measurements, (ii) inductive biases presented to the transformer during pretraining, and (iii) computational and statistical properties of downstream evaluation. We pre-specify competing representation models and falsifiable hypotheses before observing benchmark outcomes.

**Model classes (competing representations)**:

- **M1: Population-quantile discretization (baseline)**: Continuous values are mapped to discrete quantile bins (e.g., deciles) and represented as categorical tokens, following prior pipelines such as ETHOS-ARES [1].
- **M2: Clinically anchored discretization**: Bin boundaries incorporate laboratory reference intervals [5] by partitioning values into below-/within-/above-normal regions prior to quantile binning (Experiment 1).
- **M3: Soft discretization (convex combinations)**: Values retain within-bin position information by interpolating between adjacent bin embeddings, adapting ConSE-style convex combination ideas [14] to ordered numeric bins (Experiment 2).
- **M4: Learned continuous encoders**: Values are encoded via a learned projection of standardized measurements, inspired by continuous numeric embeddings (xVal [7]) while retaining the discrete code channel (Experiment 2).

**Global hypothesis (H\*)**: Input representation choices upstream of model architecture yield statistically detectable differences in downstream discrimination and calibration, even when architecture and training objectives are held fixed.

**Experiment-specific hypotheses and consequences**:

- **H1 (clinical anchoring)**: Clinically anchored binning (M2) improves clinically relevant discrimination and/or calibration relative to population-quantile binning (M1), particularly for outcomes sensitive to abnormal physiology.  
  - *Consequence*: Mean AUROC/AUPRC and/or Brier/ECE improves for M2 vs. M1 across tasks; null model is no difference beyond sampling variability (paired comparisons across seeds/configs).
- **H2 (granularity tradeoff)**: Increasing bin granularity improves representation fidelity up to a point, after which performance saturates or degrades due to sparsity and longer sequences.  
  - *Consequence*: Performance exhibits a non-monotone or saturating trend across deciles→ventiles→trentiles→centiles with measurable efficiency tradeoffs (token counts/FLOPs).
- **H3 (soft/continuous value encoding)**: Preserving numeric precision via M3 or M4 improves calibration metrics and may improve discrimination relative to purely discrete value tokens (M1/M2), conditional on controlling for temporal encoding and tokenization.  
  - *Consequence*: Lower Brier/ECE (primary) and potentially higher AUROC/AUPRC (secondary) for M3/M4 vs. discrete baselines.
- **H4 (temporal encoding)**: Time2Vec [15] provides measurable benefit over discrete time spacing tokens [1] for outcomes requiring longer-range temporal patterning. For a clean either/or comparison, our Time2Vec condition **removes time-spacing tokens** at tokenization time and provides temporal information exclusively via Time2Vec embeddings added to the token representations.  
  - *Consequence*: Improved metrics for Time2Vec vs. time spacing tokens within the same value representation model.

### 3.4. Proposed Evaluation Framework (Z)

Our systematic evaluation framework comprises three experiments examining orthogonal representation dimensions:

**Experiment 1: Granularity and Clinical Anchoring**

We evaluate six granularity configurations:
- Deciles (10 bins): Population-based, ETHOS baseline
- Ventiles (20 bins): Population-based
- Clinically-anchored ventiles (5-10-5): Reference range partitioning
- Trentiles (30 bins): Population-based
- Clinically-anchored trentiles (10-10-10): Reference range partitioning
- Centiles (100 bins): Population-based

The *granularity* parameter is applied to MEDS event types that carry `numeric_value` and are included in our tokenizer configuration (notably labs, ICU vitals, fluid outputs, and infusion start rates). *Clinical anchoring* is applied only to laboratory events with reference intervals (`ref_range_lower`, `ref_range_upper`) [5]; for non-laboratory numeric streams that lack reference ranges, bins are computed via population quantiles. (See `methods/data-columns.md` for exact inclusion/exclusion notes, e.g., OMR omission and categorical treatment of `INFUSION_END`.)

Each configuration is evaluated with and without fused code–value tokens (combining a numeric event’s code token and its discretized bin token), yielding 12 configurations and enabling explicit measurement of the accuracy–efficiency tradeoff.

**Experiment 2: Representation Mechanics**

Using the optimal granularity from Experiment 1, we evaluate three encoding methods:
- Discrete tokens: Standard embedding lookup
- Soft discretization: Convex combination of adjacent bin embeddings
- Continuous encoder: xVal-adapted scaled embedding of z-score normalized values

Each encoding is evaluated with two temporal strategies:
- Time spacing tokens: Discrete intervals following ETHOS protocol
- Time2Vec + RoPE: Learned continuous temporal encoding

This yields 6 configurations.

**Experiment 3: Vocabulary Semantics as a Controlled Intervention (paired design)**

Using the winning discretization regime from Experiment 1 and winning (value, temporal) mechanics from Experiment 2, Experiment 3 isolates **vocabulary semantics** by treating semantic mapping as an *intervention applied to the same underlying event rows*.

**Core construction (paired events, single cohort)**:
- Fix a cohort \(H_{\mathrm{ICU}}\) (ICU-eligible hospitalizations with hospital LOS ≥24h) and a fixed set of event rows \(E\) (matched-signal subset for Exp3; by default **labs + vitals**, see `methods/data-columns.md`).
- For every event row \(e \in E\), retain the same timestamps and numeric values, but define two parallel code namespaces:
  - **Native codes**: institution-specific identifiers (e.g., `LAB//50931//mg/dl`, `VTL//220045//bpm`)
  - **Standardized codes (CLIF semantics)**: mapped concepts (e.g., `LAB//glucose_serum`, `VITAL//heart_rate`)
- The only experimental variable is which namespace is emitted as the code token stream; the value channel (`numeric_values`) and time inputs are held fixed.

**Primary comparison arms**:
- **MEDS (native)**: tokenize \(E\) using native codes.
- **CLIF (standardized / CLIF)**: tokenize \(E\) using standardized codes produced by CLIF-MIMIC mapping rules.

**Negative controls (null models; required for internal validity)**:
- **Randomized mapping control**: permute native→standardized mappings *within domain* (e.g., labs within labs, vitals within vitals) while preserving the same number of categories. If **CLIF ≈ randomized mapping**, observed gains are not attributable to semantics.
- **Frequency-matched collapse control**: collapse codes into \(K\) pseudo-categories with matched frequency profile to the standardized vocabulary (tests “controlled-vocab regularization” vs “semantic mapping”).

**Localizing where semantics matter (ablation)**:
- **Partial mapping**: map only labs or only vitals (and any additional domains included by the Exp3 tokenizer config), leaving the rest native. This localizes where semantic abstraction helps/hurts.

**Reference ranges for clinically anchored winners**:
If Experiment 1 selects a clinically anchored winner and Exp3 includes lab anchoring, we require reference-interval bounds for the standardized arm. The base CLIF 2.1 labs schema includes `reference_unit` but not lower/upper bounds. We therefore augment CLIF-derived lab rows with `ref_range_lower` and `ref_range_upper` estimated from MIMIC-IV `labevents` grouped by `(lab_loinc_code, reference_unit)` so anchoring can be applied symmetrically across MEDS/CLIF. See `scripts/augment_clif_labs_with_ref_ranges.py`.

This yields at minimum 2 arms (MEDS/CLIF), with 2 additional null controls (randomized mapping; frequency-matched collapse) for rigor.

---

## Scheduling / dependency logic (implementation note)

The experiments form a conceptual dependency chain for *winner selection*:

- **Exp1 → Exp2**: Exp1 selects the best *tokenization regime* (quantizer + clinical anchoring + fused/unfused). Exp2 should use that regime for the final discrete baselines.
- **Exp2 → Exp3**: Exp2 selects the best *(representation, temporal)* mechanics; Exp3 should apply that winner when comparing MEDS vs CLIF semantics.

All Exp2 work must wait for Exp1 winner selection in this benchmark.

**Rationale (method faithfulness + internal validity for RQ2)**:
- Our **soft discretization** implementation (ConSE-inspired) explicitly requires the Exp1 bin boundaries (stored in `vocab.aux`) to interpolate between adjacent bins; therefore Exp2-soft is not defined without choosing the Exp1 quantizer/anchoring regime.
- Our **continuous encoder** is inspired by xVal [7], but in our *MEDS-compatible adaptation* we do **not** replace numbers with a dedicated `[NUM]` token. Instead, we apply the continuous embedding at **quantile-token positions** in the tokenized sequence (see §3.5). This means Exp2-continuous still depends on the Exp1 tokenization regime to define the quantile-token stream (number of bins and which tokens represent numeric values).
- Consequently, running Exp2 under an arbitrary pre-registered tokenization while Exp1 is still underway would contaminate the interpretation of RQ2 (“representation mechanics”), because differences could be driven by a mismatched discretization regime rather than by soft/continuous mechanics.

**Operational rule**:
- Exp2 tokenization and training must be parameterized by the Exp1 winner regime.
- Soft/continuous runs must use the Exp1 winner *within the unfused group* (since soft/continuous require unfused tokenization).

To prevent accidental “placeholder winners”, the job generator enforces:

- **Exp3 jobfiles are not generated unless explicit winner selections are provided** (either as explicit flags or as winner `config_id` strings).

**Why Exp3 must wait until Exp1+Exp2 winners are finalized**:

Although Exp3 includes a “MEDS arm”, it is **not** redundant with “the best Exp2 MEDS model” because Exp3 is evaluated on a *different cohort* for parity with CLIF:

- Exp1/Exp2 train on **all hospitalizations** with stay ≥24h.
- Exp3 (both MEDS and CLIF arms) train on the **MIMIC-ICU cohort** with hospital stays ≥24h, aligned by patient IDs across formats.

Therefore, Exp3 requires *re-training the MEDS-format model* under the Exp1+Exp2 winner configuration **on the ICU-aligned cohort**, so that the MEDS vs CLIF comparison isolates vocabulary semantics rather than cohort shift. Running Exp3 MEDS before Exp2 completes (or reusing an Exp2 MEDS checkpoint trained on the all-hospitalizations cohort) would be methodologically incorrect.

### 3.5. Representation Methods

This section specifies the algorithms used in Experiment 2, with
links to the original literature and the adaptations required by our
MEDS tokenization pipeline. We separate conceptual model definitions (what the
paper claims) from implementation choices (what is executed).

**Clinically-Anchored Quantization**

For laboratory code $c$ with reference range $[L_c, U_c]$, we partition values into three clinical regions:
- Below normal: $\{v : v < L_c\}$
- Within normal: $\{v : L_c \leq v \leq U_c\}$
- Above normal: $\{v : v > U_c\}$

Within each region, we apply equal-frequency quantile binning. For 5-10-5 ventile allocation:
$$Q = \{q_{0.2}, q_{0.4}, q_{0.6}, q_{0.8}\}_{below} \cup \{L_c\} \cup \{q_{0.1}, \ldots, q_{0.9}\}_{within} \cup \{U_c\} \cup \{q_{0.2}, q_{0.4}, q_{0.6}, q_{0.8}\}_{above}$$

**Soft Discretization (ConSE-inspired) [14]**

ConSE defines a convex combination of semantic embeddings using classifier
probabilities. Our adaptation replaces classifier probabilities with *local
interpolation weights* derived from bin boundaries, yielding a continuous
embedding along an ordered quantile axis.

For value $v$ falling between bin boundaries $b_i$ and $b_{i+1}$:
$$\alpha = \frac{v - b_i}{b_{i+1} - b_i}$$
$$\mathbf{e}(v) = (1 - \alpha) \cdot \mathbf{E}[b_i] + \alpha \cdot \mathbf{E}[b_{i+1}]$$

**Edge handling**: if $v < b_0$, we return $\mathbf{E}[b_0]$; if
$v \ge b_{K-1}$, we return $\mathbf{E}[b_K]$. This preserves monotonicity while
using only two embeddings per value, matching ConSE’s convexity property but
specialized to ordered numeric bins.

**Implementation alignment**:
1. Bin boundaries are derived from the tokenizer’s quantile auxiliary data.
2. A vectorized path uses `code_ids` to fetch per-code boundary tensors.
3. The embedding dimensionality is identical to the base LM token embedding.

**Continuous Encoder (xVal-adapted) [7]**

xVal embeds a numeric value by multiplying a learnable embedding direction by
the scalar value, using a dedicated `[NUM]` token and (optionally) multi-scale
variants with $\tanh(x \cdot 10^i)$ factors. We preserve the *core continuous
inductive bias* while adapting to our MEDS pipeline:

1. **Tokenization constraint**: numeric values are aligned to *quantile tokens*
   (not replaced by a `[NUM]` token). We therefore compute a continuous numeric
   embedding *at quantile-token positions* and replace those embeddings.
2. **Normalization**: values are standardized per code using training statistics
   $(\mu_c, \sigma_c)$ and clipped to $\pm 5$ standard deviations to prevent
   outliers from dominating training dynamics.
3. **Number head**: xVal’s number prediction head is omitted because we do not
   generate numeric tokens; the benchmark evaluates representation quality via
   downstream prediction.

Mathematically, for a value $v$ with code $c$:
$$z = \text{clip}\left(\frac{v - \mu_c}{\sigma_c}, -5, 5\right)$$
$$\mathbf{e}(v) =
\begin{cases}
z \cdot \mathbf{u} & \text{(default, }k=0\text{)} \\
\sum_{i=-k}^{k} \tanh(z \cdot 10^i)\cdot \mathbf{u}_i & \text{(multiscale)}
\end{cases}$$
where $\mathbf{u}$ (and $\mathbf{u}_i$) are learned embedding directions.
In this study we use the default $k=0$ setting to avoid introducing
representation-specific hyperparameter tuning.

**Time2Vec Temporal Encoding [15]**

For relative time $\tau$ (hours since admission), Time2Vec defines a vector
embedding:
$$t2v(\tau) = \left[w_0 \tau + \phi_0, \sin(w_1 \tau + \phi_1), \ldots, \sin(w_k \tau + \phi_k)\right]$$

**Implementation alignment**:
1. We use $\sin$ as the periodic activation, as in the canonical Time2Vec
   configuration.
2. We compute $\tau$ as relative hours since admission, consistent with MIMIC-IV
   date shifting [13].
3. We add Time2Vec embeddings to token embeddings and apply layer normalization
   (additive composition), which is a common transformer-compatible integration
   strategy. For a clean comparison, the Time2Vec condition **removes time
   spacing tokens** at tokenization, so temporal information is carried solely
   by the continuous encoding.

**Figure placeholder**: Schematic of Exp2 representation flow (token embedding →
value embedding replacement → Time2Vec addition → transformer).


### 3.6. Implementation Details

**Model Configuration**:
- Architecture: decoder-only transformer initialized from the LLaMA family configuration (RoPE positional encoding) [12], then scaled down for tractable experimentation
- Parameters: **≈87M** in our pipeline (exact count depends on tokenizer vocabulary size)
- Hidden dimension: 1024
- Intermediate size: 2048
- Layers: 8
- Attention heads: 8
- Sequence length used in experiments: 1,024 tokens (padded/truncated per tokenizer configuration); longer-context capacity of the base architecture is not exercised in the present benchmark

**Training Configuration**:
- Pretraining objective: causal language modeling with cross-entropy loss (as in ETHOS-style autoregressive timeline modeling [1])
- Pretraining implementation (Exp 1): `fms_ehrs/scripts/tune_model.py` (TRL SFT training with packed collation), following the `Quantifying-Surprise-EHRs` reference pattern:
  - DDP via `torchrun` (4 or 8 GPUs/job, depending on queue packing)
  - Optuna HPO with small `n_trials` (3 as in the reference SLURM script)
  - **Single-epoch default**: `n_epochs=1` per trial (see “epoch budget” note below)
  - HPO search space (reference): learning rate in \([5\times10^{-5}, 5\times10^{-4}]\) (log-uniform) and gradient accumulation steps in \(\{1,2,3\}\)
  - Model architecture overrides (reference): `hidden_size=1024`, `intermediate_size=2048`, `num_hidden_layers=8`, `num_attention_heads=8` (≈87M parameters in our runs; exact depends on vocab size)
  - Packed datasets trained in **iterable** mode (no materialization), consistent with the reference `Datasets.get_train_dataset(..., iterable=True)`
- Representation training (Exp 2 / Exp 3): `fms_ehrs/scripts/train_representation.py` (default in this benchmark: `n_epochs=1`, `per_device_train_batch_size=1`, `gr_acc_min=4`, `gr_acc_max=12`). Optuna can additionally tune essential representation knobs (categorical choices) under the same trial budget.
- Downstream evaluation: representation-based prediction using logistic regression on extracted hidden states, mirroring the Quantifying-Surprise-EHRs workflow in technical detail:
  - **Extract reps (gpuq; 2 GPUs)**: vendored `slurm/ref_qse/09_extract_reps.sh` runs `torchrun --nproc_per_node=2 ... extract_hidden_states.py`, saving `features-<model>.npy` into each split folder
  - **Predict (tier2q; CPU-only)**: vendored `slurm/ref_qse/10_xfer_rep_based_preds.sh` runs `transfer_rep_based_preds.py --classifier logistic_regression`; we scale tier2q resources (CPUs/RAM) for our larger cohort via `slurm/11_run_stage3_tier2q_lr.sh`
- Random seeds: 5 per configuration (42, 123, 456, 789, 1024)

**Epoch budget (engineering choice; preregistered)**:
Modern pretraining practice prioritizes *more diverse, high-quality data* over repeated epochs on a fixed dataset. Because this benchmark is explicitly a multi-run sweep (20–100+ training runs), we default to **single-epoch training** for Exp1–Exp3 to:
1. Reduce overfitting risk from repeated exposure under small trial budgets,
2. Keep the compute budget comparable across many conditions (internal validity of comparisons), and
3. Improve experimental throughput under a 24h/job SLURM limit.

Operationally, our training scripts implement “epochs” as dataset repetition and compute:
\[
\texttt{max\_steps} \propto \frac{n_{\text{train}} \cdot n_{\text{epochs}}}{\texttt{per\_device\_train\_batch\_size} \cdot \#\text{GPUs}}
\]
so reducing to \(n_{\text{epochs}}=1\) directly reduces the number of optimizer steps per Optuna trial.

**Fairness and HPO budget control (Exp 1–3)**:
We use **matched-budget tuning of essential hyperparameters** to ensure that
each method is evaluated under valid operating conditions without granting
extra compute advantages. Concretely:
1. **Same HPO protocol**: Optuna is used across Experiments 1–3 with the same
   trial counts and epoch budgets (unless explicitly stated otherwise).  
2. **Method-specific essential knobs (matched budget)**: representation-specific
   hyperparameters that materially affect **internal validity** of the mechanistic comparison (e.g., Time2Vec dimension,
   xVal multiscale depth, soft-discretization bin count) are tuned within *the
   same* trial budget per method.
3. **Matched compute per run**: identical GPU counts, epochs, and gradient
   accumulation ranges are used across Exp2 variants to ensure walltime and
   token exposure are comparable.
4. **Parameter count transparency**: added parameters from Time2Vec or xVal are
   reported alongside results.

We pre-register candidate values for each essential knob and restrict Optuna to
categorical choices from those lists. This preserves fairness while avoiding
method under-optimization that could bias comparisons.

**Table placeholder**: HPO search space and matched budgets for Exp2 (learning
rate/grad accumulation + essential representation knobs), including total
trials and GPU-hours per method.

**Default pre-registered grids (Exp2)**:
- Time2Vec dimension: $\{32, 64, 128\}$ (covers low/medium/high capacity while
  keeping added parameters modest relative to the base LM).
- xVal multiscale depth: $\{1, 3\}$ (default $k=0$ vs. a minimal multiscale
  variant with $k=1$).
- Soft discretization bin count: fixed to the Experiment 1 winning granularity
  (to avoid confounding the Exp1 granularity effect).

**Selection rationale**:
1. **Coverage**: include a low/medium/high range to capture diminishing returns
   without exploding the search space.
2. **Compute parity**: restrict to categorical grids that keep trial counts and
   walltime comparable across methods.
3. **Internal validity**: only tune knobs that materially affect representation
   mechanics (e.g., Time2Vec dimension, xVal scales), not downstream training
   hyperparameters beyond the shared Optuna space.

**Max sequence length policy**:
We fix `max_seq_length=1024` across all experiments for comparability and to
fit within the 24h walltime constraint. If we increase this ceiling, we must
rerun tokenization and training for **all** experiments, because Exp1 uses
packed sequences (longer context increases packing cost) while Exp2/Exp3 use
padded sequences (longer context increases per-step memory and compute). A
length change would therefore shift compute and truncation profiles differently
across experiments, violating fairness unless applied uniformly.

**Feasibility under a 24h walltime limit (benchmark data scale)**:

The reference workflow reduces walltime per Optuna trial by using 8-GPU DDP. 
Our benchmark additionally uses a larger cohort (MIMIC-HOSP + MIMIC-ICU) and a broader set of MEDS-derived
columns (see `methods/data-columns.md`), increasing token volume per admission. As a representative example,
for `deciles_none_unfused_time_tokens` (MEDS; all-hospitalizations cohort):

- Train admissions: 297,817
- Validation admissions: 41,869
- Mean `seq_len` (train): 2,041 tokens
- Max `seq_len` (train): 569,422 tokens (rare extreme outliers)

These facts motivate matching the reference infrastructure pattern (8-GPU DDP + iterable packed datasets)
to keep walltime per trial within the 24h constraint.

**Infrastructure**:
- Experiment orchestration: bash job files + SLURM arrays (fms-ehrs pattern)
- Job scheduling: SLURM arrays with capped concurrency, respecting per-job GPU requests (e.g., Exp1 Stage1 uses 8 GPUs/job under the reference pattern)
- Logging: Weights & Biases
- Hardware: Randi cluster (gpuq partition; nodes advertise `gpu:8` via Slurm GRES). [PLACEHOLDER: insert GPU model from `nvidia-smi` once recorded]

---

## 4. Experimental Setup

### 4.1. Dataset

**MIMIC-IV v3.1** [13]: A freely accessible critical care database from Beth Israel Deaconess Medical Center.

| Statistic | Value |
|-----------|-------|
| Patients | 364,627 |
| Hospital admissions | 546,028 |
| ICU stays | 94,458 |
| Laboratory events | 142,131,243 |
| Unique laboratory codes | 1,128 |
| Codes with reference ranges | 392 (34.8%) |
| Events with reference ranges | 114,135,372 (80.3%) |
| Top 200 codes coverage | 97.06% |

**Preprocessing**: We use the MEDS extraction pipeline adapted from ETHOS-ARES [1] for MIMIC-IV → MEDS conversion. All tokenization is performed by our `fms-ehrs` framework; in our Phase 1 validation run (see `methods/data-columns.md` for provenance), the resulting MEDS tokenization produced a **19,363**-token vocabulary with **297,817 / 41,869 / 85,530** train/val/test hospitalization timelines after the ≥24h stay filter. Extended MEDS extraction includes `ref_range_lower` and `ref_range_upper` columns for clinical anchoring.

**Cohorts**: We define two cohorts based on experimental requirements:
- **Experiments 1 & 2**: All hospitalizations with stay ≥24 hours
- **Experiment 3**: ICU-eligible hospitalizations with stay ≥24 hours, where **all Exp3 arms are constructed from the same event rows** and differ only by code-namespace emission (native vs standardized vs null controls).

**Splitting**: Patient-level 70/10/20 train/validation/test split with temporal ordering respected within each patient.

**Deidentification Note**: MIMIC-IV dates are shifted randomly per patient, preserving relative time differences within patients while preventing cross-patient temporal comparison [13]. Our temporal encodings use relative time (hours since admission) rather than absolute timestamps.

### 4.2. Baselines

| Method | Description | Source |
|--------|-------------|--------|
| ETHOS-Decile | 10-bin population quantization, time spacing tokens | [1] |
| ETHOS-Ventile | 20-bin population quantization | This work |
| Clinical-Ventile | 5-10-5 reference-anchored | This work |
| Soft-Discrete | Convex combinations of bin embeddings | Adapted from [14] |
| Continuous-xVal | Z-score + xVal-style scaled embedding | Inspired by [7] |
| Time2Vec | Learned temporal encoding | [15] |

### 4.3. Evaluation Metrics

We define evaluation metrics consistent with clinically meaningful model assessment for structured EHRs (FoMoH [10]) and contemporary guidance on predictive model evaluation beyond discrimination (Van Calster et al. [20]). Metrics are computed from predicted probabilities on held-out splits and summarized as mean±SD across random seeds.

### 4.3.1. Outcome Definitions and Label Extraction

We evaluate representation quality using the four prediction tasks used in `fms-ehrs` (Burkhart et al. [9]): **same-admission mortality**, **long length of stay** (\(>7\) days), **ICU admission**, and **invasive mechanical ventilation (IMV)**. We compute two auxiliary 24-hour window flags (`icu_admission_24h`, `imv_event_24h`) to define “after-24h” cohorts without label leakage, following the logic in `fms_ehrs/scripts/extract_outcomes.py`.

**CLIF (Experiment 3, standardized arm)**: Outcomes are extracted using the reference CLIF implementation (`fms_ehrs/scripts/extract_outcomes.py`), which relies on time-stamped CLIF tokens (e.g., `RESP_IMV`) and 24h-truncated tokenized timelines.

**MEDS (Experiments 1–2, and Experiment 3 MEDS arm)**: We do **not** compute IMV timing from token presence because the MEDS tokenizer configuration `fms_ehrs/config/mimic-meds-ed.yaml` appends procedures as **suffix tokens at discharge time** (`suffix: PROC`), which destroys procedure timing and makes `imv_event_24h` incorrect if derived from a 24h-truncated token sequence. Instead, we compute outcomes directly from **MEDS event timestamps** under storetime semantics (`benchmarks/mimic-meds-extraction/configs/event_configs_v3.1_full.yaml`) and join them onto tokenized timelines using `scripts/extract_outcomes_meds.py`. IMV is defined by MEDS `PROCEDURE//{itemid}` events with \(itemid \in \{224385, 225792\}\) (initial mapping; validated against CLIF in Experiment 3).

**Discrimination (current code)**:
- AUROC: Area under ROC curve, threshold-independent discrimination

**Threshold metrics (current code)**:
- Accuracy, balanced accuracy, precision, recall (computed by `fms_ehrs/framework/logger.py`)

**Calibration + additional discrimination (planned)**:
We will add AUPRC, Brier score, and ECE once implemented end-to-end in the LR evaluation stage and logged consistently for all experiments.

**Fairness**:
- Subgroup AUROC Gap: Maximum AUROC difference across sex, race, age quartiles

**Efficiency**:
- Token count: Mean sequence length per patient
- Training FLOPs: Computational cost to convergence
- Inference latency: Wall-clock time per prediction

### 4.4. Experimental Protocol

**Training**: Causal language modeling with cross-entropy loss. Early stopping on validation perplexity with patience of 5 epochs.

**Evaluation (reference-repo-aligned)**: For each trained checkpoint, we extract fixed patient-level representations using `extract_hidden_states.py` and evaluate them using `transfer_rep_based_preds.py --classifier logistic_regression`, mirroring the `Quantifying-Surprise-EHRs` workflow. We report discrimination metrics from the resulting probabilistic predictions and summarize results across five random seeds.

**Experiment 2 numeric value channel**: For soft discretization and continuous encoders, the model consumes the raw per-event scalar measurement (`numeric_value`) aligned to token positions. Tokenization emits a parallel `numeric_values` array aligned to `tokens` (and `padded_numeric_values` aligned to `padded`) such that only quantile-token positions carry the measurement and all other positions are null/masked. This preserves true measurement values while keeping the discrete code-token channel unchanged.

**Statistical Analysis**:
- Paired bootstrap hypothesis tests (n=1000) with Bonferroni correction
- Effect sizes (Cohen's d) for practical significance
- 95% confidence intervals from 5-seed ensembles

**Staged Execution**:
1. Single-seed demo (20 runs) to validate pipeline
2. Full 5-seed runs (100 total) for statistical robustness

---

## 5. Results

### 5.1. Main Results

[PLACEHOLDER: Primary comparison table with all methods across all tasks]

| Method | Mortality AUROC | LOS AUROC | ICU AUROC | IMV AUROC | Mean |
|--------|-----------------|-----------|-----------|-----------|------|
| ETHOS-Decile | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| ... | ... | ... | ... | ... | ... |

### 5.2. Experiment 1: Granularity Effects

[PLACEHOLDER: Analysis of granularity impact across tasks]

[PLACEHOLDER: Comparison of population-based vs clinically-anchored binning]

[PLACEHOLDER: Effect of fused tokens on sequence length and performance]

### 5.3. Experiment 2: Representation Mechanics

[PLACEHOLDER: Discrete vs soft vs continuous encoding comparison]

[PLACEHOLDER: Time spacing tokens vs Time2Vec temporal encoding]

[PLACEHOLDER: Interaction effects between value encoding and temporal encoding]

### 5.4. Experiment 3: Vocabulary Semantics

[PLACEHOLDER: MIMIC native vs CLIF standardized comparison]

[PLACEHOLDER: Analysis of vocabulary mapping coverage and impact]

### 5.5. Ablation Studies

[PLACEHOLDER: Component-wise ablations for soft discretization]

[PLACEHOLDER: xVal scale variants (k=0 vs multiscale) and sensitivity]

[PLACEHOLDER: Time2Vec dimensionality and periodicity analysis]

### 5.6. Efficiency Analysis

[PLACEHOLDER: Token count comparison across representations]

[PLACEHOLDER: Training FLOP analysis]

[PLACEHOLDER: Inference latency benchmarks]

### 5.7. Calibration and Fairness

[PLACEHOLDER: ECE and Brier score comparison]

[PLACEHOLDER: Subgroup AUROC gap analysis]

---

## 6. Discussion

### 6.1. Interpretation of Results

[PLACEHOLDER: Why does [winning method] outperform baselines?]

[PLACEHOLDER: Connection between empirical findings and theoretical expectations]

[PLACEHOLDER: Surprising results and potential explanations]

### 6.2. Implications for Clinical AI

[PLACEHOLDER: Practical recommendations for EHR foundation model design]

[PLACEHOLDER: Trade-offs between performance, efficiency, and interpretability]

[PLACEHOLDER: Considerations for clinical deployment]

### 6.3. Limitations

1. **Single-site evaluation**: All experiments use MIMIC-IV from a single institution. Cross-site validation on eICU, institutional EHRs, and international cohorts is essential to establish generalizability.

2. **Model scale**: Our ≈87M parameter model is substantially smaller than state-of-the-art foundation models. Optimal representation choices may differ at larger scales.

3. **Reference range limitations**: MIMIC-IV provides population-level reference ranges without demographic stratification. Age-, sex-, and race-specific reference intervals would improve clinical anchoring.

4. **Task coverage**: Four prediction tasks may not capture the full diversity of clinical applications. Few-shot evaluation on broader task sets would strengthen conclusions.

5. **Temporal scope**: Training on data from a single institution over a limited time period may not capture temporal distribution shifts common in clinical deployment.

### 6.4. Reflection on Clinical Semantic Preservation

[PLACEHOLDER: How well do the evaluated methods preserve clinical semantics?]

[PLACEHOLDER: Remaining challenges for clinically-meaningful representations]

[PLACEHOLDER: Future directions for semantic-aware encoding]

---

## 7. Conclusion

### Paragraph 1: Summary

[PLACEHOLDER: Restate the problem, contribution, and key findings]

This work addressed the understudied problem of input representation design for EHR foundation models. Through systematic evaluation of quantization granularity, encoding mechanics, and vocabulary semantics across 100 training runs on MIMIC-IV, we established empirical guidelines for representation choices. [PLACEHOLDER: Key quantitative findings].

### Paragraph 2: Broader Impact

[PLACEHOLDER: Implications for clinical AI field]

[PLACEHOLDER: Potential negative societal impacts and mitigations]

### Paragraph 3: Future Work

Several directions merit further investigation:

1. **Cross-site validation**: Evaluating representation choices on external datasets to establish generalizability beyond MIMIC-IV.

2. **Scale effects**: Investigating whether optimal representations differ for larger foundation models (1B+ parameters).

3. **Demographic anchoring**: Developing reference range annotations stratified by age, sex, and other demographic factors for more clinically-aligned binning.

---

## References

[1] Renc, P., Grzeszczyk, M. K., Oufattole, N., Goode, D., Jia, Y., Bieganski, S., McDermott, M. B. A., Was, J., Samir, A. E., Cunningham, J. W., Bates, D. W., & Sitek, A. (2025). Foundation Model of Electronic Medical Records for Adaptive Risk Estimation. *GigaScience*, 14, giaf107. https://doi.org/10.1093/gigascience/giaf107. Repository: https://github.com/ipolharvard/ethos-ares (MIT License). **Our MEDS extraction pipeline is adapted from this work.**

[2] Li, Y., Rao, S., Solares, J. R. A., Hassaine, A., Ramakrishnan, R., Canber, D., Zhu, Y., Rahimian, F., & Salimi-Khorshidi, G. (2020). BEHRT: Transformer for Electronic Health Records. *Scientific Reports*, 10, 7155.

[3] Rasmy, L., Xiang, Y., Xie, Z., Tao, C., & Zhi, D. (2021). Med-BERT: Pretrained contextualized embeddings on large-scale structured electronic health records for disease prediction. *npj Digital Medicine*, 4, 86.

[4] Odgaard, M., Klein, K. V., Thysen, S. M., Sørensen, H. T., Ehrenstein, V., & Krag, M. (2024). CORE-BEHRT: A Carefully Optimized and Rigorously Evaluated BEHRT. *arXiv preprint arXiv:2404.15201*.

[5] Ozarda, Y., & Sikaris, K. (2024). Reflections on current reference interval practices. *Clinical Chemistry and Laboratory Medicine*, 62(1), 17–28.

[6] Kotsiantis, S., & Kanellopoulos, D. (2006). Discretization techniques: A recent survey. *GESTS International Transactions on Computer Science and Engineering*, 32(1), 47–58.

[7] Golkar, S., Pettee, M., Eickenberg, M., Bietti, A., Krawezik, G., Cranmer, K., Dasgupta, S., Dey, B., Lemieux, G., Louppe, G., Mishra, S., Mohseni, K., Radev, D., Seljak, U., & Villar, S. (2023). xVal: A Continuous Number Encoding for Large Language Models. *arXiv preprint arXiv:2310.02989*.

[8] Kim, J., Lee, N., Kim, J., & Kim, K. (2025). MedRep: Medical Concept Representation for General Electronic Health Record Foundation Models. *arXiv preprint arXiv:2504.08329*.

[9] Burkhart, M. C., Ramadan, B., Liao, Z., Chhikara, K., Rojas, J. C., Parker, W. F., & Beaulieu-Jones, B. K. (2025). Foundation models for electronic health records: representation dynamics and transferability. *arXiv preprint arXiv:2504.10422*.

[10] Pang, C., Jeanselme, V., Choi, Y. S., Jiang, X., Jing, Z., Kashyap, A., Kobayashi, Y., Li, Y., Pollet, F., Natarajan, K., & Joshi, S. (2025). FoMoH: A clinically meaningful foundation model evaluation for structured electronic health records. *arXiv preprint arXiv:2505.16941*.

[11] Wornow, M., Thapa, R., Steinberg, E., Fries, J. A., & Shah, N. H. (2023). EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models. *arXiv preprint arXiv:2307.02028*.

[12] Dubey, A., et al. (2024). The Llama 3 Herd of Models. *arXiv preprint arXiv:2407.21783*.

[13] Johnson, A., Bulgarelli, L., Pollard, T., Gow, B., Moody, B., Horng, S., Celi, L. A., & Mark, R. (2023). MIMIC-IV (version 3.1). *PhysioNet*. https://doi.org/10.13026/kpb9-mt58

[14] Norouzi, M., Mikolov, T., Bengio, S., Singer, Y., Shlens, J., Frome, A., Corrado, G. S., & Dean, J. (2014). Zero-Shot Learning by Convex Combination of Semantic Embeddings. *ICLR*. arXiv:1312.5650.

[15] Kazemi, S. M., & Poupart, P. (2019). Time2Vec: Learning a Vector Representation of Time. *NeurIPS 2019*.

[16] Jiang, L. Y., Chen, A., Han, X., et al. (2025). Generalist Foundation Models Are Not Clinical Enough for Hospital Operations. *arXiv preprint*.

[17] Steinberg, E., Fries, J. A., Xu, Y., & Shah, N. (2023). MOTOR: A Time-To-Event Foundation Model For Structured Medical Records. *arXiv preprint*.

[18] Shao, C., Li, D., Meng, F., & Zhou, J. (2025). Continuous Autoregressive Language Models. *arXiv preprint*.

[19] Xie, E., Martinez, R. R., Chang, W., & Ballinger, B. (2025). JETS: A Self-Supervised Joint Embedding Time Series Foundation Model for Behavioral Data in Healthcare. *arXiv preprint*.

[20] Van Calster, B., Collins, G. S., Vickers, A. J., et al. (2025). Evaluation of performance measures in predictive artificial intelligence models to support medical decisions. *The Lancet Digital Health*.

---

## Appendix A: Extended Technical Details

### A.1. Clinically-Anchored Quantization Algorithm

```python
def compute_anchored_breaks(values, ref_lower, ref_upper, allocation="5-10-5"):
    """Compute bin boundaries anchored to reference ranges."""
    below = values[values < ref_lower]
    within = values[(values >= ref_lower) & (values <= ref_upper)]
    above = values[values > ref_upper]
    
    if allocation == "5-10-5":
        n_below, n_within, n_above = 5, 10, 5
    elif allocation == "10-10-10":
        n_below, n_within, n_above = 10, 10, 10
    
    breaks = []
    breaks.extend(np.quantile(below, np.linspace(0, 1, n_below+1)[1:-1]))
    breaks.append(ref_lower)
    breaks.extend(np.quantile(within, np.linspace(0, 1, n_within+1)[1:-1]))
    breaks.append(ref_upper)
    breaks.extend(np.quantile(above, np.linspace(0, 1, n_above+1)[1:-1]))
    
    return np.unique(breaks)
```

### A.2. Soft Discretization Implementation

[PLACEHOLDER: Full implementation details]

### A.3. Time2Vec Architecture

[PLACEHOLDER: Architecture diagram and hyperparameters]

---

## Appendix B: Pre-training Details

[PLACEHOLDER: Pre-training data statistics]

[PLACEHOLDER: Convergence curves]

[PLACEHOLDER: Computational resources used]

---

## Appendix C: Baseline Implementation Details

[PLACEHOLDER: Hyperparameter search ranges]

[PLACEHOLDER: Modifications to published methods]

[PLACEHOLDER: Links to implementations]

---

## Appendix D: Dataset Details

### D.1. MIMIC-IV v3.1 Statistics

| Category | Count |
|----------|-------|
| Total patients | 364,627 |
| Laboratory codes | 1,128 |
| Codes with ref ranges | 392 (34.8%) |
| Events with ref ranges | 114,135,372 (80.3%) |
| Top 200 codes coverage | 97.06% |

### D.2. Tokenization Statistics (fms-ehrs)

Validated on MEDS-formatted MIMIC data using `fms_ehrs/config/mimic-meds-ed.yaml` (see `methods/data-columns.md` for full execution provenance):

| Statistic | Value |
|-----------|-------|
| Vocabulary size | 19,363 tokens |
| Timelines (≥24h stay filter) | train/val/test = 297,817 / 41,869 / 85,530 |
| Timelines (24h-cut) | train/val/test = 296,209 / 41,652 / 85,057 |
| Mean unpadded tokens (train) | 2,041.07 (median 296; IQR 132–942) |
| Tokenization runtime | 41m 28s (log: `slurm/output/local_tokenize_20260107_032155.log`) |
| Memory requirement | Observed peak RSS ≈ 58 GB; recommend 150 GB job limit for safety |
| Configuration | deciles, time spacing tokens, non-fused, `max_padded_len=1024`, `include_24h_cut` |

This represents the first non-CLIF configuration for `fms-ehrs`, demonstrating successful adaptation to MEDS-formatted data from the ETHOS extraction pipeline.

**Sequence length distribution (train split, deciles/non-fused/time tokens)**:
mean 2,041.07; median 296; p90 4,286; p95 8,452; p99 32,732; max 569,422.
Coverage at fixed lengths: 75.9% ≤ 1,024 tokens; 82.6% ≤ 2,048 tokens.
These statistics justify `max_seq_length=1024` for tractable 24h runs while
making the truncation trade-off explicit.

### D.3. Minimal Preprocessing and QC (NeurIPS-style transparency)

We apply a conservative, pre-registered cleaning policy that avoids hidden
filtering while ensuring schema validity and temporal fidelity. Each rule is
deterministic and applied uniformly across all experiments, consistent with
NeurIPS reproducibility checklist norms that emphasize transparent preprocessing
and explicit exclusion criteria.

**Evidence-based inspection (tokenized timelines)**:
We manually inspected the five shortest (length 10–12) and five longest
(length 339k–569k) timelines from the train split. Shortest timelines contain
only demographic and administrative tokens (e.g., sex/race/insurance/admit/discharge),
with **no numeric values** (numeric fraction = 0.0). Longest timelines contain
dense vitals, labs, medications, ICU transfers, and time-spacing tokens; all
begin with `TL_START` and end with `TL_END`, and their numeric-value fraction
is stable (≈0.27–0.32). We observed no malformed sequences, implying that the
extremes reflect true long-stay or high-frequency event admissions rather than
tokenization artifacts.

We also reviewed the ICU vital-sign tokenization generated from `chartevents`.
Vitals are encoded as `VITAL//{itemid}//{unit}` tokens with aligned numeric
values and time-spacing tokens. In the longest timelines, repeated vital
measurements retain consistent `{itemid, unit}` identities and appear in
chronological order, indicating that high-frequency charting increases length
but does **not** collapse distinct clinical semantics. Therefore, we retain
fine-grained vital events rather than aggregating them, and rely on the explicit
time-spacing tokens to preserve temporal structure.

**QC rules and justifications**:
1. **Schema validity** (drop rows missing `subject_id`, `time`, or `code`):  
   *Justification*: these fields define the MEDS event identity; missing values
   cannot be imputed without unverifiable assumptions.  
   *Evidence*: tokenized sequences show intact `TL_START/TL_END` structure even
   at extremes, indicating schema completeness post-extraction.

2. **Timeline integrity** (drop admissions with zero events after filtering;
   enforce chronological ordering under `storetime`):  
   *Justification*: empty timelines are undefined for autoregressive modeling;
   storetime ordering prevents label leakage.  
   *Evidence*: shortest observed timelines (length 10–12) are non-empty and
   semantically coherent, so no additional pruning is warranted.

3. **Value sanity** (remove non-finite numeric values; keep zeros):  
   *Justification*: NaN/Inf propagate to loss/gradients and create invalid
   embeddings; zeros are valid clinical measurements.  
   *Evidence*: numeric fraction remains plausible in the longest sequences and
   zero numeric fraction in the shortest sequences is expected (no numeric events).

4. **Reference range completeness** (anchored binning only when both
   `ref_range_lower` and `ref_range_upper` are present; otherwise fall back to
   population quantiles for that code):  
   *Justification*: partial ranges are not clinically interpretable; mixing
   anchored and unanchored bins within a code would bias discretization.  
   *Evidence*: MIMIC-IV reference ranges are always paired or absent (Appendix D.2).

5. **Deterministic filtering** (all criteria fixed *a priori* and applied
   uniformly across cohorts and experiments):  
   *Justification*: prevents “p‑hacking” via selective exclusion.  
   *Evidence*: we do not remove extreme‑length timelines; instead we document
   truncation coverage at fixed `max_seq_length` and apply it uniformly.

**Non‑numeric columns**:
All non‑numeric event types (diagnoses, procedures, medications, transfers,
etc.) are encoded as categorical code tokens and do **not** enter the numeric
value channel. Only rows with valid `numeric_value` populate the aligned numeric
array used for soft discretization or continuous encoding; all other positions
are masked as null/NaN. This preserves the discrete semantic stream while
ensuring continuous encoders operate only on valid numeric measurements.

### D.4. Reference Range Validation

**Critical Finding**: Reference ranges in MIMIC-IV v3.1 are always paired (both `ref_range_lower` and `ref_range_upper` present) or both missing. Zero instances of partial ranges across 142M events.

### D.5. Clinical Validation: Glucose Example

```
Lab: Glucose (itemid 50931)
Reference Range: [70.0, 105.0] mg/dL
Clinically-anchored ventile breaks:

Below (<70):  [54, 61, 65, 67, 70]         = 5 bins
Within:       [79, 84, 87, 90, 92, 95,     = 10 bins
               97, 100, 103, 105]
Above (>105): [115, 127, 146, 184]         = 5 bins
```

---

## Appendix E: Additional Results

[PLACEHOLDER: Full result tables for all metrics and tasks]

[PLACEHOLDER: Per-task breakdowns]

[PLACEHOLDER: Failure case analysis]

---

## Appendix F: Code and Data Availability

**Repositories**:
- `input-representation-benchmark/`: MEDS extraction adapter, experiment orchestration, MEDS-specific utilities, manuscript sources
- `fms-ehrs/`: Tokenization framework, model training, representation mechanics, and evaluation utilities

**License**: MIT

**Dependencies**:
- MEDS extraction pipeline: Adapted from ETHOS-ARES [1] (MIT License, Copyright © 2024 Paweł Renc)
- Tokenization: `fms-ehrs` framework
- Model: HuggingFace Transformers

**Reproduction instructions** are maintained as an operational document in the top-level `README.md`
(SLURM submission details, environment setup, and CLIF preprocessing), to keep this manuscript focused on
scientific claims, methods, and results.

**Computational Requirements**:
- **GPU (training)**: 1 GPU per job; max 8 concurrent jobs (policy). [PLACEHOLDER: insert GPU model(s) once recorded]
- **CPU/RAM (MEDS extraction/tokenization)**: CPU-heavy; we provision 8 CPUs and ~150GB RAM for extraction/tokenization jobs.
- **CPU/RAM (training jobs)**: we provision 4 CPUs and 64GB RAM per training job.
- **Storage**: [PLACEHOLDER: quantify disk required for raw MIMIC + MEDS parquet + tokenized artifacts + checkpoints]
- **Runtime**: hardware-dependent; we will report measured wall-clock time and GPU-hours per run alongside performance metrics.

---

## Appendix G: Parameter Deltas for Exp2 Grids

We report **additional parameters** introduced by Exp2 representation and
temporal modules (over the base LM). Counts exclude non-trainable buffers.
Let $H$ denote the LM hidden size (here $H=1024$).

| Component | Candidate setting | $\Delta$ parameters (formula) | $\Delta$ parameters (H=1024) |
|----------|-------------------|-------------------------------|-------------------------------|
| Time2Vec | $d=32$ | $2d + dH + 3H$ | 35,904 |
| Time2Vec | $d=64$ | $2d + dH + 3H$ | 68,736 |
| Time2Vec | $d=128$ | $2d + dH + 3H$ | 134,400 |
| xVal | scales $S=1$ | $S \cdot H$ | 1,024 |
| xVal | scales $S=3$ | $S \cdot H$ | 3,072 |
| Soft discretization | $N^*$ bins | $N^* \cdot H$ | $N^* \times 1{,}024$ (e.g., 20 → 20,480) |

**Notes**:
- Time2Vec parameters include the projection to $H$ and layer norm.
- $N^*$ is fixed to the Exp1 winner for final Exp2 runs; if Exp2 runs before Exp1,
  $N^*$ is set to the pre-registered default (ventiles) and rerun if Exp1 selects
  a different granularity.
