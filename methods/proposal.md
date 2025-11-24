Benchmarking Input Representation Methods for EHR Foundation Models

1. Introduction
Electronic Health Records (EHRs) consist of heterogeneous data streams, combining discrete diagnostic codes with continuous physiological signals. Translating this multi-modal input into sequences compatible with transformer architectures poses a fundamental representation challenge. Current state-of-the-art frameworks, such as ETHOS-ARES, use methods like the decile binning to circumvent this limitation, but the optimal encoding of continuous values for clinical semantics remains an open research question. In this paper, we propose diverse input representation methods and benchmark them. Our aim is to study how different layers of input representation—quantization granularity, representation mechanism (e.g., discrete vs. continuous embedding), and vocabulary structure—affect model generalization and reasoning in clinical contexts.

2. Experimental Design
This study evaluates the impact of several different input representations on a series of standard clinical prediction tasks (in-hospital mortality, long length of stay (≥7 days), ICU admission after 24 hours, and IMV event after 24 hours).

2.1. Experiment 1: Granularity and Semantic Anchoring
Objective: To determine the optimal binning method for several standard prediction tasks and to evaluate the impact of using "clinically anchored" bins in which the binning considers the reference ranges for the quantities being binned.

Conditions:
* Decile: Baseline (ETHOS-ARES)
* Ventile: 20-bin quantization
* Clinically Anchored Ventile: Partitioning the value space into three disjoint regions with a 5-10-5 allocation:
    * Below Normal (x < L): 5 bins.
    * Within Normal (L ≤ x ≤ U): 10 bins (providing higher resolution for normal physiological variations).
    * Above Normal (x > U): 5 bins.
* Trentile: 30-bin quantization
* Clinically Anchored Trentile: Partitioning the value space into three disjoint regions with a 10-10-10 allocation
* Percentile: 100-bin quantization

2.2. Experiment 2: Representation Mechanics
Objective: Compare how well discrete tokenization works against continuous and hybrid representation methods when using the optimal technique identified in Experiment 1. Discrete tokens are standard for transformers, and yet they are not a natural fit for continuous data. This experiment also begins to explore how methods that are more numerically "friendly" could be beneficial in a clinical setting where efficiency could lead to significant gains in the utility of these models.

Conditions:
* Optimally-binned Discrete Tokens: Standard categorical embedding of binned values.
* Fused Tokens: Combining code and value into a single token to improve sequence efficiency (reducing sequence length).
* Convex Combinations: Representing values as a weighted interpolation of adjacent bin embeddings (soft discretization) to model continuity.
* Learned Encoders: Direct projection of continuous values using MLPs or similar encoding mechanisms (e.g., xVal).

2.3. Experiment 3: Vocabulary Semantics and Generalizability
Objective: Evaluate the trade-off between vocabulary granularity and semantic regularization by comparing the performance of two distinct foundation models trained on the same MIMIC-IV data. While both models take data in the standardized MEDS format, they will differ in the granularity of their code vocabularies. This experiment investigates whether the semantic aggregation inherent in a Common Data Model (CDM) yields a loss of critical clinical contexts compared to raw source codes, or if it provides advantageous regularization that improves model generalization.

Conditions:
* MIMIC-IV Native Vocabulary: The model is trained on MEDS data where the code column contains raw, source-specific identifiers (e.g., `LAB//50931`). This represents the maximum available granularity but is specific to the institution.
* CLIF Standardized Vocabulary: The model is trained on MEDS data where the code column contains standardized CLIF concepts (e.g., `LAB//glucose_serum`). This represents a semantic aggregation step where multiple source codes are mapped to single canonical concepts, reducing vocabulary size and enforcing semantic regularization.

3. Methodology: The Input Representation Framework
This project establishes a reproducible benchmarking environment by extending the ETHOS-ARES codebase (commit 2d54383). The codebase is compatible with the upstream repository while allowing for the modular injection of diverse quantization logic.

3.1. Codebase Architecture
The framework uses the ETHOS-ARES MEDS extraction and tokenization pipeline without modifying the core repository. Our experiments are conducted with these principles:

* Custom Event Configuration: Extending the MEDS extraction configuration to retrieve the additional clinical metadata needed, such as ref_range_lower and ref_range_upper from the MIMIC-IV labevents table.
* Modular Pipeline: Replacing the baseline decile quantization module with custom implementations to add our methods of interest
* Clean-Extension Principle: All custom logic only resides in our repository (input-representation-benchmark).

3.2. Preliminary Findings on MIMIC-IV v3.1
* Coverage: 80.3% of lab events (114M) have reference ranges and are compatible with the anchored method.
* Granularity: 155 of the top 203 lab codes successfully achieved full 20-bin resolution under strict filtering.
* Validation: Glucose (itemid 50931) verified to produce clinically meaningful breaks at standard reference boundaries (L=70, U=105).

4. Significance of the Study
This study aims at providing rigorous benchmarking and thus comprehensive understanding of input representations in clinical foundation models. By systematically adjusting granularity, mechanics of representation, and semantic ontology, we hope to provide empirical guidelines for constructing robust, generalizable EHR embeddings.

5. References / Links
Data model and description: https://mimic.mit.edu/docs/iv/modules/
MEDS FM: https://github.com/mmcdermott/MEDS_EIC_AR
Our FM: https://arxiv.org/pdf/2504.10422
