# Third-Party Notices

This document contains attribution and license information for third-party software
and code used in the Input Representation Benchmark.

---

## ETHOS-ARES

**Repository:** https://github.com/ipolharvard/ethos-ares

**License:** MIT License

**Copyright:** (c) 2024 Paweł Renc

### Components Used

The MEDS (Medical Event Data Standard) extraction pipeline located in
`benchmarks/mimic-meds-extraction/meds_pipeline/` is adapted from the ETHOS-ARES
repository. This includes:

- `pre_MEDS.py` - Pre-MEDS data wrangling for MIMIC-IV
- `configs/pre_MEDS.yaml` - Hydra configuration for pre-MEDS processing
- `configs/extract_MIMIC.yaml` - MEDS extraction pipeline configuration
- `configs/local_parallelism_runner.yaml` - Parallelization configuration

### Modifications

The following modifications were made for the input-representation-benchmark:

1. **Split fractions**: Changed from 90/10 (train/test) to 70/10/20 (train/val/test)
   to align with the benchmark's evaluation methodology.
2. **Attribution headers**: Added source attribution and license information to
   each file.
3. **Event configuration**: Uses a custom `event_configs_v3.1_full.yaml` with
   `storetime` semantics (instead of `charttime`) to ensure event ordering reflects
   information availability.

### Citations

If you use the MEDS extraction pipeline, please cite:

**Primary Citation (ETHOS-ARES):**

```bibtex
@article{10.1093/gigascience/giaf107,
    author = {Renc, Pawel and Grzeszczyk, Michal K and Oufattole, Nassim and Goode, Deirdre and Jia, Yugang and Bieganski, Szymon and McDermott, Matthew B A and Was, Jaroslaw and Samir, Anthony E and Cunningham, Jonathan W and Bates, David W and Sitek, Arkadiusz},
    title = {Foundation model of electronic medical records for adaptive risk estimation},
    journal = {GigaScience},
    volume = {14},
    pages = {giaf107},
    year = {2025},
    month = {09},
    issn = {2047-217X},
    doi = {10.1093/gigascience/giaf107},
    url = {https://doi.org/10.1093/gigascience/giaf107},
}
```

**Original ETHOS Paper:**

```bibtex
@article{renc_zero_2024,
    title = {Zero shot health trajectory prediction using transformer},
    volume = {7},
    issn = {2398-6352},
    url = {https://www.nature.com/articles/s41746-024-01235-0},
    doi = {10.1038/s41746-024-01235-0},
    journal = {npj Digital Medicine},
    author = {Renc, Pawel and Jia, Yugang and Samir, Anthony E. and Was, Jaroslaw and Li, Quanzheng and Bates, David W. and Sitek, Arkadiusz},
    month = sep,
    year = {2024},
    pages = {1--10},
}
```

### Full License Text

```
MIT License

Copyright (c) 2024 Paweł Renc

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## MEDS Transforms

**Repository:** https://github.com/mmcdermott/MEDS_transforms

**License:** MIT License

The MEDS extraction pipeline depends on the MEDS_transforms package for the core
ETL operations. This package is installed as a dependency and used via its
`MEDS_transform-runner` command.

### Citation

```bibtex
@software{mcdermott2024meds_transforms,
    author = {McDermott, Matthew B.A.},
    title = {MEDS Transforms},
    url = {https://github.com/mmcdermott/MEDS_transforms},
    year = {2024}
}
```

---

## fms-ehrs

**Repository:** (sister repository)

**Components Used:**

The tokenization framework and representation methods used in this benchmark
integrate with the `fms-ehrs` repository, which provides:

- Tokenizer base classes (`fms_ehrs.framework.tokenizer_base`)
- Quantization strategies (ventiles, clinical anchoring)
- Soft discretization methods
- Continuous encoding methods
- Time2Vec temporal encoding

This is a sister repository and should be installed alongside this benchmark.
See the README for setup instructions.

---

## Model Architecture References

### LLaMA 3.2 Base Architecture

**Citation:**

```bibtex
@article{grattafiori2024llama3,
    title = {The Llama 3 Herd of Models},
    author = {Grattafiori, Aaron and others},
    journal = {arXiv preprint arXiv:2407.21783},
    year = {2024}
}
```

### Scaled-Down Configuration (≈87M parameters; exact depends on vocabulary size)

The specific scaled-down model configuration (hidden_size=1024, intermediate_size=2048,
num_layers=8, num_heads=8) is borrowed from:

**Citation:**

```bibtex
@article{burkhart2025quantifying,
    title = {Quantifying surprise in clinical care: Detecting highly informative events in electronic health records with foundation models},
    author = {Burkhart, Michael C. and Ramadan, Bashar and Solo, Luke and Parker, William F. and Beaulieu-Jones, Brett K.},
    journal = {arXiv preprint arXiv:2507.22798},
    year = {2025}
}
```

---

## Time2Vec

**Citation:**

```bibtex
@article{kazemi2019time2vec,
    title = {Time2Vec: Learning a Universal Representation of Time},
    author = {Kazemi, Seyed Mehran and Goel, Rishab and Jain, Kshitij and Kobyzev, Ivan and Sethi, Akshay and Forsyth, Peter and Poupart, Pascal},
    journal = {arXiv preprint arXiv:1907.05321},
    year = {2019}
}
```
