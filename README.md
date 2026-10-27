# Integrated Database and Large Language Modelling for Clinical Support
# Distinction 81/100
A feasibility study: can data from different hospitals be harmonised into one format, used to train a model that suggests a breast cancer treatment group, and paired with a guideline-grounded explanation that is honest about its own limits?

MSc Data Science dissertation, University of York. Author: Pallavi Bharadwaj. Supervisor: Dr Kofi Appiah.

> **Research prototype only. This is not a clinical tool and must not be used to make treatment decisions.**

---

## The problem

Breast cancer multidisciplinary team (MDT) meetings need several specialists in the same room to agree a treatment plan. Models that predict MDT recommendations have mostly been built and tested inside single, well-resourced hospitals. This project asks what happens when the data comes from places that record the same facts in different ways.

## Research questions

1. How accurately can a classifier predict the recorded MDT recommendation, and how does that depend on the data available?
2. Does a coarser, clinically grouped target change what is achievable?
3. Can a mapping layer carry the model to a second hospital without retraining?
4. Does the model behave coherently on a different continent's data?

## How it works

```
KATH (Ghana)  \
SGMC (Ghana)   >--> Harmonise --> Predict --> Explain
SEER (USA)    /     (map columns,  (Random Forest   (NICE NG101 passages
                     then values)   or gradient      + a small local LLM;
                                    boosting)        the classifier decides)
```

| Stage | What it does |
|---|---|
| **Data** | KATH (5,000 patients, Ghana), SGMC (1,000, Ghana), SEER (4,024, USA). |
| **Harmonise** | Two stages: column mapping, then value mapping (for example, SGMC's HER2 score of 0 to 3+ becomes Negative, Equivocal or Positive). Value counts are checked before and after each mapping. |
| **Predict** | Random Forest and `HistGradientBoostingClassifier` on nested feature sets of 5, 15 and 32 variables. |
| **Explain** | Retrieval-augmented generation over NICE guideline NG101. The classifier is the only decision-maker; the language model only explains. |

### Data split

From KATH's 5,000 patients, 2,911 had all 32 variables recorded. These were split 70 / 15 / 15 (stratified by target, seed 42): 2,038 train, 436 validation, 437 test. The same split is used for every model in the controlled comparison.

## Key results

**Better data beat a cleverer model.** Fifteen-class test accuracy on the same 2,911 patients and the same 437 test patients:

| Variables | Random Forest | Gradient boosting |
|---|---|---|
| 5 | 56.06% | 50.34% |
| 15 | 59.95% | 54.92% |
| 32 | 69.79% | 75.29% |

Richer data moved accuracy by 13.7 to 25.0 points; switching algorithm moved it by 5.0 to 5.7 points. That is a ratio of about 2.4 to 4.5 (my own calculation from a single split, so treat it as a guide).

**Grouping the target into six clinical groups** (Surgery, Systemic-Chemo, Endocrine, Radiotherapy, Palliative/Supportive, Further-workup):

- 86.73% test accuracy, macro-F1 0.742
- 87.60% (SD 0.94%) under five-fold cross-validation
- Fifteen-class equivalent: 75.29% accuracy, macro-F1 0.566

Part of this gain is arithmetic: grouping merges classes that are hard to tell apart.

**Harmonisation does not consistently cost predictive power.** Random Forests trained on SGMC's own data to predict whether chemotherapy was recommended (992 patients, five random 75/25 splits): macro-F1 0.643 on native values and 0.671 on harmonised values. The gap is smaller than the run-to-run variation (SD 0.052 and 0.026), so the claim is "no consistent cost", not "harmonising helps".

**RAG explanations.** I hand-reviewed 30 generated explanations. A redesigned prompt (five labelled sections, a rule never to imply the guideline confirms the prediction for this patient, and a required Limitations section) removed four failure patterns, including fabricated treatment histories (6 of 30 to 0). The cause is a hypothesis supported by the result, not proof.

## Limitations and withdrawn results

I report these because they matter more than the headline numbers.

- **Withdrawn: 77.4% SGMC agreement.** A rule-based check scored the KATH-trained model at 77.4% agreement on 992 SGMC patients, but a constant predictor scores 80.2%, so the check cannot tell a good model from a lazy one. The transfer shows the mapping works (predictions spread across nine of fifteen categories), but no accuracy is claimed.
- **Withdrawn: survival as proof of clinical validity.** On SEER, adjusting for stage reduced the association's chi-square from 97.68 to 6.44 (about 93%). The remaining difference is small and cannot be read causally.
- **Further-workup is not learnable.** 0 of 20 test patients were predicted correctly; it behaves as a catch-all label.
- **Calibration.** Brier score 0.226, ECE 0.092. 383 of 437 predictions (87.6%) fall in the highest confidence bin at mean confidence 0.993 against 0.906 observed accuracy, so the model is overconfident. A recalibration remedy is proposed but not applied.
- **Complete-case cohort.** The 2,911 patients with all 32 variables are not a random sample of KATH.
- **Cross-hospital tests use only the five base variables** (four shared with SEER), because the other sites do not record the rest. The SEER model is close to a stage lookup.
- **RAG residue.** Some explanations still contain unsupported wording (for example, "which aligns with the ML prediction" in 7 of 30). Explanations need checking by a person.
- **Guideline scope.** NG101 covers early and locally advanced breast cancer, so retrieved text can fit poorly for stage IV patients.

## Repository contents

```
.
├── notebooks/        Main notebook: pipeline, evaluation, RAG layer
├── report/           Dissertation PDF
├── README.md
└── requirements.txt  (see Setup)
```

> Adjust this section to match what you actually upload.

## Setup

The notebook was developed in Google Colab with Python 3. Main libraries:

```
pandas, numpy, scikit-learn, pypdf, sentence-transformers, transformers, torch
```

RAG components: `all-MiniLM-L6-v2` for embeddings, top-5 retrieval over 232 chunks from 108 pages (180 words with 40 overlap), and `Qwen2.5-1.5B-Instruct` as the explanation model. The guideline PDF path in the notebook is a Google Drive path; change it to wherever you save the NICE NG101 PDF.

## Data availability

**No patient data is included in this repository.** KATH and SGMC are not redistributable, and SEER must be obtained from its own source under its data-use terms. The code expects those files to be supplied by the user.

## Guideline choice

NICE NG101 was used because its open licence permits this use. NCCN guidelines were ruled out because their terms prohibit use with AI tools.

## Author

Pallavi Bharadwaj, MSc Data Science, University of York.
Portfolio: https://pallavii-bharadwaj.github.io

## Licence

Add a licence here (for example MIT for the code). Dissertation text and any third-party material keep their own terms.
