# InfoSign

Code for **Task-specific information signatures reveal conditional modality dependence in multimodal clinical AI**.

InfoSign combines nine AI-READI data sources for clinical prediction and examines their contributions through modality ablation and concept interventions.

| Experiment | Evaluation |
|---|---|
| **Full Clinical model** | Downstream classification, basic-only clinical inputs at inference, and 11 modality ablations |
| **Basic Clinical model** | Downstream classification and the same 11 modality ablations |
| **Mixed CBM** | Full/masked-input classification, predicted-concept restoration, and measured-concept supplementation |

## Setup and data

Use Python 3.12 and a compatible [CUDA PyTorch/torchvision build](https://pytorch.org/get-started/locally/), then install the dependencies:

```bash
git clone https://github.com/chili-lab/InfoSign.git
cd InfoSign
python -m pip install -r requirements.txt
```

**Participant data are not included.** Obtain **AI-READI v3.0.0** from the [official dataset page](https://doi.org/10.60775/fairhub.3) using its [download instructions](https://docs.aireadi.org/docs/3/preliminary/azure_storage/azure-download). The study uses 2,280 participants and the provided `recommended_split` assignments.

Place the dataset under `data/aireadi/`, preserving all manifests and referenced files:

```text
data/aireadi/
├── participants.tsv
├── clinical_data/              # person, measurement, observation,
│                              # condition_occurrence, procedure_occurrence CSVs
├── cardiac_ecg/
├── environment/
├── retinal_flio/
├── retinal_oct/
├── retinal_octa/
├── retinal_photography/
├── wearable_activity_monitor/
└── wearable_blood_glucose/
```

For another data location, edit [`config.yaml`](config.yaml): set `preprocessing.dataset_root` and `study.clinical_dir`, and keep both `source_root` settings pointed at the same prepared-input directory. Personal overrides can be saved in `config.local.yaml` and passed with `--config config.local.yaml`.

## Run

From the repository root:

```bash
# Prepare labels, native inputs, train-only scalers, and shared image features.
python -u prepare.py --devices 0 1 2 3

# Train and evaluate all experiments on validation and test sets.
python -u run.py --devices 0 1 2 3
```

Use `--devices 0` for one GPU. Preparation preserves participant splits and caches features from a frozen ImageNet-pretrained ResNet-18. Weights download on first use; offline users can set `preprocessing.retinal_weights_path`. Already prepared inputs can be used directly with `run.py`.

Completed units are reused when rerunning the same command. Use `prepare.py --dry-run` to preview preparation, or `--stage cache` to resume after native preprocessing. Use a new output directory when changing experiment settings. To regenerate reports without training:

```bash
python run.py --config outputs/study_without_observation/runtime.yaml --stage report
```

### Outcome labels

Outcome labels are derived from the downloaded clinical records during preparation; they are not included in this repository. MoCA labels are built during native preprocessing (`--stage raw`). The two cardiovascular outcomes and the three history outcomes are built together with the input registry (`--stage cache`). The history outcomes (`osteoporosis_history_binary`, `kidney_history_binary` and `lung_history_binary`) use `preprocessing/build_history_task_labels.py` and `preprocessing/configs/history_tasks.json`, read both `observation.csv` and `condition_occurrence.csv`, and are written to `outputs/prepared/history_labels/<task>_labels.csv`.

Training reads the `y` column from each task's master CSV registered in `outputs/prepared/inputs.json`, which the cache stage creates. If native preprocessing is already complete, `python -u prepare.py --stage cache --devices 0` builds the registry and reuses existing participant caches. CBM files under the study output's `labels/` directory contain concept supervision, not outcome labels.

## Experiments

The six outcomes are **MoCA < 26**, **broad cardiovascular conditions** (hypertension, dyslipidemia, circulation problems, retinal vascular occlusion, or other heart issues), **major cardiovascular events** (myocardial infarction or stroke), and histories of **osteoporosis**, **kidney problems**, and **lung problems**.

Full Clinical uses eligible structured clinical features and eight nonclinical modalities. Basic Clinical retains 23 encoded basic patient features and the same eight modalities. Each model is evaluated after removing each of the nine sources individually, all retinal sources together, or both wearable sources together: **11 ablations**.

Mixed CBM freezes the Full model and trains a shared concept predictor/classifier on paired full and masked representations. **Masking removes nonbasic clinical features and their availability indicators**, retaining basic information and the other eight modalities. From this masked baseline:

- **Predicted restoration** replaces one Concept ID's channels with the same participant's full-input concept predictions, probing internal concept dependence. Restoring all predicted IDs simultaneously reproduces full-input CBM predictions.
- **Measured supplementation** supplies available, encodable recorded values for learnable concept items, estimating the potential value of additional clinical records. Missing or unencodable items retain masked predictions.

Channels sharing a Concept ID are changed together. [Concept definitions](infosign/cbm/concepts.json) retain eligible Measurement, Condition, Device, and Procedure concepts with task-specific outcome exclusions. Observation-domain concepts are excluded, but `observation.csv` remains required for labels and eligible concepts stored there.

Training uses seed 42 and training-only minority oversampling. Full and Basic Clinical checkpoints are selected by validation AUROC, and Mixed CBM checkpoints by the mean of full- and masked-input validation AUROC (`backbone_checkpoint_metric` and `cbm_checkpoint_metric` in `config.yaml`). Validation BCE breaks ties, and fixed classification thresholds maximize validation F1. Validation/test retain natural class proportions. Reports include AUROC, positive-class F1, balanced accuracy, Brier scores, and 10,000 paired, outcome-stratified bootstrap resamples; intervals do not adjust for concept selection or training-seed variability.

### Validation-selected concept recovery

After `run.py` has completed, select Concept IDs on validation and evaluate the same IDs on test:

```bash
python -u run.py --config outputs/study_without_observation/runtime.yaml \
  --stage concept-validation --devices 0 1 2 3
```

No model is retrained. For each task, every Concept ID is restored on its own in the masked Mixed CBM, and IDs whose validation ΔAUROC exceeds μ + `selection_sd`·σ are selected, where μ and σ are the mean and population SD over all evaluated IDs (`selection_sd: 1.0` in `config.yaml`). Predicted and measured recovery are selected separately. Each selected set is then restored jointly and compared with the same model under basic and full clinical inputs, using its saved masked-validation threshold. With `restore_all_concepts: true`, simultaneous recovery of all Concept IDs is also reported. Test results never change the selection; negative test effects and empty selections are retained.

Results are written to `validation_selected/` in the study output directory:

- `tables/CONCEPT_SELECTION_val.csv`: every candidate ID, its validation gain, the cutoff and the selection flag.
- `tables/SELECTED_CONCEPT_RECOVERY_{val,test}.csv`: single-ID effects of the selected IDs with paired 95% CIs.
- `tables/JOINT_CONCEPT_RECOVERY_{val,test}.csv`: full, basic, joint and all-concept conditions, with absolute metrics, gains over basic inputs and remaining gaps to full inputs (AUROC, F1, balanced accuracy, Brier).
- `tables/SUMMARY_VALIDATION_SELECTED_{val,test}.md`: a readable summary of the above.
- `<task>/PREDICTIONS_{val,test}.csv`: participant-level probabilities for every evaluated condition.

Use `--stage concept-validation-report` to regenerate the summaries without recomputing effects. To combine tasks from several completed study directories, pass `--cbm-roots ROOT_A ROOT_B` (the first root containing a task's `mixed_cbm/model.pt` is used) together with a new `--output-root`. Selection is descriptive, not a significance test, and validation-based selection does not undo earlier exploratory use of the test set.

## Outputs and code

Prepared data are stored in `outputs/preprocessed/` and `outputs/prepared/`. Experiment reports are under `outputs/study_without_observation/tables/`, with validation and test versions:

| Output | Contents |
|---|---|
| `SUMMARY_test.md` | Classification, modality ablations, and all-concept restoration |
| `SUMMARY_CONCEPT_RECOVERY_test.md` | Both concept interventions; IDs highlighted by test ranking (exploratory) |
| `MODALITY_ABLATION_{full,basic}_test.csv` | All modality removals |
| `{PREDICTED,MEASURED}_CONCEPT_RECOVERY_{ALL,SELECTED}_test.csv` | All individual-ID effects and selected concepts |
| `MIXED_CBM_PERFORMANCE_test.csv`, `ALL_CONCEPTS_RESTORED_test.csv` | Full/masked CBM performance and simultaneous predicted restoration |
| `concept_diagnostics/` | Concept quality, probability diagnostics, and observed-label-subset results |
| `../validation_selected/tables/` | Validation-selected concept recovery (see above) |

| Location | Purpose |
|---|---|
| `prepare.py`, `run.py`, `config.yaml` | Preparation, experiment execution, and settings |
| `preprocessing/` | Clinical, ECG, sensor, and retinal preprocessing |
| `infosign/encoders.py`, `infosign/train.py` | Modality encoders, fusion, and Full/Basic training |
| `infosign/cbm/` | Concept supervision, Mixed CBM, interventions, and reports |
| `infosign/analysis.py`, `infosign/stats.py`, `infosign/report.py` | Ablations, statistics, and reporting |
