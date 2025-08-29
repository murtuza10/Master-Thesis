## Master Thesis: Comparative Evaluation of NER Techniques for Agriculture Entities 

A complete, reproducible pipeline for domain-specific Named Entity Recognition (NER) over OpenAgrar/Bonares corpora. It covers:
- Data extraction from INCEpTION CAS/XMI → text/BIO
- Zero-/Few-shot prompting with multiple LLMs
- Fine Tuning Encoder Models
- Classic and LLM fine-tuning
- Vector embeddings workflow
- Evaluation (strict, partial, embeddings-assisted)
- HPC (SLURM) job scripts and logs


### Quick Start
1) Create environment
```bash
conda create -n ner-pipeline python=3.10 -y && conda activate ner-pipeline
pip install -U pip
pip install pandas numpy scikit-learn tqdm ujson datasets transformers accelerate peft sentencepiece evaluate
pip install torch --index-url https://download.pytorch.org/whl/cu121  # or cpu
pip install dkpro-cassis lxml beautifulsoup4
pip install openpyxl faiss-cpu
```

2) Run a zero-shot prompt locally (example)
```bash
python Zero-Shot-Prompting/NER_Prompting_generic_chat.py
```

3) Evaluate predictions
```bash
python Evaluation_Files/calculate_metrics_multiple.py
```


## Repository Map (high-level)
- `admin-*/` INCEpTION export (CAS/XMI, logs, guidelines)
- `All_Output_Folders/` Aggregated experiment outputs by date/shot
- `ConceptsList/` Gazetteers and concept JSONs used in rules/prompts
- `curation/`, `datasets_annotations/` Additional annotation exports
- `Dataset-25-July-Document-Level/`, `FinalDatasets-*`, `SentenceDatasets*`, `NewDatasets27August/` Prepared datasets (sentence/document level; BIO tokens/labels)
- `Evaluation_Files/` Evaluation and conversion utilities
- `Evaluation_Results/`, `Results/` Metrics, tables, final artifacts
- `FewShotPrompting/`, `Zero-Shot-Prompting/` Prompting scripts and SLURM jobs
- `Fine-tune/` Classic transformers fine-tuning (DeBERTa/BERT/etc)
- `Fine-tune-LLM/`, `Fine-tune-LLM-*` LLM supervised fine-tuning (SFT)
- `LLM-Predictions-*` LLM predictions + extracted JSON for eval
- `OpenAgrar_BIO_*`, `Test_BIO_*` BIO labels/tokens per split/corpus
- `Rule-based-annotations/` Rule-based annotators to auto-tag text
- `Text_Files_*` Text extracted from CAS/XMI for prompting
- `WordEmbeddings/` Embedding pipelines and FAISS index
- `XMI_Files*` Canonical CAS/XMI exports used to generate datasets


## Data Preparation

### From INCEpTION CAS/XMI → Text
- Source CAS/XMI: `XMI_Files/`, `XMI_Files_OpenAgrar/`, `XMI_Files_Bonares/` and in `admin-*/source/`
- Convert to plain text per document for prompting in:
  - `Text_Files_OpenAgrar/`
  - `Text_Files_For_LLM_Input/`
  - Utilities: see `Evaluation_Files/extract_text_from_cas.py`

### BIO Generation
- OpenAgrar: `generate_bio_openagrar.py` (root) and helpers in `Evaluation_Files/`
- Bonares: `generate_bio_bonares.py` (root)
- CAS-based: `Evaluation_Files/generate_bio_from_cas.py` and variants
- Outputs land under e.g. `OpenAgrar_BIO_tokens/`, `OpenAgrar_BIO_labels/`, and indexed counterparts

### Ready-to-train Datasets
- Sentence-level: `FinalDatasets-10July/`, `SentenceDatasets-16August/`, `NewDatasets27August/`
- Document-level: `Dataset-25-July-Document-Level/`, `Fine-tune-LLM-Document/`
- Common formats:
  - Chat format JSONL: `*_chat*.jsonl`
  - Input/Output JSONL: `*_input_output.jsonl`
  - Text/Entity JSON: `*_text_entity.json`


## Prompting (Zero/Few Shot)

### Zero-shot
- Scripts in `Zero-Shot-Prompting/` (generic and per-model variants)
- Run locally:
```bash
python Zero-Shot-Prompting/NER_Prompting_generic_chat.py
```

### Few-shot
- Scripts in `FewShotPrompting/` (1–5 shot variants present)
- Example local run:
```bash
python FewShotPrompting/NER_Prompting_generic_chat_Oneshot.py
```

### SLURM (HPC) Jobs for Prompting
- Submit prepared jobs:
```bash
sbatch FewShotPrompting/job-generic-usage-complete.sbatch
```
- Batched runners: `FewShotPrompting/run_all_jobs*.sbatch`
- Logs and captured outputs are written under `FinalOutput-*/` and `Evaluation_Results/`


## Fine-tuning

### Classic Transformers (token classification)
- Directory: `Fine-tune/`
- Main entry points: `fine_tune_mdeberta.py`, `fine_tune_bert_base_cased.py`, etc.
- Example (local GPU/CPU):
```bash
python Fine-tune/fine_tune_mdeberta.py \
  --train_file FinalDatasets-10July/train.jsonl \
  --valid_file FinalDatasets-10July/valid.jsonl \
  --model microsoft/deberta-v3-base
```
- SLURM:
```bash
sbatch Fine-tune/finetune-generic.sbatch
```

### LLM SFT
- Sentence-level: `Fine-tune-LLM-Sentence/`
- Document-level: `Fine-tune-LLM-Document/`
- Generic: `Fine-tune-LLM/` with `finetune-generic-llm.sbatch`
- Data helpers: `generate_dataset_for_llm.py`, `convert_data_autotrain.py`, `final_dataset.py`


## Embeddings Workflow
- Directory: `WordEmbeddings/`
- Build embeddings and FAISS index:
```bash
python WordEmbeddings/ner_embeddings_pipeline.py
# or DeepSeek variant
python WordEmbeddings/ner_embeddings_pipeline-deepseek.py
```
- SLURM batch: `WordEmbeddings/job-embeddings*.sbatch`
- Index: `WordEmbeddings/ner_embeddings.index`


## Evaluation
- Directory: `Evaluation_Files/`
- Strict/partial metrics:
  - `calculate_metrics_multiple.py`, `calculate_metrics_multiple_partial.py`
- Embeddings-assisted evaluation:
  - `calculate_metrics_multiple_embeddings.py`, `calculate_metrics_multiple_embeddings_excel.py`
- Fine-tuned models:
  - `calculate_metrics_finetune_llm.py` (sentence), `calculate_metrics_finetune_llm_document.py` (document)
- Utilities:
  - BIO generation from predictions: `generate_bio_from_json.py`, `generate_bio_from_json_finetune.py`
  - JSON extraction from model outputs: `extract_json_block*.py`
- Example:
```bash
python Evaluation_Files/extract_json_block_predictions.py \
  --input Results/Final_TestFiles_29thAugust_Prompting/preds.jsonl \
  --output Evaluation_Results/TestFiles_29July/preds_extracted.json

python Evaluation_Files/calculate_metrics_multiple.py \
  --gold Evaluation_Results/TestFiles_29July/gold_extracted.json \
  --pred Evaluation_Results/TestFiles_29July/preds_extracted.json
```
- Results are stored under `Evaluation_Results/` and `Results/` (and dated folders in `All_Output_Folders/`).


## HPC Usage (SLURM)
- Generic templates:
  - Prompting: `FewShotPrompting/job-generic-usage-complete*.sbatch`
  - Fine-tuning (transformers): `Fine-tune/finetune-generic.sbatch`
  - LLM SFT: `Fine-tune-LLM/finetune-generic-llm.sbatch`
  - Embeddings: `WordEmbeddings/job-embeddings*.sbatch`
- Submit a job:
```bash
sbatch <path-to-sbatch-file>
```
- Monitor outputs: `gpu_job_*.out`, `slurm-*.out`, and per-job `job_monitor_logs_*` directories


## Rule-based Annotations
- Directory: `Rule-based-annotations/`
- Main scripts: `annotate_text_files_inception.py` and `_updated.py`
- Extend rules in `extra_rules.py`


## Key Datasets and Splits
- `FinalDatasets-10July/`, `FinalDatasets-21July/`, `NewDatasets27August/`
  - train/valid/test in multiple formats
- `Dataset-25-July-Document-Level/` document-level corpora and BIO
- `OpenAgrar_BIO_*`, `Test_BIO_*` prepared BIO tokens/labels/indexed


## Reproduce Main Experiments
1) Prepare sentence-level train/valid/test (already included under `FinalDatasets-*`)
2) Choose a strategy:
   - Zero-/Few-shot prompting: run in `Zero-Shot-Prompting/` or `FewShotPrompting/`
   - Fine-tune transformer: `Fine-tune/fine_tune_mdeberta.py` via SLURM
   - LLM SFT: `Fine-tune-LLM*/finetune-generic-llm.sbatch`
3) Extract predictions to JSON, then evaluate with `Evaluation_Files/`
4) Compare numbers in `Evaluation_Results/` and `Results/`


## Notes
- Type systems: see `Evaluation_Files/TypeSystem.xml` and `full-typesystem.xml`
- Guidelines: `admin-*/guideline/250507_Annotation_Guidelines_2.0.pdf`
- Many folders are date- and shot-specific to preserve experiment lineage


