# Predictive Modeling of Diagnoses and Treatments in EHR based on Multimodal Context

## Setup
1. Get access to MIMIC-III
2. Download TP-BERTa checkpoint pretrained on a single task type (binary) from the [official repository](https://github.com/jyansir/tp-berta)
3. Change paths accordingly for constants `MIMIC_DIR`, `PROJECT_DIR`, `TABULAR_CHECKPOINT_DIR` in `main.py`.

## Run best model
```
python main.py \
    --num_chunks 16 \
    --run_name MIHST_evaluate \
    --max_epochs 20 \
    --num_heads_labattn 1 \
    --patience_threshold 3 \
    --debug False \
    --evaluate_temporal True \
    --use_multihead_attention True \
    --weight_aux 0 \
    --num_layers 1 \
    --num_attention_heads 1 \
    --setup random \
    --limit_ds 0 \
    --is_baseline False \
    --aux_task "none" \
    --use_all_tokens False \
    --apply_transformation False \
    --apply_weight False \
    --reduce_computation True \
    --apply_temporal_loss True \
    --save_model True \
    --use_tabular True \
    --subset_tabular False \
    --freeze_tabular False \
    --k_list 256 \
    --bin_strategy frequency \
    --pool_features temporal-max \
    --filter_features less \
    --late_fuse none \
    --weight_temporal 0.1 0.1 0.1 0.1 0.6
```
