import json
from data.preprocess import DataProcessor
import os
from data.custom_dataset import CustomDataset
from data.utils import get_dataset, get_tokenizer, get_tabular_tokenizer, get_dataloader
from model.model import Model
import torch
import pandas as pd
import numpy as np
import torch.optim as optim
import ast
import os
import itertools
import torch.utils.checkpoint
from torch.cuda.amp import GradScaler, autocast
from training.trainer import Trainer
import argparse


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


if __name__ == "__main__":
    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy("file_system")
    parser = argparse.ArgumentParser(
        description="Train model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # TODO: update the help definitions
    parser.add_argument("-n", "--num_chunks", type=int, default=4, help="number of chunks")
    parser.add_argument("-r", "--run_name", type=str, default="test", help="run name")
    parser.add_argument("-m", "--max_epochs", type=int, default=20, help="number of max epochs")
    parser.add_argument("-l", "--num_heads_labattn", type=int, default=1, help="number of heads for lab attention")
    parser.add_argument("-p", "--patience_threshold", type=int, default=3, help="patience threshold")
    parser.add_argument("-d", "--debug", type=boolean_string, default='False', help="whether to run model in debug mode")
    parser.add_argument("-e", "--evaluate_temporal", type=boolean_string, default='True', help="whether to evaluate temporal")
    parser.add_argument("-u", "--use_multihead_attention", type=boolean_string, default='True', help=" ")
    parser.add_argument("-w", "--weight_aux", type=float, default=0, help="  ")
    parser.add_argument("-z", "--num_layers", type=int, default=0, help="  ")
    parser.add_argument("-x", "--num_attention_heads", type=int, default=1, help="  ")
    parser.add_argument("-s", "--setup", type=str, default="latest", help="  ")
    parser.add_argument("-i", "--limit_ds", type=int, default=0, help="  ")
    parser.add_argument("-b", "--is_baseline", type=boolean_string, default=False, help="  ")
    parser.add_argument("-a", "--aux_task", type=str, default="next_document_embedding", help="  ")
    parser.add_argument("-t", "--apply_transformation", type=boolean_string, default=False, help="  ")
    parser.add_argument("-k", "--use_all_tokens", type=boolean_string, default=False, help="  ")
    parser.add_argument("-c", "--apply_weight", type=boolean_string, default=False, help="  ")
    parser.add_argument("-f", "--reduce_computation", type=boolean_string, default=False, help="  ")
    parser.add_argument("-y", "--apply_temporal_loss", type=boolean_string, default=False, help="  ")
    parser.add_argument("-g", "--save_model", type=boolean_string, default=False, help="  ")
    parser.add_argument("-ch", "--load_from_checkpoint", type=boolean_string, default=False, help="  ")
    parser.add_argument("-o", "--lr", type=float, default=5e-5, help="  ")
    parser.add_argument("-j", "--random_sample", type=boolean_string, default=True, help="  ")
    parser.add_argument("-mm", "--use_tabular", type=boolean_string, default=False, help="  ")
    parser.add_argument("-tt", "--textualize", type=boolean_string, default=False, help="  ")
    parser.add_argument("-st", "--subset_tabular", type=boolean_string, default=False, help="  ")
    parser.add_argument("-nt", "--num_tabular_features", type=int, default=10, help="  ")
    parser.add_argument("-ft", "--freeze_tabular", type=boolean_string, default=True, help="  ")
    parser.add_argument('-kl','--k_list', nargs='+', type=int, default=[4], help="List of bin numbers to try")
    parser.add_argument('-bs','--bin_strategy', nargs='+', type=str, default=["frequency"], help="List of bin strategies to try. options: frequency, width")
    parser.add_argument('-pf','--pool_features', type=str, default="none", help="options: max, sum, temporal-max, temporal-avg, none")
    parser.add_argument('-lf','--late_fuse', type=str, default="none", help="options: embeddings, predictions, none")
    parser.add_argument('-ta','--use_tabular_attn', type=boolean_string, default=False, help="  ")
    parser.add_argument('-me','--use_modality_embeddings', type=boolean_string, default=False, help="  ")
    parser.add_argument('-fe','--use_flag_embeddings', type=boolean_string, default=False, help="  ")
    parser.add_argument('-fa','--filter_features', type=str, default="basic", help="  ")
    parser.add_argument('-lp','--load_pretrained', type=boolean_string, default=False, help="  ")
    parser.add_argument('-oa','--use_one_alpha', type=boolean_string, default=False, help="  ")


    args = parser.parse_args()
    args_config = vars(args)

    # device
    USE_GPU = True
    dtype = torch.float32
    if USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    cpu = torch.device("cpu")
    print(device)

    ### for debugging use cpu
    # device = cpu

    config = {
        #    "run_name": "Run_test_TLWAN"
        "run_name": args_config["run_name"],
        "project_path": "/vol/bitbucket/ch2223/temporal-multimodal-learning",
        # pminervini/RoBERTa-base-PM-M3-Voc-hf
        # "base_checkpoint": os.path.join("", "RoBERTa-base-PM-M3-Voc-hf"),
        "base_checkpoint": "pminervini/RoBERTa-base-PM-M3-Voc-hf",
        "num_attention_heads": args_config["num_attention_heads"],
        "num_layers": args_config["num_layers"],
        "lr": args_config["lr"],
        "max_chunks": args_config["num_chunks"],
        "grad_accumulation": args_config["num_chunks"],
        "use_positional_embeddings": True,
        "use_reverse_positional_embeddings": True,
        "priority_mode": "Last",
        "priority_idxs": [1],
        "use_document_embeddings": False,
        "use_reverse_document_embeddings": False,
        "use_category_embeddings": True,
        "use_modality_embeddings": args_config["use_modality_embeddings"],
        "use_flag_embeddings": args_config["use_flag_embeddings"],
        "num_labels": 50,
        "use_all_tokens": args_config["use_all_tokens"],
        "num_heads_labattn": args_config["num_heads_labattn"],
        "final_aggregation": "cls",
        "only_discharge_summary": False,
        "patience_threshold": args_config["patience_threshold"],
        "max_epochs": args_config["max_epochs"],
        "save_model": args_config["save_model"],
        "load_from_checkpoint": args_config["load_from_checkpoint"],
        # "checkpoint_name": "Run_all_notes_last_second_transf",
        "checkpoint_name": args_config["run_name"],
        "evaluate_temporal": args_config["evaluate_temporal"],
        "use_multihead_attention": args_config["use_multihead_attention"],
        "debug": args_config["debug"],
        "weight_aux": args_config["weight_aux"],
        "setup": args_config["setup"],
        "limit_ds": args_config["limit_ds"],
        "is_baseline": args_config["is_baseline"],
        "aux_task": args_config["aux_task"],
        "apply_transformation": args_config["apply_transformation"],
        "apply_weight": args_config["apply_weight"],
        "reduce_computation": args_config["reduce_computation"],
        "apply_temporal_loss": args_config["apply_temporal_loss"],
        "random_sample": args_config["random_sample"],
        "use_tabular": args_config["use_tabular"],
        "textualize": args_config["textualize"],
        "subset_tabular": args_config["subset_tabular"],
        "tabular_base_checkpoint": "model/tpberta-single/tp-bin",
        "max_tabular_features": args_config["num_tabular_features"],
        "freeze_tabular": args_config["freeze_tabular"],
        "k_list": args_config["k_list"],
        "bin_strategy": args_config["bin_strategy"],
        "pool_features": args_config["pool_features"],
        "late_fuse": args_config["late_fuse"],
        "use_tabular_attn": args_config["use_tabular_attn"],
        "filter_features": args_config["filter_features"],
        "load_pretrained": args_config["load_pretrained"],
        "use_one_alpha": args_config["use_one_alpha"]
    }

    print("Bin param", config['k_list'])
    print("Bin strategies", config['bin_strategy'])
    
    with open(os.path.join("", f"results/config_{config['run_name']}.json"), "w") as f:
        json.dump(config, f)

    # get tokenizer
    tokenizer = get_tokenizer(config["base_checkpoint"])
    tabular_tokenizer = get_tabular_tokenizer(config["tabular_base_checkpoint"])

    # process and aggregate raw data
    dp = DataProcessor(dataset_path="/vol/bitbucket/ch2223/temp-mm/data/mimiciii", 
                       config=config, 
                       start_token_id=(tabular_tokenizer.mask_token_id + 1))
    notes_agg_df, categories_mapping, labs_agg_df = dp.aggregate_data()

    # Get training / validation / test
    dataset_config = {
        "max_chunks": config["max_chunks"],
        "setup": config["setup"],
        "limit_ds": config["limit_ds"],
        "use_tabular": config["use_tabular"],
        "textualize": config["textualize"],
        "k_list": config["k_list"],
        "bin_strategy": config["bin_strategy"]
    }
    training_set = get_dataset(
        notes_agg_df, labs_agg_df, "TRAIN", tokenizer=tokenizer, tabular_tokenizer=tabular_tokenizer, **dataset_config
    )
    training_generator = get_dataloader(training_set)

    validation_set = get_dataset(
        notes_agg_df, labs_agg_df, "VALIDATION", tokenizer=tokenizer, tabular_tokenizer=tabular_tokenizer, **dataset_config
    )
    validation_generator = get_dataloader(validation_set)

    test_set = get_dataset(
        notes_agg_df, labs_agg_df, "TEST", tokenizer=tokenizer, tabular_tokenizer=tabular_tokenizer, **dataset_config
    )
    test_generator = get_dataloader(test_set)

    # only to run on CPU
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config["num_categories"] = len(categories_mapping)
    model = Model(config, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0)

    steps_per_epoch = int(
        np.ceil(len(training_generator) / config["grad_accumulation"])
    )

    # steps_per_epoch = 1
    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["lr"],
        three_phase=True,
        total_steps=config["max_epochs"] * steps_per_epoch,
    )

    scaler = GradScaler()

    training_args = {
        "TOTAL_COMPLETED_EPOCHS": 0,
        "CURRENT_BEST": 0,
        "CURRENT_PATIENCE_COUNT": 0,
    }

    # code to load from checkpoint
    if config["load_from_checkpoint"]:
        checkpoint = torch.load(
            os.path.join(
                config["project_path"], f"results/{config['checkpoint_name']}.pth"
            )
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Move optimizer to GPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        training_args["TOTAL_COMPLETED_EPOCHS"] = checkpoint["epochs"]
        training_args["CURRENT_BEST"] = checkpoint["current_best"]
        training_args["CURRENT_PATIENCE_COUNT"] = checkpoint["current_patience_count"]

    else:
        # pd.DataFrame({}).to_csv(
        #     os.path.join(config['project_path'], f"results/{config['run_name']}.csv")
        # )  # Create dummy csv because of GDrive bug
        cutoff_times = ["all", "2d", "5d", "13d", "noDS", "all_aux"]
        for time in cutoff_times:
            pd.DataFrame({}).to_csv(
                os.path.join(
                    config["project_path"], f"results/{config['run_name']}_{time}.csv"
                )
            )  # Create dummy csv because of GDrive bug
        # load from pretrained
        if config["load_pretrained"]: 
            pretrained_checkpoint = torch.load(
            os.path.join(
                config["project_path"], f"results/MMULA_evaluate.pth"
            )
            )
            model.load_state_dict(pretrained_checkpoint["model_state_dict"], strict=False)
    results = {}


    trainer = Trainer(
        model,
        optimizer,
        scaler,
        lr_scheduler,
        config,
        device,
        dtype,
        categories_mapping,
    )
    
    trainer.train(
        training_generator,
        training_args,
        validation_generator,
        grad_accumulation_steps=config["grad_accumulation"],
        epochs=config["max_epochs"],
    )
