from data.custom_dataset import CustomDataset
import torch
import transformers


def get_dataset(notes_agg_df, labs_agg_df, split, tokenizer, tabular_tokenizer, max_chunks, setup, limit_ds=0, use_tabular=False, textualize=False, k_list=[4]):
    return CustomDataset(
        notes_agg_df[notes_agg_df.SPLIT == split],
        labs_agg_df[labs_agg_df.SPLIT == split],
        tokenizer=tokenizer,
        tabular_tokenizer=tabular_tokenizer,
        max_chunks=max_chunks,
        setup=setup,
        limit_ds=limit_ds,
        use_tabular=use_tabular,
        textualize=textualize,
        k_list=k_list
    )


def get_dataloader(dataset):
    dataloader_params = {
        "batch_size": 1,
        "shuffle": True,
        "num_workers": 6,
        "pin_memory": True,
    }
    return torch.utils.data.DataLoader(dataset, **dataloader_params)


def get_tokenizer(checkpoint):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        checkpoint, model_max_length=512
    )
    return tokenizer


def get_tabular_tokenizer(checkpoint):
    tokenizer = transformers.RobertaTokenizer.from_pretrained(checkpoint)
    
    return tokenizer


def get_cutoffs(hours_elapsed, category_ids):
    cutoffs = {"2d": [-1], "5d": [-1], "13d": [-1], "noDS": [-1], "all": [-1]}
    for i, (hour, cat) in enumerate(zip(hours_elapsed, category_ids)):
        if cat != 5:
            if hour < 2 * 24:
                cutoffs["2d"] = [i]
            if hour < 5 * 24:
                cutoffs["5d"] = [i]
            if hour < 13 * 24:
                cutoffs["13d"] = [i]
            cutoffs["noDS"] = [i]
        # cutoffs['all'] = i
    return cutoffs
