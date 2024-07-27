from data.custom_dataset import CustomDataset
import torch
import transformers


def get_dataset(notes_agg_df, labs_agg_df, split, tokenizer, tabular_tokenizer, max_chunks, setup, limit_ds=0):
    return CustomDataset(
        notes_agg_df[notes_agg_df.SPLIT == split],
        labs_agg_df[labs_agg_df.SPLIT == split],
        tokenizer=tokenizer,
        tabular_tokenizer=tabular_tokenizer,
        max_chunks=max_chunks,
        setup=setup,
        limit_ds=limit_ds,
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
