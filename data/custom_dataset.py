import pandas as pd
import numpy as np
import torch
import tqdm as tqdm
from tqdm import tqdm
import ast
import os
import itertools
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    """Custom dataset for real-time ICD-9 code prediction.

    The complete EHR sequence is very long, therefore the custom dataset
    allows for multiple configurations to select the chunks to use.

    In particular, the following set-ups are available:
    - latest: only the last max_chunks chunks are used
    - uniform: the first and last note are always used, and the remaining
    chunks are randomly sampled (this is the "random" set-up in the paper)
    - random: all notes are used (this is the "complete EHR set-up in the paper)
    - limit_ds: limit DS to 4 chunks (only for the "random" setup, corresponding to
        the Limited DS set-up in the paper)

    The elements of the resulting dataset include:
    - input_ids: tokenized input
    - attention_mask: attention mask
    - seq_ids: sequence ids
    - category_ids: category ids
    - label: ICD-9 code
    - hadm_id: HADM_ID
    - hours_elapsed: hours elapsed since admission
    - cutoffs: cutoffs for the different time windows (2d, 5d, 13d, noDS, all
    - percent_elapsed: percentage of time elapsed since admission (only for the latest setup
    """

    def __init__(
        self,
        notes_agg_df,
        labs_agg_df,
        tokenizer,
        tabular_tokenizer,
        max_chunks,
        setup="latest",  # 'uniform
        limit_ds=0,
        batch_size=None,
        use_tabular=False,
    ):
        self.notes_agg_df = notes_agg_df
        self.labs_agg_df = labs_agg_df
        self.tokenizer = tokenizer
        self.tabular_tokenizer = tabular_tokenizer
        self.max_chunks = max_chunks
        self.batch_size = batch_size
        self.setup = setup
        self.limit_ds = limit_ds
        self.use_tabular = use_tabular
        np.random.seed(1)

    def __len__(self):
        return len(self.notes_agg_df)

    def tokenize(self, doc):
        return self.tokenizer(
            doc,
            truncation=True,
            return_overflowing_tokens=True,
            padding="max_length",
            return_tensors="pt",
        )

    def _get_note_end_chunk_ids(self, seq_ids):
        id = seq_ids[0]
        change_points = []
        for i, seq_id in enumerate(seq_ids):
            if seq_id != id:
                change_points.append(i - 1)
                id = seq_id
        # append last index, as it is always the indication of the last note
        change_points.append(i)
        return change_points

    def _get_cutoffs(self, hours_elapsed, category_ids):
        cutoffs = {"2d": -1, "5d": -1, "13d": -1, "noDS": -1, "all": -1}
        for i, (hour, cat) in enumerate(zip(hours_elapsed, category_ids)):
            if cat != 5:
                if hour < 2 * 24:
                    cutoffs["2d"] = i
                if hour < 5 * 24:
                    cutoffs["5d"] = i
                if hour < 13 * 24:
                    cutoffs["13d"] = i
                cutoffs["noDS"] = i
            # cutoffs['all'] = i
        return cutoffs

    def filter_mask(self, seq_ids):
        # TODO: this is producing a different mask every time, we should fix seed everytime
        """Get selected indices according to the logic:
        1. All indices of the first note
        2. All indices of the last note (a.k.a. discharge summary))
        3. Randomly select the remaining indices from the middle notes"""
        np.random.seed(1)
        first_indices = np.where(seq_ids == seq_ids[0])[0][: self.max_chunks]
        # limit DS to 4 chunks
        last_indices = np.where(seq_ids == seq_ids[-1])[0]
        # limit last indices if more than max_chunks - len(first_indices)
        # selecting the last max_chunks - len(first_indices) indices
        num_last_indices = min(len(last_indices), self.max_chunks - len(first_indices))
        last_indices = last_indices[len(last_indices) - num_last_indices :]
        # if limit ds, then only keep the last limit_ds indices
        last_indices = last_indices[-self.limit_ds :]
        middle_indices = np.where(
            np.logical_and(seq_ids > seq_ids[0], seq_ids < seq_ids[-1])
        )[0]
        middle_indices = np.sort(
            np.random.choice(
                middle_indices,
                max(
                    0,
                    min(
                        len(middle_indices),
                        self.max_chunks - len(first_indices) - len(last_indices),
                    ),
                ),
                replace=False,
            )
        )
        return first_indices.tolist() + middle_indices.tolist() + last_indices.tolist()

    def encode_tabular(self, data, use_num_multiply=False):
        """
        Adapted from TP-BERTa. Encode tabular data (labs) for an HADM_ID's data.
        """
        if data.empty:
            return {'input_ids': [],
                'input_scales': [],
                'features_cls_mask': [],
                'token_type_ids': [],
                'position_ids': [],
                'hours_elapsed': [],
                'percent_elapsed': []}
        
        data = data.squeeze(axis=0) # convert to pd series
        
        ft_max_tok = 5
        encoded_feature_names = [self.tabular_tokenizer.encode(l) for l in data.LABEL]

        N = len(encoded_feature_names)

        # prepare encoded pieces
        cls_token_id = self.tabular_tokenizer.cls_token_id
        sep_token_id = self.tabular_tokenizer.sep_token_id
        pad_token_id = self.tabular_tokenizer.pad_token_id
        mask_token_id = self.tabular_tokenizer.mask_token_id

        num_fix_part, num_token_types, num_position_ids, \
            num_feature_cls_mask, num_input_scales = [], [], [], [], []
        
        num_num_token = 0

        for i, efn in enumerate(encoded_feature_names):
            if len(efn) >= ft_max_tok:
                efn = efn[:ft_max_tok]
            else:
                efn += ([pad_token_id] * (ft_max_tok - len(efn)))
            num_fix_part.extend([cls_token_id] + efn + [data.BIN[i]])
            num_token_types.extend([0] + [0] * len(efn) + [1]) # continous type (1)
            num_feature_cls_mask.extend([1] + [0] * len(efn) + [0])
            num_position_ids.extend([0] + [i for i in range(1, len(efn)+1)] + [0])
            if use_num_multiply:
                num_input_scales.extend([data.NORM_VAL[i]] * (1+ len(efn))) #TODO: why noot 2+?
                # num_input_scales.append(data['NORM_VAL'][i].repeat(1 + len(efn), axis=1)) # scale for feature name toks and bin tok
        num_fix_part = np.array(num_fix_part).reshape(N,-1)
        # num_fix_part[num_fix_part == mask_token_id] = data['BIN'].reshape(-1)
        num_token_types = np.array(num_token_types).reshape(N,-1)
        num_num_token = num_token_types.shape[1]
        num_position_ids = np.array(num_position_ids).reshape(N,-1)
        num_feature_cls_mask = np.array(num_feature_cls_mask).reshape(N,-1)
        if not use_num_multiply:
            num_input_scales = np.ones_like(num_token_types, dtype=np.float32)
            num_input_scales[num_token_types == 1] = data.NORM_VAL
            # num_input_scales = num_input_scales.flatten()
            # num_token_types = num_token_types.flatten()
        else:
            num_input_scales = np.array(num_input_scales).reshape(N,-1)
            # num_input_scales = np.concatenate(num_input_scales, axis=1)
        

        # Get hours elapsed
        hours_elapsed = np.array(
            list(
                itertools.chain.from_iterable(
                    [
                        [data.HOURS_ELAPSED[i]] for i in range(N) # one per feature row
                    ]
                )
            )
        )

        percent_elapsed = np.array(data.PERCENT_ELAPSED)
        
        return {'input_ids': num_fix_part,
                'input_scales': num_input_scales,
                'features_cls_mask': num_feature_cls_mask,
                'token_type_ids': num_token_types,
                'position_ids': num_position_ids,
                'hours_elapsed': hours_elapsed,
                'percent_elapsed': percent_elapsed}
    
    def __getitem__(self, idx):
        np.random.seed(1)
        encoded = {}
        data = self.notes_agg_df.iloc[idx]
        
        output = [self.tokenize(doc) for doc in data.TEXT]
        hadm_id = data.HADM_ID
        # doc[input_ids] is (# chunks, 512), i.e., if note is longer than 512, it returns len/512 # chunks
        input_ids = torch.cat(
            [doc["input_ids"] for doc in output]
        )  # this concatenates to (overall # chunks, 512)
        attention_mask = torch.cat([doc["attention_mask"] for doc in output])
        seq_ids = np.array(
            list(
                itertools.chain.from_iterable(
                    [[i] * len(output[i]["input_ids"]) for i in range(len(output))]
                )
            )
        )
        category_ids = np.array(
            list(
                itertools.chain.from_iterable(
                    [
                        [data.CATEGORY_INDEX[i]] * len(output[i]["input_ids"])
                        for i in range(len(output))
                    ]
                )
            )
        )
        hours_elapsed = np.array(
            list(
                itertools.chain.from_iterable(
                    [
                        [data.HOURS_ELAPSED[i]] * len(output[i]["input_ids"])
                        for i in range(len(output))
                    ]
                )
            )
        )

        percent_elapsed = np.array(
            list(
                itertools.chain.from_iterable(
                    [
                        [data.PERCENT_ELAPSED[i]] * len(output[i]["input_ids"])
                        for i in range(len(output))
                    ]
                )
            )
        )

        label = torch.FloatTensor(data.ICD9_CODE_BINARY)

        input_ids = input_ids[:]
        attention_mask = attention_mask[:]
        seq_ids = torch.LongTensor(seq_ids)
        category_ids = torch.LongTensor(category_ids)
        seq_ids = seq_ids[:]
        category_ids = category_ids[:]

        # if latest setup, select last max_chunks
        if self.setup == "latest":
            input_ids = input_ids[-self.max_chunks :]
            attention_mask = attention_mask[-self.max_chunks :]
            seq_ids = torch.LongTensor(seq_ids)
            category_ids = torch.LongTensor(category_ids)
            seq_ids = seq_ids[-self.max_chunks :]
            category_ids = category_ids[-self.max_chunks :]
            hours_elapsed = hours_elapsed[-self.max_chunks :]
            percent_elapsed = percent_elapsed[-self.max_chunks :]

        # in a uniform setting, select first and last note
        # and randomly sample the rest
        elif self.setup == "uniform":
            indices_mask = self.filter_mask(np.array(seq_ids))
            print(indices_mask)
            input_ids = input_ids[indices_mask]
            print(input_ids)
            attention_mask = attention_mask[indices_mask]
            seq_ids = seq_ids[indices_mask]
            category_ids = category_ids[indices_mask]
            hours_elapsed = hours_elapsed[indices_mask]
            percent_elapsed = percent_elapsed[indices_mask]

        elif self.setup == "random":
            # keep all notes and random sample during training
            # while keeping all notes for inference
            # crop at 95th percentile of chunks, i.e., 181.0
            # to avoid some outliers that cause OOM
            input_ids = input_ids[-181:]
            attention_mask = attention_mask[-181:]
            seq_ids = seq_ids[-181:]
            category_ids = category_ids[-181:]
            hours_elapsed = hours_elapsed[-181:]
            percent_elapsed = percent_elapsed[-181:]

        else:
            raise ValueError("Invalid setup")

        # recalculate seq ids based on filtered indices
        seq_id_vals = torch.unique(seq_ids).tolist()
        seq_id_dict = {seq: idx for idx, seq in enumerate(seq_id_vals)}
        seq_ids = seq_ids.apply_(seq_id_dict.get)
        cutoffs = self._get_cutoffs(hours_elapsed, category_ids)

        if self.use_tabular:
            lab_data = self.encode_tabular(self.labs_agg_df[self.labs_agg_df.HADM_ID == hadm_id])
            if self.setup == "latest":
                encoded["tabular"] = {
                    "input_ids": lab_data['input_ids'][-self.max_chunks :],
                    "input_scales": lab_data['input_scales'][-self.max_chunks :],
                    "features_cls_mask": lab_data['features_cls_mask'][-self.max_chunks :],
                    "token_type_ids": lab_data['token_type_ids'][-self.max_chunks :],
                    "position_ids": lab_data['position_ids'][-self.max_chunks :],
                    "hours_elapsed": lab_data['hours_elapsed'][-self.max_chunks :],
                    "percent_elapsed": lab_data['percent_elapsed'][-self.max_chunks :],
                }
            elif self.setup == "uniform":
                raise NotImplementedError
                indices_mask = self.filter_mask(np.array(seq_ids))
                print(indices_mask)
                input_ids = input_ids[indices_mask]
                print(input_ids)
                encoded["tabular"] = {
                    "input_ids": lab_data['input_ids'][indices_mask],
                    "input_scales": lab_data['input_scales'][indices_mask],
                    "features_cls_mask": lab_data['features_cls_mask'][indices_mask],
                    "token_type_ids": lab_data['token_type_ids'][indices_mask],
                    "position_ids": lab_data['position_ids'][indices_mask],
                    "hours_elapsed": lab_data['hours_elapsed'][indices_mask],
                    "percent_elapsed": lab_data['percent_elapsed'][indices_mask],
                }

            elif self.setup == "random":
                encoded["tabular"] = {
                    "input_ids": lab_data['input_ids'],
                    "input_scales": lab_data['input_scales'],
                    "features_cls_mask": lab_data['features_cls_mask'],
                    "token_type_ids": lab_data['token_type_ids'],
                    "position_ids": lab_data['position_ids'],
                    "hours_elapsed": lab_data['hours_elapsed'],
                    "percent_elapsed": lab_data['percent_elapsed'],
                }

        encoded["notes"] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "seq_ids": seq_ids,
            "category_ids": category_ids,
            "label": label,
            "hadm_id": hadm_id,
            "hours_elapsed": hours_elapsed,
            "percent_elapsed": percent_elapsed,
            "cutoffs": cutoffs
            }
        
        return encoded
