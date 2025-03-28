import numpy as np
import torch
import tqdm as tqdm
import itertools
from torch.utils.data import Dataset


FT_MAX_TOK = 5

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
        textualize=False,
        k_list=[4],
        bin_strategy=["frequency"]
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
        self.textualize = textualize
        self.k_list = k_list
        self.bin_strategy = bin_strategy
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
    
    def tabular_tokenize(self, tab, max_length=None):
        return self.tabular_tokenizer.encode(
            tab,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            padding="max_length",
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
                'percent_elapsed': [],
                "flag_ids": []}
        
        data = data.squeeze(axis=0) # convert to pd series
        
        encoded_feature_names = [self.tabular_tokenize(l, FT_MAX_TOK) for l in data.LABEL]

        N = len(encoded_feature_names)
        K = len(self.bin_strategy)*len(self.k_list)

        # prepare encoded pieces
        cls_token_id = self.tabular_tokenizer.cls_token_id

        num_fix_part, num_token_types, num_position_ids, \
            num_feature_cls_mask, num_input_scales = [], [], [], [], []
        
        name_to_prefix = {"frequency": "FBIN", "width": "WBIN"}

        for i, efn in enumerate(encoded_feature_names):
            for k in self.k_list:
                for strat in self.bin_strategy:
                    bin_name = f'{name_to_prefix[strat]}_{k}'
                    if K > 1:
                        # add bin name if there's more than one bin strategy
                        name_id = self.tabular_tokenize(bin_name, 5)
                        efn += name_id
                        
                    num_fix_part.extend([cls_token_id] + efn + [data[bin_name][i]])
                    num_token_types.extend([0] + [0] * len(efn) + [1]) # continous type (1)
                    num_feature_cls_mask.extend([1] + [0] * len(efn) + [0])
                    num_position_ids.extend([0] + [i for i in range(1, len(efn)+1)] + [0])
                    if use_num_multiply:
                        num_input_scales.extend([data.NORM_VAL[i]] * (1+ len(efn))) 
        num_fix_part = np.array(num_fix_part).reshape(N*K,-1)
        num_token_types = np.array(num_token_types).reshape(N*K,-1)
        num_position_ids = np.array(num_position_ids).reshape(N*K,-1)
        num_feature_cls_mask = np.array(num_feature_cls_mask).reshape(N*K,-1)
        if not use_num_multiply:
            num_input_scales = np.ones_like(num_token_types, dtype=np.float32)
            num_input_scales[num_token_types == 1] = np.repeat(np.array(data.NORM_VAL), K)
            assert num_input_scales.shape[0] == (N*K)
        else:
            num_input_scales = np.array(num_input_scales).reshape(N*K,-1)        

        # Get hours elapsed
        hours_elapsed = np.array(
            list(
                itertools.chain.from_iterable(
                    [
                        [data.HOURS_ELAPSED[i]] * K 
                        for i in range(N) # one per feature-bin strategy-k row
                    ]
                )
            )
        )

        percent_elapsed = np.array(
            list(
                itertools.chain.from_iterable(
                    [
                        [data.PERCENT_ELAPSED[i]] * K
                        for i in range(N) # one per feature-bin strategy row
                    ]
                )
            )
        ) 

        flag_ids = np.array(
            list(
                itertools.chain.from_iterable(
                    [
                        [data.FLAG_INDEX[i]] * K
                        for i in range(N) # one per feature-bin strategy row
                    ]
                )
            )
        ) 
        
        return {'input_ids': num_fix_part,
                'input_scales': num_input_scales,
                'features_cls_mask': num_feature_cls_mask,
                'token_type_ids': num_token_types,
                'position_ids': num_position_ids,
                'hours_elapsed': hours_elapsed,
                'percent_elapsed': percent_elapsed,
                'flag_ids': flag_ids}
    
    def __getitem__(self, idx):
        np.random.seed(1)
        encoded = {}
        data = self.notes_agg_df.iloc[idx]
        hadm_id = data.HADM_ID
        all_text = data.TEXT
        all_times = data.PERCENT_ELAPSED
        all_hours = data.HOURS_ELAPSED
        all_category = data.CATEGORY_INDEX

        if self.use_tabular and self.textualize:
            lab_data = self.labs_agg_df[self.labs_agg_df.HADM_ID == hadm_id]
            lab_data = lab_data.squeeze(axis=0)
            if not lab_data.empty:
                merged_text = data.TEXT + lab_data.TEXT # merge lists
                merged_times = data.PERCENT_ELAPSED + lab_data.PERCENT_ELAPSED  
                merged_hours = data.HOURS_ELAPSED + lab_data.HOURS_ELAPSED
                merged_category = data.CATEGORY_INDEX + [15] * len(lab_data.TEXT)
                all_times = [x for x in sorted(merged_times)]
                all_text = [x for _, x in sorted(zip(merged_times, merged_text))] # sort by percent elapsed
                all_hours = [x for _, x in sorted(zip(merged_times, merged_hours))]
                all_category = [x for _, x in sorted(zip(merged_times, merged_category))]
            # calculate output by tokenizing each item in merged list
        
        output = [self.tokenize(doc) for doc in all_text]
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
                        [all_category[i]] * len(output[i]["input_ids"])
                        for i in range(len(output))
                    ]
                )
            )
        )
        hours_elapsed = np.array(
            list(
                itertools.chain.from_iterable(
                    [
                        [all_hours[i]] * len(output[i]["input_ids"])
                        for i in range(len(output))
                    ]
                )
            )
        )

        percent_elapsed = np.array(
            list(
                itertools.chain.from_iterable(
                    [
                        [all_times[i]] * len(output[i]["input_ids"])
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

        if self.use_tabular and not self.textualize:
            lab_data = self.encode_tabular(self.labs_agg_df[self.labs_agg_df.HADM_ID == hadm_id])
            encoded["tabular"] = {
                "input_ids": lab_data['input_ids'],
                "input_scales": lab_data['input_scales'],
                "features_cls_mask": lab_data['features_cls_mask'],
                "token_type_ids": lab_data['token_type_ids'],
                "position_ids": lab_data['position_ids'],
                "hours_elapsed": lab_data['hours_elapsed'],
                "percent_elapsed": lab_data['percent_elapsed'],
                "flag_ids": lab_data['flag_ids']
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
