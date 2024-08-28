import pandas as pd
import os
import ast
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer


class DataProcessor:
    """Prepare data for model.

    Some of the functions of this class are extracted from the HTDC (Ng et al, 2023):
    aggregate_hadm_id, add_category_information and multi_hot_encode.

    We added the _add_temporal_information function to add time information,
    which is used for the temporal experiments for real-time ICD-9 coding.
    """

    def __init__(self, dataset_path, config, start_token_id):
        self.notes_df = pd.read_csv(os.path.join(dataset_path, "NOTEEVENTS.csv"))
        self.labs_df = pd.read_csv(os.path.join(dataset_path,"LABEVENTS.csv"))
        if config["debug"]:
            print("Debug mode!")
            self.notes_df = self.notes_df.sort_values(by="HADM_ID")[:3000]
        # self.labels_df = pd.read_csv(
        #     os.path.join(dataset_path, "splits/caml_splits.csv")
        # )
        self.dataset_path = dataset_path
        self.labels_df = self.process_labels(
            os.path.join(dataset_path, "LABEL_SPLITS.csv")
        )
        self.config = config
        self.start_token_id = start_token_id
        self.k_list = config["k_list"]
        self.filter_discharge_summary()
    
    def process_labels(self, label_path):
        LABEL_SPLITS = pd.read_csv(label_path, dtype={"ICD9_CODE": str})
        top50_labels = list(LABEL_SPLITS['ICD9_CODE'].value_counts()[:50].index)
        LABEL_SPLITS_50_UNFILTERED = LABEL_SPLITS[LABEL_SPLITS['ICD9_CODE'].isin(top50_labels)]
        LABEL_SPLITS_50 = LABEL_SPLITS_50_UNFILTERED[['HADM_ID','SPLIT_50','ICD9_CODE']].drop_duplicates()
        # Get one hot encoding of columns ICD9_CODE
        one_hot = pd.get_dummies(LABEL_SPLITS_50['ICD9_CODE'])
        # Drop column ICD9_CODE as it is now encoded
        LABEL_SPLITS_50 = LABEL_SPLITS_50.drop('ICD9_CODE',axis = 1)
        # Join the encoded df and filter only relevant columns
        LABEL_SPLITS_50 = LABEL_SPLITS_50.join(one_hot)[['HADM_ID','SPLIT_50']+top50_labels]
        # Aggreagte on HADM_ID to get multi-label df
        LABEL_SPLITS_50 = LABEL_SPLITS_50.groupby(['HADM_ID','SPLIT_50']).sum().reset_index()
        LABEL_SPLITS_50['ICD9_CODE_BINARY'] = LABEL_SPLITS_50[top50_labels].values.tolist()
        LABEL_SPLITS_50 = LABEL_SPLITS_50.drop(top50_labels, axis=1)
        LABEL_SPLITS_50["SPLIT"] = LABEL_SPLITS_50["SPLIT_50"]
        
        return LABEL_SPLITS_50

    def aggregate_data(self):
        """Preprocess data and aggregate."""
        notes_agg_df = self.aggregate_hadm_id()
        notes_agg_df, categories_mapping = self.add_category_information(notes_agg_df)
        notes_agg_df = self.add_temporal_information(notes_agg_df)
        # notes_agg_df = self.add_multi_hot_encoding(notes_agg_df)
        # notes_agg_df = self.prepare_setup(notes_agg_df)
        labs_agg_df = self.aggregate_labs(notes_agg_df[["HADM_ID", "ADMISSION_TIME", "DISCHARGE_TIME"]], self.config['filter_abnormal'])
        labs_agg_df = self.add_temporal_information(labs_agg_df)
        notes_agg_df = self.prepare_setup(notes_agg_df, labs_agg_df['HADM_ID'])
        return notes_agg_df, categories_mapping, labs_agg_df

    def prepare_setup(self, notes_agg_df, col):
        """Prepare notes depending on experiment set-up"""
        unique_ids = col.unique().tolist()
        print("before", notes_agg_df.shape)
        print("ids", len(unique_ids))
        notes_agg_df = notes_agg_df[notes_agg_df['HADM_ID'].isin(unique_ids)]
        print("after", notes_agg_df.shape)
        return notes_agg_df

    def filter_discharge_summary(self):
        """Filter only DS if needed.
        Based on HTDC
        """
        if self.config["only_discharge_summary"]:
            self.notes_df = self.notes_df[self.notes_df.CATEGORY == "Discharge summary"]

    def aggregate_charts(self, hadm_id, conn):
        """Aggregates chart events for one ID hadm_id.
        """
        all_chart = pd.read_sql(f"SELECT d_items.LABEL, chartevents.SUBJECT_ID, chartevents.HADM_ID, chartevents.CHARTTIME, chartevents.VALUE, chartevents.VALUEUOM FROM chartevents INNER JOIN d_items ON chartevents.ITEMID = d_items.ITEMID WHERE chartevents.HADM_ID = {hadm_id} ORDER BY chartevents.CHARTTIME ASC", conn, columns=['HADM_ID', 'SUBJECT_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM', 'VALUEUOM'])
        all_chart = all_chart[all_chart.hadm_id.isna() == False]
        all_chart["hadm_id"] = all_chart["hadm_id"].apply(int)
        
        all_chart = all_chart[all_chart.charttime.isna() == False]
        all_chart["charttime"] = pd.to_datetime(all_chart.charttime)

        charts_agg_df = (
            all_chart.sort_values(
                by=["charttime"],
                na_position="last",
            )
            .groupby(["subject_id", "hadm_id"]) # TODO: check if subject_id is necessary
            .agg({"label": list, "charttime": list, "valuenum": list, "valueuom": list})
        ).reset_index()

        charts_agg_df.rename(columns={"subject_id": "SUBJECT_ID", "hadm_id": "HADM_ID", "label": "LABEL", "charttime": "CHARTTIME", "valuenum": "VALUENUM", "valueuom": "VALUEUOM"}, inplace=True)

        return charts_agg_df

    def _bin_num(self, df, important_features, start_token_id, k_list):

        df = pd.merge(df, df.pivot(columns='LABEL', values='VALUENUM'), left_index=True, right_index=True)
        
        # normalize features with train set
        df = df.merge(self.labels_df[['HADM_ID', 'SPLIT']], on=["HADM_ID"], how="left")
        df['is_na'] = df[important_features].isnull().all(1)
        df = df[df['is_na'] == False]

        df['ORIG_VAL'] = df[important_features].values.tolist()

        normalizer = QuantileTransformer(
                output_distribution='normal',
                n_quantiles=max(min(df[df.SPLIT == 'TRAIN'].shape[0] // 30, 1000), 10),
                subsample=None,
                random_state=24,
            )
        df.loc[df.SPLIT == 'TRAIN', important_features] = normalizer.fit_transform(df.loc[df.SPLIT == 'TRAIN', important_features])
        df.loc[df.SPLIT == 'VALIDATION', important_features] = normalizer.transform(df.loc[df.SPLIT == 'VALIDATION', important_features])
        df.loc[df.SPLIT == 'TEST', important_features] = normalizer.transform(df.loc[df.SPLIT == 'TEST', important_features])

        fbin_names, wbin_names = {}, {}
        for k in k_list:
            fbin_names[k] = []
            wbin_names[k] = []
            for ft in important_features:
                df.loc[df.SPLIT == 'TRAIN', f'{ft}_FBIN_{k}'], fbins = pd.qcut(df.loc[df.SPLIT == 'TRAIN', ft], q=k, labels=False, retbins=True, duplicates='drop')
                df.loc[df.SPLIT == 'VALIDATION', f'{ft}_FBIN_{k}'] = pd.cut(df.loc[df.SPLIT == 'VALIDATION', ft], bins=fbins, labels=False)
                df.loc[df.SPLIT == 'TEST', f'{ft}_FBIN_{k}'] = pd.cut(df.loc[df.SPLIT == 'TEST', ft], bins=fbins, labels=False)

                df.loc[df.SPLIT == 'TRAIN', f'{ft}_WBIN_{k}'], wbins = pd.cut(df.loc[df.SPLIT == 'TRAIN',ft], bins=k, labels=False, retbins=True, duplicates='drop')
                df.loc[df.SPLIT == 'VALIDATION', f'{ft}_WBIN_{k}'] = pd.cut(df.loc[df.SPLIT == 'VALIDATION', ft], bins=wbins, labels=False)
                df.loc[df.SPLIT == 'TEST', f'{ft}_WBIN_{k}'] = pd.cut(df.loc[df.SPLIT == 'TEST', ft], bins=wbins, labels=False)
                
                df.loc[:, f'{ft}_FBIN_{k}'] += 1
                df.loc[:, f'{ft}_WBIN_{k}'] += 1
                fbin_names[k].append(f'{ft}_FBIN_{k}')
                wbin_names[k].append(f'{ft}_WBIN_{k}')
        
            df[f'FBIN_{k}'] = df[fbin_names[k]].values.tolist()
            df[f'WBIN_{k}'] = df[wbin_names[k]].values.tolist()
        df['NORM_VAL'] = df[important_features].values.tolist()

        # Function to remove NAs and extract the single value
        def clean_bin(bin_list):
            # Remove NAs from the list
            cleaned_list = [x for x in bin_list if not pd.isna(x)]
            # Return the single value, assuming there's exactly one value left
            return cleaned_list[0] if cleaned_list else np.nan
        
        all_bin_names = []
        for k in k_list:
            df[f'FBIN_{k}'] = df[f'FBIN_{k}'].apply(clean_bin)
            df[f'FBIN_{k}'] += (start_token_id - 1)
            df[f'WBIN_{k}'] = df[f'WBIN_{k}'].apply(clean_bin)
            df[f'WBIN_{k}'] += (start_token_id - 1)
            all_bin_names.extend([f'FBIN_{k}', f'WBIN_{k}'])

        df['NORM_VAL'] = df['NORM_VAL'].apply(clean_bin)
        df['ORIG_VAL'] = df['ORIG_VAL'].apply(clean_bin)

        return df, all_bin_names

    
    def aggregate_labs(self, adm_disch_time, filter_abnormal=False):
        # filter NA
        self.labs_df = self.labs_df[self.labs_df.HADM_ID.isna() == False]
        self.labs_df["HADM_ID"] = self.labs_df["HADM_ID"].apply(int)

        # handle time
        # self.labs_df.CHARTTIME = self.labs_df.CHARTTIME.fillna(
        #     self.labs_df.CHARTDATE + " 12:00:00"
        # drop NAs in charttime
        self.labs_df = self.labs_df[self.labs_df.CHARTTIME.isna() == False]
        self.labs_df["CHARTTIME"] = pd.to_datetime(self.labs_df.CHARTTIME)

        self.labs_df = self.labs_df.merge(adm_disch_time, on=["HADM_ID"], how="inner")
        self.labs_df = self.labs_df[self.labs_df["CHARTTIME"] < self.labs_df["DISCHARGE_TIME"]]
        
        self.labs_df["FLAG"] = self.labs_df["FLAG"].fillna('unknown')
        self.labs_df.loc["FLAG_INDEX"] = 0 # default
        self.labs_df.loc[self.labs_df["FLAG"] == 'abnormal', "FLAG_INDEX"] = 1
        self.labs_df.loc[self.labs_df["FLAG"] == 'unknown', "FLAG_INDEX"] = 2
        print({"delta": 0, "abnormal": 1, "unknown": 2})

        imp_lab_path = "lab_ft-imp.txt"
        if filter_abnormal:
            print("Using abnormal lab importance list.")
            # self.labs_df = self.labs_df[self.labs_df['FLAG'] == "abnormal"]
            imp_lab_path = "lab_abnormal_ft-imp.txt"

        # merge with dict to get label name
        D_LABITEMS = pd.read_csv(os.path.join(self.dataset_path, "D_LABITEMS.csv"))
        self.labs_df = self.labs_df.merge(D_LABITEMS.loc[:, ['ITEMID', 'LABEL']], on='ITEMID', how='inner').loc[:, ['SUBJECT_ID', 'HADM_ID', 'LABEL', 'CHARTTIME', 'VALUENUM', 'FLAG_INDEX']]

        with open(os.path.join(self.dataset_path, imp_lab_path), 'r') as f:
            imp_labs = f.read().splitlines()
        f.close()
        self.labs_df = self.labs_df[self.labs_df['LABEL'].isin(imp_labs)]

        self.labs_df, all_bin_names = self._bin_num(self.labs_df, imp_labs, self.start_token_id, self.k_list)
        base_string = "{label}: {value}"
        self.labs_df['TEXT'] = self.labs_df.apply(lambda r: base_string.format(label=r['LABEL'], value=r['ORIG_VAL']), axis=1)

        # sort for chartdate and time
        agg_cols = ["LABEL", "CHARTTIME", "NORM_VAL", "TEXT", "ORIG_VAL", "FLAG_INDEX"] + all_bin_names
        labs_agg_df = (
            self.labs_df.sort_values(
                by=["CHARTTIME"],
                na_position="last",
            )
            .groupby(["SUBJECT_ID", "HADM_ID"])
            .agg({col:list for col in agg_cols})
        ).reset_index()

        labs_agg_df = labs_agg_df.merge(adm_disch_time, on=["HADM_ID"], how="inner")

        # merge with label to get splits
        labs_agg_df = labs_agg_df.merge(self.labels_df, on=["HADM_ID"], how="left")
        labs_agg_df = labs_agg_df[labs_agg_df.SPLIT_50.isna() != True]

        return labs_agg_df
    
    def aggregate_hadm_id(self):
        """Aggregate all notes of the same HADM_ID
        Based on HTDC
        """
        # Filter NA hadm_id
        self.notes_df = self.notes_df[self.notes_df.HADM_ID.isna() == False]
        self.notes_df["HADM_ID"] = self.notes_df["HADM_ID"].apply(int)

        # if time is missing -> assume 12:00:00
        self.notes_df.CHARTTIME = self.notes_df.CHARTTIME.fillna(
            self.notes_df.CHARTDATE + " 12:00:00"
        )
        self.notes_df["CHARTTIME"] = pd.to_datetime(self.notes_df.CHARTTIME)

        self.notes_df["is_discharge_summary"] = (
            self.notes_df.CATEGORY == "Discharge summary"
        )
        notes_agg_df = (
            self.notes_df.sort_values(
                by=["CHARTDATE", "CHARTTIME", "is_discharge_summary"],
                na_position="last",
            )
            .groupby(["SUBJECT_ID", "HADM_ID"])
            .agg({"TEXT": list, "CHARTDATE": list, "CHARTTIME": list, "CATEGORY": list})
        ).reset_index()

        # Modify the TEXT, CHARTDATE, CHARTTIME, CATEGORY columns
        # by limiting the list to include all elements until the first DS
        # the apply function is used to apply the lambda function to each row
        # and it should filter based on the CATEGORY column
        notes_agg_df["TEXT"] = notes_agg_df[["TEXT", "CATEGORY"]].apply(
            lambda x: x.TEXT[: x.CATEGORY.index("Discharge summary") + 1]
            if "Discharge summary" in x.CATEGORY
            else x.TEXT,
            axis=1,
        )

        notes_agg_df["CHARTDATE"] = notes_agg_df[["CHARTDATE", "CATEGORY"]].apply(
            lambda x: x.CHARTDATE[: x.CATEGORY.index("Discharge summary") + 1]
            if "Discharge summary" in x.CATEGORY
            else x.CHARTDATE,
            axis=1,
        )
        notes_agg_df["CHARTTIME"] = notes_agg_df[["CHARTTIME", "CATEGORY"]].apply(
            lambda x: x.CHARTTIME[: x.CATEGORY.index("Discharge summary") + 1]
            if "Discharge summary" in x.CATEGORY
            else x.CHARTTIME,
            axis=1,
        )
        notes_agg_df["CATEGORY"] = notes_agg_df["CATEGORY"].apply(
            lambda x: x[: x.index("Discharge summary") + 1]
            if "Discharge summary" in x
            else x,
        )

        notes_agg_df["ADMISSION_TIME"] = notes_agg_df["CHARTTIME"].apply(
            lambda s: s[0]
        )

        notes_agg_df["DISCHARGE_TIME"] = notes_agg_df["CHARTTIME"].apply(
            lambda s: s[-1]
        )

        # Aggregate with the labels df
        notes_agg_df = notes_agg_df.merge(self.labels_df, on=["HADM_ID"], how="left")

        # Keep only rows for top 50 ICD9 codes
        notes_agg_df = notes_agg_df[notes_agg_df.SPLIT_50.isna() != True]
        return notes_agg_df

    def _timedelta_to_hours(self, timedelta):
        td = timedelta.components
        return td.days * 24 + td.hours + td.minutes / 60.0 + td.seconds / 3600.0

    def _calculate_time_elapsed(self, s):
        try:
            return [s.CHARTTIME[i] - s.ADMISSION_TIME for i in range(len(s.CHARTTIME))]
        except ValueError as e:
            print(s)
    
    def _calculate_percent_elapsed(self, s):
        return [
                max(0, (s.CHARTTIME[i] - s.ADMISSION_TIME) / (s.DISCHARGE_TIME - s.ADMISSION_TIME)) if s.DISCHARGE_TIME - s.ADMISSION_TIME > pd.Timedelta(0) else 1
                for i in range(len(s.CHARTTIME)) # NOTE: edited to include percent elapsed for discharge summary
            ] # TODO: ensure DS percent is always largest
    
    def add_temporal_information(self, agg_df):
        """Add time information."""
        # Add temporal information
        # if ("ADMISSION_TIME" not in agg_df.columns) and ("DISCHARGE_TIME" not in agg_df.columns):
        #     agg_df = agg_df.merge(adm_disch_time, on=["HADM_ID"], how="inner")

        if agg_df.empty:
            return agg_df
        
        agg_df["TIME_ELAPSED"] = agg_df[["CHARTTIME", "ADMISSION_TIME"]].apply(
            self._calculate_time_elapsed,
            axis=1
        )
        
        agg_df["PERCENT_ELAPSED"] = agg_df[["CHARTTIME", "ADMISSION_TIME", "DISCHARGE_TIME"]].apply(
            self._calculate_percent_elapsed,
            axis=1
        )
        agg_df["HOURS_ELAPSED"] = agg_df["TIME_ELAPSED"].apply(
            lambda s: [self._timedelta_to_hours(td) for td in s]
        )
        return agg_df
    
    def _func(self, x):
        if x == 0:
            return 2
        else:
            return x

    def _get_reverse_seqid_by_category(self, category_ids):
        # This creates the CATEGORY_REVERSE_SEQID field for use in note selection later
        # For each category, the last note is assigned to index 0, the second last note is assigned index 1, and so on
        category_ids = pd.Series(category_ids)
        category_ranks = category_ids.groupby(category_ids).cumcount(ascending=False)
        return list(category_ranks)

    def add_category_information(self, notes_agg_df):
        # Create Category IDs
        categories = list(
            notes_agg_df["CATEGORY"]
            .apply(lambda x: pd.Series(x))
            .stack()
            .value_counts()
            .index
        )
        categories_mapping = {categories[i]: i for i in range(len(categories))}
        print(categories_mapping)

        notes_agg_df["CATEGORY_INDEX"] = notes_agg_df["CATEGORY"].apply(
            lambda x: [categories_mapping[c] for c in x]
        )

        # The "Nursing/Other" category is present in the train set but not the dev/test sets
        # We group them together with notes in the "Nursing" category as described in our paper

        notes_agg_df["CATEGORY_INDEX"] = notes_agg_df["CATEGORY_INDEX"].apply(
            lambda x: [self._func(y) for y in x]
        )

        notes_agg_df["CATEGORY_REVERSE_SEQID"] = notes_agg_df["CATEGORY_INDEX"].apply(
            self._get_reverse_seqid_by_category
        )
        return notes_agg_df, categories_mapping

    def _multi_hot_encode(self, codes, code_counts):
        """Return a multi hot encoded vector.

        The resulting multi-hot encoded vector contains ALL labels.
        The top 50 labels can then be filtered using [:50]

        Args:
            codes (list): sample labels
            code_counts (pd.series): series mapping code to frequency

        Return:
            multi_hot (list): list of 0s and 1s
        """
        res = []

        ref = code_counts.index.tolist()

        for c in ref:
            if c in codes:
                res.append(1.0)
            else:
                res.append(0.0)

        return res

    def add_multi_hot_encoding(self, notes_agg_df):
        # liter eval: evaluate the string into a list
        notes_agg_df["ICD9_CODE"] = notes_agg_df["ICD9_CODE"].apply(
            lambda x: ast.literal_eval(x)
        )
        code_counts = (
            notes_agg_df["ICD9_CODE"]
            .apply(lambda x: pd.Series(x))
            .stack()
            .value_counts()
        )

        notes_agg_df["ICD9_CODE_BINARY"] = notes_agg_df["ICD9_CODE"].apply(
            lambda x: self._multi_hot_encode(x, code_counts)
        )

        # We focus on the MIMIC-III-50 splits
        notes_agg_df["SPLIT"] = notes_agg_df["SPLIT_50"]

        notes_agg_df = notes_agg_df[notes_agg_df.SPLIT.isna() != True]

        return notes_agg_df

