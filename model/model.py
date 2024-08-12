import torch
import torch.nn as nn
import transformers
import torch.nn.functional as F
import ipdb
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoModel,
    RobertaConfig
)
from model.tpberta_modeling import *

import sys
import os
import numpy as np

from data.utils import get_cutoffs


class TabularEncoder(nn.Module):
    """ Encodes tabular data."""
    def __init__(self, pretrained_dir, num_labels=50, freeze_tabular=True):
        super().__init__() 

        # Use pretrained encoder
        self.config = RobertaConfig.from_pretrained(pretrained_dir)
        self.model = TPBertaForClassification.from_pretrained(os.path.join(pretrained_dir, 'pytorch_models/best'), config=self.config, num_class=num_labels)

        if freeze_tabular:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            # freeze everything except embeddings
            print("Finetuning tabular encoder.")
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.tpberta.embeddings.parameters():
                param.requires_grad = True
            for param in self.model.tpberta.intra_attention.parameters():
                param.requires_grad = True

    def forward(self, input_ids, input_scales, token_type_ids, position_ids, features_cls_mask):
        
        embedding_output = self.model.tpberta.embeddings(input_ids=input_ids, input_scales=input_scales, \
                          token_type_ids=token_type_ids, position_ids=position_ids)

        feature_chunk = self.model.tpberta.intra_attention(
            embedding_output,
            query_mask=features_cls_mask,
            input_ids=input_ids,
        )
        return feature_chunk

class TabularMapper(nn.Module):
    """Map tabular embeddings to chunk embedding space.
    """
    def __init__(self, tabular_dim, chunk_dim):
        super().__init__() 

        self.linear = nn.Linear(tabular_dim, chunk_dim)
        self.relu = nn.LeakyReLU()
    
    def forward(self, tabular_emb):
        mapped_emb = self.linear(tabular_emb)
        mapped_emb = self.relu(mapped_emb)

        return mapped_emb


class LabelAttentionClassifier(nn.Module):
    """ Legacy code from Clare's implementation of HTDC (Ng et al, 2022).
    This code is not used in our implementation, but is included for completeness."""
    def __init__(self, hidden_size, num_labels):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_labels = num_labels

        self.label_queries = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.hidden_size, self.num_labels), dtype=torch.float
            ),
            requires_grad=True,
        )
        self.label_weights = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.hidden_size, self.num_labels), dtype=torch.float
            ),
            requires_grad=True,
        )

    def forward(self, encoding, cutoffs=None):
        # encoding: Tensor of size num_chunks x hidden_size

        attention_weights = F.softmax(
            encoding @ self.label_queries, dim=0
        )  # (num_chunks x num_labels)

        attention_value = encoding.T @ attention_weights  # hidden_size x num labels

        score = torch.sum(attention_value * self.label_weights, dim=0)  # num_labels
        # probability = torch.sigmoid(score)

        # return probability
        return score.unsqueeze(0)  # CHANGED THIS FOR DEBUGGING


class HierARDocumentTransformer(nn.Module):
    """Hierarchical Autoregressive Transformer.

    This class includes the hierarchical autoregressive transformer,
    which runs over the document embeddings applying masked
    multihead attention to the previous document embeddings.
    """
    def __init__(self, hidden_size, num_layers=1, nhead=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=nhead
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, num_layers=num_layers
        )

    def forward(self, document_encodings):
        # flag is causal = True so that it cannot attend to future document embeddings
        mask = nn.Transformer.generate_square_subsequent_mask(
            sz=document_encodings.shape[0]
        )
        document_encodings = self.transformer_encoder(
            document_encodings, mask=mask
        ).squeeze(
            1
        )  # shape Nc x 1 x D

        return document_encodings


class NextDocumentCategoryPredictor(nn.Module):
    """Document Category Predictor.

    This class generates the next document category
    based on the current document embedding.
    """
    def __init__(self, hidden_size, num_categories):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_categories = num_categories
        self.linear = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.linear2 = nn.Linear(
            self.hidden_size // 2, num_categories + 1
        )  # 11 is the number of categories

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, document_encodings):
        # predict next document category
        categories = self.relu(self.linear(document_encodings))
        categories = self.softmax(self.linear2(categories))
        return categories


class NextDocumentEmbeddingPredictor(nn.Module):
    """Document Embedding Generator.

    This class generates the next (or last) document embedding
    based on the current document embedding.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(self.hidden_size // 2, self.hidden_size)
        self.relu2 = nn.ReLU()

    def forward(self, document_encodings):
        # predict next document embedding
        next_document_encodings = self.relu1(self.linear(document_encodings))
        next_document_encodings = self.linear2(next_document_encodings)
        return next_document_encodings


class TemporalMultiHeadLabelAttentionClassifier(nn.Module):
    """ Masked Multihead Label Attention Classifier.
    
    Performs masked multihead attention using label embeddings
    as queries and document encodings as keys and values.

    This class also applies linear projection and sigmoid
    to obtain the final probability of each label.
    """
    def __init__(
        self,
        hidden_size,
        seq_len,
        num_labels,
        num_heads,
        device,
        all_tokens=True,
        reduce_computation=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.device = device
        self.all_tokens = all_tokens
        self.reduce_computation = reduce_computation

        self.multiheadattn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, batch_first=True
        )

        self.label_queries = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.num_labels, self.hidden_size), dtype=torch.float
            ),
            requires_grad=True,
        )
        self.label_weights = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.hidden_size, self.num_labels), dtype=torch.float
            ),
            requires_grad=True,
        )

    def forward(self, encoding, all_tokens=True, cutoffs=None):
        # encoding: Tensor of size (Nc x T) x H
        # mask: Tensor of size Nn x (Nc x T) x H
        # temporal_encoding = Nn x (N x T) x hidden_size
        T = self.seq_len
        if not self.all_tokens:
            T = 1  # only use the [CLS]-token representation
        Nc = int(encoding.shape[0] / T)
        H = self.hidden_size
        Nl = self.num_labels

        # label query: shape L, H
        # encoding: hape NcxT, H
        # query shape:  Nn, L, H
        # key shape: Nn, Nc*T, H
        # values shape: Nn, Nc*T, H
        # key padding mask: Nn, Nc*T (true if ignore)
        # output: N, L, H
        mask = torch.ones(size=(Nc, Nc * T), dtype=torch.bool).to(device=self.device)
        for i in range(Nc):
            mask[i, : (i + 1) * T] = False

        # only mask out at 2d, 5d, 13d and no DS to reduce computation
        # get list of cutoff indices from cutoffs dictionary

        if self.reduce_computation:
            cutoff_indices = [cutoffs[key][0] for key in cutoffs]
            mask = mask[cutoff_indices, :]

        attn_output = self.multiheadattn.forward(
            query=self.label_queries.repeat(mask.shape[0], 1, 1),
            key=encoding.repeat(mask.shape[0], 1, 1),
            value=encoding.repeat(mask.shape[0], 1, 1),
            key_padding_mask=mask,
            need_weights=False,
        )[0]

        score = torch.sum(
            attn_output
            * self.label_weights.unsqueeze(0).view(
                1, self.num_labels, self.hidden_size
            ),
            dim=2,
        )
        return score


class Model(nn.Module):
    """Model for ICD-9 code temporal predictions.

    Code based on HTDC (Ng et al, 2022).
    
    Our contributions:
    - Hierarchical autoregressive transformer
    - Auxiliary tasks, including:
        - next document embedding predictor 
        (which can also be used for last emb. pred.)
        - next document category predictor
    """

    def __init__(self, config, device):
        super().__init__()
        for key in config:
            setattr(self, key, config[key])

        self.seq_len = 512
        self.hidden_size = 768
        self.device = device
        self._initialize_embeddings()

        # base transformer
        self.transformer = AutoModel.from_pretrained(self.base_checkpoint)

        # LWAN
        if self.use_multihead_attention:
            self.label_attn = TemporalMultiHeadLabelAttentionClassifier(
                self.hidden_size,
                self.seq_len,
                self.num_labels,
                self.num_heads_labattn,
                device=device,
                all_tokens=self.use_all_tokens,
                reduce_computation=self.reduce_computation,
            )
            # self.label_attn = TemporalLabelAttentionClassifier(
            #     self.hidden_size,
            #     self.seq_len,
            #     self.num_labels,
            #     self.num_heads_labattn,
            #     device=device,
            #     all_tokens=self.use_all_tokens,
            # )
        else:
            self.label_attn = LabelAttentionClassifier(
                self.hidden_size, self.num_labels
            )
        # hierarchical AR transformer
        if not self.is_baseline:
            self.document_regressor = HierARDocumentTransformer(
                self.hidden_size, self.num_layers, self.num_attention_heads
            )
        if self.aux_task in ("next_document_embedding", "last_document_embedding"):
            self.document_predictor = NextDocumentEmbeddingPredictor(self.hidden_size)
        elif self.aux_task == "next_document_category":
            self.category_predictor = NextDocumentCategoryPredictor(
                self.hidden_size, self.num_categories
            )
        elif self.aux_task != "none":
            raise ValueError(
                "auxiliary_task must be next_document_embedding or next_document_category or none"
            )
        
        # tabular encoder
        if self.use_tabular:
            self.tabular_encoder = TabularEncoder(self.tabular_base_checkpoint, freeze_tabular=self.freeze_tabular)
            self.tabular_dim = self.tabular_encoder.config.hidden_size
            self.tabular_mapper = TabularMapper(self.tabular_dim, self.hidden_size)
            self.chunk_length = 500
            if self.use_tabular_attn:
                self.tabular_regressor = HierARDocumentTransformer(
                    self.hidden_size, self.num_layers, self.num_attention_heads
                )

    def _initialize_embeddings(self):
        self.pelookup = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.max_chunks, 1, self.hidden_size), dtype=torch.float
            ),
            requires_grad=True,
        )
        self.reversepelookup = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.max_chunks, 1, self.hidden_size), dtype=torch.float
            ),
            requires_grad=True,
        )
        self.delookup = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.max_chunks, 1, self.hidden_size), dtype=torch.float
            ),
            requires_grad=True,
        )
        self.reversedelookup = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.max_chunks, 1, self.hidden_size), dtype=torch.float
            ),
            requires_grad=True,
        )
        self.celookup = nn.parameter.Parameter(
            torch.normal(0, 0.1, size=(15, 1, self.hidden_size), dtype=torch.float),
            requires_grad=True,
        )

        self.melookup = nn.parameter.Parameter(
            torch.normal(0, 0.1, size=(2, 1, self.hidden_size), dtype=torch.float),
            requires_grad=True,
        )
    
    def _squeeze_data(self, tabular_data):
        return {k:v[0] for k,v in tabular_data.items()}
    
    def combine_sequences(self, note_sequence, tabular_sequence, note_times, 
                           tabular_times):
        combined_matrix = torch.cat([note_sequence, tabular_sequence], dim=0) # (N+M) x D
        combined_times = torch.cat([note_times, tabular_times])

        assert combined_matrix.shape[0] == combined_times.shape[0]
        
        # Get sorted index of times
        sorted_indices = torch.argsort(combined_times)
        
        sorted_matrix = combined_matrix[sorted_indices]
        sorted_times = combined_times[sorted_indices]
        
        return sorted_matrix, sorted_times

    def tabular_pooling(self, feature_embedding, k, pooling_type='max'):
        # NK x D
        n, d = feature_embedding.shape
        feature_embedding = feature_embedding.reshape(-1,k,d)
        pooled_ind = np.array([i for i in range(0,n,k)])
        if pooling_type == 'max':
            pooled = feature_embedding.max(dim=1).values # n x d
        elif pooling_type == 'sum':
            pooled = feature_embedding.sum(dim=1) # n x d
        else:
            raise ValueError
        assert pooled.shape == (n//k, d)
        return pooled, pooled_ind
    
    def temporal_pooling(self, feature_embedding, time_elapsed, pooling_type='max'):
        # get unique times
        unique_times = torch.unique(time_elapsed)
        # for each unique time, pool features with the time
        complete_pooled = []
        pooled_ind = []
        for i in range(unique_times.shape[0]):
            time_ind = np.where(time_elapsed == unique_times[i])[0]
            pooled_ind.append(time_ind[0])
            time_subset = feature_embedding[time_ind] # M x D
            if pooling_type == 'max':
                time_pooled = time_subset.max(dim=0, keepdim=True).values
            elif pooling_type == 'sum':
                time_pooled = time_subset.sum(dim=0, keepdim=True).values # 1 x D
            else:
                raise ValueError
            complete_pooled.append(time_pooled)
        temporal_pooled = torch.cat(complete_pooled, dim=0)

        return temporal_pooled, np.array(pooled_ind)


    def forward(
        self,
        input_ids,
        attention_mask,
        seq_ids,
        category_ids,
        cutoffs,
        percent_elapsed,
        hours_elapsed,
        note_end_chunk_ids=None,
        tabular_data=None,
        token_type_ids=None,
        is_evaluation=False,
        return_attn_weights=False,
        **kwargs
    ):
        max_seq_id = seq_ids[-1].item()
        reverse_seq_ids = max_seq_id - seq_ids

        chunk_count = input_ids.size()[0]
        reverse_pos_ids = (chunk_count - torch.arange(chunk_count) - 1).to(self.device)

        sequence_output = self.transformer(input_ids, attention_mask).last_hidden_state

        if self.use_positional_embeddings:
            sequence_output += self.pelookup[: sequence_output.size()[0], :, :]
        if self.use_reverse_positional_embeddings:
            sequence_output += torch.index_select(
                self.reversepelookup, dim=0, index=reverse_pos_ids
            )

        if self.use_document_embeddings:
            sequence_output += torch.index_select(self.delookup, dim=0, index=seq_ids)
        if self.use_reverse_document_embeddings:
            sequence_output += torch.index_select(
                self.reversedelookup, dim=0, index=reverse_seq_ids
            )

        if self.use_category_embeddings:
            sequence_output += torch.index_select(
                self.celookup, dim=0, index=category_ids
            )
        if self.use_modality_embeddings:
            modality_ids = torch.zeros_like(category_ids, dtype=torch.long)
            sequence_output += torch.index_select(
                self.melookup, dim=0, index=modality_ids
            )

        if self.use_all_tokens:
            # before: sequence_output shape [batchsize, seqlen, hiddensize] = [# chunks, 512, hidden size]
            # after: sequence_output shape [#chunks*512, 1, hidden size]
            sequence_output_all = sequence_output.view(-1, 1, self.hidden_size)
            sequence_output_all = sequence_output_all[:, 0, :]
            sequence_output = sequence_output[:, [0], :]

        else:
            sequence_output = sequence_output[:, [0], :]

        sequence_output = sequence_output[
            :, 0, :
        ]  # remove the singleton to get something of shape [#chunks, hidden_size] or [#chunks*512, hidden_size]

        # TODO: add tabular data here
        if self.use_tabular and tabular_data:
            if len(tabular_data['input_ids'].shape) > 2:
                tabular_data = {k:v[0] for k,v in tabular_data.items()}
            tabular_input_ids = tabular_data['input_ids']
            tabular_input_scales = tabular_data['input_scales']
            features_cls_mask = tabular_data['features_cls_mask']
            tabular_token_type_ids = tabular_data['token_type_ids']
            tabular_position_ids = tabular_data['position_ids']
            tabular_percent_elapsed = tabular_data['percent_elapsed']
            tabular_hours_elapsed = tabular_data['hours_elapsed']
            complete_tabular_output = []
            for i in range(0, tabular_input_ids.shape[0], self.chunk_length):
                tabular_output = self.tabular_encoder(
                    input_ids=tabular_input_ids[i : i + self.chunk_length].to(self.device, dtype=torch.long), 
                    input_scales=tabular_input_scales[i : i + self.chunk_length].to(self.device, dtype=torch.float32),
                    token_type_ids=tabular_token_type_ids[i : i + self.chunk_length].to(self.device, dtype=torch.long),
                    position_ids=tabular_position_ids[i : i + self.chunk_length].to(self.device, dtype=torch.long), 
                    features_cls_mask=features_cls_mask[i : i + self.chunk_length].to(self.device, dtype=torch.long)
                    )
                complete_tabular_output.append(tabular_output)
            tabular_output = torch.cat(complete_tabular_output, dim=0)
            assert tabular_output.shape[0] == tabular_input_ids.shape[0]

            tabular_output = tabular_output[:, 0, :]
            tabular_output = self.tabular_mapper(tabular_output)
            # pool tabular features
            if self.pool_features != 'none':
                if self.pool_features == 'temporal':
                    tabular_output, pooled_ind = self.temporal_pooling(tabular_output, tabular_percent_elapsed, pooling_type='max')
                else:    
                    tabular_output, pooled_ind = self.tabular_pooling(tabular_output, len(self.k_list), pooling_type=self.pool_features)
                tabular_percent_elapsed = tabular_percent_elapsed[pooled_ind]
                tabular_hours_elapsed = tabular_hours_elapsed[pooled_ind]
            
            tabular_percent_elapsed = tabular_percent_elapsed.to(self.device, dtype=torch.float16)
            tabular_hours_elapsed = tabular_hours_elapsed.to(self.device, dtype=torch.long)
            
            if self.use_modality_embeddings:
                modality_ids = torch.ones_like(tabular_percent_elapsed, dtype=torch.long)
                tabular_output += torch.index_select(
                    self.melookup, dim=0, index=modality_ids
                ).squeeze(1)  
            
            if self.use_tabular_attn:
                tabular_output = self.tabular_regressor(
                                            tabular_output.view(-1, 1, self.hidden_size)
                                        )  
                
            if self.late_fuse == "none":              
                tabular_cat_proxy = torch.ones_like(tabular_hours_elapsed) * -1    
                sequence_output, _ = self.combine_sequences(sequence_output, tabular_output, percent_elapsed, tabular_percent_elapsed)
                combined_cat, combined_hours = self.combine_sequences(category_ids, tabular_cat_proxy, hours_elapsed, tabular_hours_elapsed)
                cutoffs = get_cutoffs(combined_hours, combined_cat)
        
        # if not baseline, add document autoregressor
        if not self.is_baseline:
            # document regressor returns document embeddings and predicted categories
            sequence_output = self.document_regressor(
                sequence_output.view(-1, 1, self.hidden_size)
            )
            assert not torch.any(torch.isnan(sequence_output))
        # make aux predictions
        if self.aux_task in ("next_document_embedding", "last_document_embedding"):
            if self.apply_transformation:
                aux_predictions = self.document_predictor(sequence_output)
            else:
                aux_predictions = sequence_output
        elif self.aux_task == "next_document_category":
            aux_predictions = self.category_predictor(sequence_output)
        elif self.aux_task == "none":
            aux_predictions = None
        # apply label attention at document-level

        # NOTE: fuse past embeddings with tabular
        tabular_scores = None
        
        if self.use_tabular and tabular_data:
            tabular_cat_proxy = torch.ones_like(tabular_hours_elapsed) * -1  
            
            if self.late_fuse == "embeddings":
                sequence_output, _ = self.combine_sequences(sequence_output, tabular_output, percent_elapsed, tabular_percent_elapsed)
                combined_cat, combined_hours = self.combine_sequences(category_ids, tabular_cat_proxy, hours_elapsed, tabular_hours_elapsed)
                cutoffs = get_cutoffs(combined_hours, combined_cat)  
            elif self.late_fuse == "predictions":
                # feed tabular data through LWAN
                tabular_cutoffs = get_cutoffs(tabular_hours_elapsed, tabular_cat_proxy)
                tabular_scores = self.label_attn(tabular_output, cutoffs=tabular_cutoffs) # M x L x D

        if is_evaluation == False:
            if self.use_all_tokens:
                scores = self.label_attn(sequence_output_all, cutoffs=cutoffs)
            else:
                scores = self.label_attn(sequence_output, cutoffs=cutoffs)
            return scores, sequence_output, aux_predictions, tabular_scores

        else:
            if self.use_all_tokens:
                return sequence_output_all
            else:
                return sequence_output
