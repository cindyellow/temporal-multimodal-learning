from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
import json
from data.utils import get_cutoffs


def return_attn_scores(lwan, encoding, all_tokens=True, cutoffs=None):
    """ Calculate the attention scores for a document encoding using
    a trained LWAN network.
    
    Args:
        lwan: trained LWAN network
        encoding: document encoding
    
    Returns:
        attn_output_weights: attention weights
        score: attention scores"""
    
    # encoding: Tensor of size (Nc x T) x H
    # mask: Tensor of size Nn x (Nc x T) x H
    # temporal_encoding = Nn x (N x T) x hidden_size
    T = lwan.seq_len
    if not lwan.all_tokens:
        T = 1  # only use the [CLS]-token representation
    Nc = int(encoding.shape[0] / T)
    H = lwan.hidden_size
    Nl = lwan.num_labels

    # label query: shape L, H
    # encoding: hape NcxT, H
    # query shape:  Nn, L, H
    # key shape: Nn, Nc*T, H
    # values shape: Nn, Nc*T, H
    # key padding mask: Nn, Nc*T (true if ignore)
    # output: N, L, H
    mask = torch.ones(size=(Nc, Nc * T), dtype=torch.bool).to(device=lwan.device)
    for i in range(Nc):
        mask[i, : (i + 1) * T] = False

    # only mask out at 2d, 5d, 13d and no DS to reduce computation
    # get list of cutoff indices from cutoffs dictionary

    attn_output, attn_output_weights = lwan.multiheadattn.forward(
        query=lwan.label_queries.repeat(mask.shape[0], 1, 1),
        key=encoding.repeat(mask.shape[0], 1, 1),
        value=encoding.repeat(mask.shape[0], 1, 1),
        key_padding_mask=mask,
        need_weights=True,
    )

    score = torch.sum(
        attn_output
        * lwan.label_weights.unsqueeze(0).view(1, lwan.num_labels, lwan.hidden_size),
        dim=2,
    )
    return attn_output_weights, score


def update_weights_per_class(
    labels, cutoffs, category_ids, attn_output_weights, weights_per_class
):
    labels_sample = []
    for i in range(50):
        if labels[i] == 1:
            labels_sample.append(i)
    for cutoff in cutoffs.keys():
        cutoff_idx = cutoffs[cutoff]
        for l in labels_sample:
            attn_weights = (
                attn_output_weights[cutoff_idx, l, :]
                .cpu()
                .detach()
                .numpy()
                .reshape(1, -1)
            )
            for chunk in range(cutoff_idx + 1):
                c = category_ids[chunk].item()
                weights_per_class[cutoff][c].append(
                    attn_output_weights[cutoff_idx, l, chunk].item()
                )
    # update the 'all' key
    cutoff_idx = attn_output_weights.shape[0] - 1
    for l in labels_sample:
        attn_weights = (
            attn_output_weights[cutoff_idx, l, :].cpu().detach().numpy().reshape(1, -1)
        )
        for chunk in range(cutoff_idx + 1):
            c = category_ids[chunk].item()
            weights_per_class["all"][c].append(
                attn_output_weights[cutoff_idx, l, chunk].item()
            )
    return weights_per_class

def select_tabular_window(tabular_data, percent_elapsed, max_features=16):
    # filter tabular data to this time window
    if not tabular_data:
        return None
    left_bound = percent_elapsed[0] if percent_elapsed[0] > 0 else -1
    right_bound = percent_elapsed[-1]

    middle_indices = np.where(
            np.logical_and(tabular_data['percent_elapsed'][0] > left_bound,  
                           tabular_data['percent_elapsed'][0] <= right_bound)
    )[0]
    
    if middle_indices.size == 0:
        return None
    return {"input_ids": tabular_data['input_ids'][0][middle_indices],
        "input_scales": tabular_data['input_scales'][0][middle_indices],
        "features_cls_mask": tabular_data['features_cls_mask'][0][middle_indices],
        "token_type_ids": tabular_data['token_type_ids'][0][middle_indices],
        "position_ids": tabular_data['position_ids'][0][middle_indices],
        "hours_elapsed": tabular_data['hours_elapsed'][0][middle_indices],
        "percent_elapsed": tabular_data['percent_elapsed'][0][middle_indices],
        "flag_ids": tabular_data['flag_ids'][0][middle_indices],
        }


def evaluate(
    mymetrics,
    model,
    generator,
    device,
    pred_cutoff=0.5,
    evaluate_temporal=False,
    optimise_threshold=False,
    num_categories=1,
    is_baseline=False,
    aux_task=None,
    setup="latest",
    reduce_computation=False,
    qualitative_evaluation=False,
    use_tabular=False,
    textualize=False,
    subset_tabular=False,
):
    """ Evaluate the model on the validation set.
    """
    if qualitative_evaluation:
        weights_per_class = {
            cutoff: {c: [] for c in range(15)}
            for cutoff in ["2d", "5d", "13d", "noDS", "all"]
        }

    model.eval()
    with torch.no_grad():
        ids = []
        preds = {"hyps": [], "refs": [], "hyps_aux": [], "refs_aux": []}

        if evaluate_temporal:
            preds["hyps_temp"] = {"2d": [], "5d": [], "13d": [], "noDS": []}
            preds["refs_temp"] = {"2d": [], "5d": [], "13d": [], "noDS": []}

        avail_doc_count = []
        print(f"Starting validation loop...")
        for t, data in enumerate(tqdm(generator)):
            # TODO: fix code so that sequence ids embeddings can be used
            # right now they cannot be used
            labels = data["notes"]["label"][0][: model.num_labels]
            input_ids = data["notes"]["input_ids"][0]
            attention_mask = data["notes"]["attention_mask"][0]
            seq_ids = data["notes"]["seq_ids"][0]
            category_ids = data["notes"]["category_ids"][0]
            percent_elapsed = data["notes"]["percent_elapsed"][0]
            avail_docs = seq_ids.max().item() + 1
            # note_end_chunk_ids = data["notes"]["note_end_chunk_ids"]
            hours_elapsed = data["notes"]["hours_elapsed"][0]
            cutoffs = data["notes"]["cutoffs"]

            if (use_tabular 
                and not textualize 
                and len(data["tabular"]['input_ids']) > 0
            ): # check if there's tabular data available
                tabular_data = data["tabular"]
            else:
                tabular_data=None
            if setup == "random":
                complete_sequence_output = []
                # run through data in chunks of max_chunks
                tabular_elapsed = []
                for i in range(0, input_ids.shape[0], model.max_chunks):
                    # only get the document embeddings
                    tabular_subset = None
                    if use_tabular:
                        tabular_subset = select_tabular_window(tabular_data, 
                                                            percent_elapsed[i : i + model.max_chunks], 
                                                            model.max_tabular_features)                            
                    sequence_output, tabular_hours_elapsed = model(
                        input_ids=input_ids[i : i + model.max_chunks].to(
                            device, dtype=torch.long
                        ),
                        attention_mask=attention_mask[i : i + model.max_chunks].to(
                            device, dtype=torch.long
                        ),
                        seq_ids=seq_ids[i : i + model.max_chunks].to(
                            device, dtype=torch.long
                        ),
                        category_ids=category_ids[i : i + model.max_chunks].to(
                            device, dtype=torch.long
                        ),
                        cutoffs=None,
                        percent_elapsed=percent_elapsed[i : i + model.max_chunks].to(
                            device, dtype=torch.float16
                        ),
                        hours_elapsed=hours_elapsed[i : i + model.max_chunks].to(
                            device, dtype=torch.long
                        ),
                        is_evaluation=True,
                        tabular_data=tabular_subset,
                    )
                    complete_sequence_output.append(sequence_output)
                    if tabular_hours_elapsed is not None:
                        tabular_elapsed.extend(tabular_hours_elapsed)
                # concatenate the sequence output
                sequence_output = torch.cat(complete_sequence_output, dim=0)

                # update cutoff
                if len(tabular_elapsed) > 0:
                    tabular_elapsed = torch.tensor(tabular_elapsed)
                    tabular_cat_proxy = torch.ones_like(tabular_elapsed) * -1
                    combined_cat, combined_hours = model.combine_sequences(category_ids, tabular_cat_proxy, hours_elapsed, tabular_elapsed)
                    cutoffs = get_cutoffs(combined_hours, combined_cat)

                # run through LWAN to get the scores
                scores = model.label_attn(sequence_output, cutoffs=cutoffs)
                if qualitative_evaluation:
                    # NOTE: didn't adapt for tabular
                    attn_output_weights, scores = return_attn_scores(
                        model.label_attn, sequence_output.to(device), cutoffs=cutoffs
                    )
                    weights_per_class = update_weights_per_class(
                        labels,
                        cutoffs,
                        category_ids,
                        attn_output_weights,
                        weights_per_class,
                    )

            else:
                scores, _, aux_predictions, tabular_scores, tabular_hours_elapsed = model(
                    input_ids=input_ids.to(device, dtype=torch.long),
                    attention_mask=attention_mask.to(device, dtype=torch.long),
                    seq_ids=seq_ids.to(device, dtype=torch.long),
                    category_ids=category_ids.to(device, dtype=torch.long),
                    cutoffs=cutoffs,
                    percent_elapsed=percent_elapsed.to(device, dtype=torch.float16),
                    hours_elapsed=hours_elapsed.to(device, dtype=torch.long),
                    # note_end_chunk_ids=note_end_chunk_ids,
                    tabular_data=tabular_data,
                )
                # update cutoff
                if tabular_hours_elapsed is not None:
                    tabular_cat_proxy = torch.ones_like(tabular_hours_elapsed) * -1
                    combined_cat, combined_hours = model.combine_sequences(category_ids.to(device, dtype=torch.long), tabular_cat_proxy, hours_elapsed.to(device, dtype=torch.long), tabular_hours_elapsed)
                    cutoffs = get_cutoffs(combined_hours, combined_cat)

            if aux_task == "next_document_category":
                if len(category_ids) > 1 and aux_predictions is not None:
                    true_categories = F.one_hot(
                        torch.concat(
                            [category_ids[1:], torch.tensor([num_categories])]
                        ),
                        num_classes=num_categories + 1,
                    )
                    preds["hyps_aux"].append(aux_predictions.detach().cpu().numpy())
                    preds["refs_aux"].append(true_categories.detach().cpu().numpy())

            probs = F.sigmoid(scores)
            ids.append(data["notes"]["hadm_id"][0].item())
            avail_doc_count.append(avail_docs)
            preds["hyps"].append(probs[-1, :].detach().cpu().numpy())
            preds["refs"].append(labels.detach().cpu().numpy())
            if evaluate_temporal:
                cutoff_times = ["2d", "5d", "13d", "noDS"]
                for n, time in enumerate(cutoff_times):
                    if cutoffs[time][0] != -1:
                        if reduce_computation:
                            preds["hyps_temp"][time].append(
                                probs[n, :].detach().cpu().numpy()
                            )
                        else:
                            preds["hyps_temp"][time].append(
                                probs[cutoffs[time][0], :].detach().cpu().numpy()
                            )
                        preds["refs_temp"][time].append(labels.detach().cpu().numpy())

        if optimise_threshold:
            pred_cutoff = mymetrics.get_optimal_microf1_threshold_v2(
                np.asarray(preds["hyps"]), np.asarray(preds["refs"])
            )
        else:
            pred_cutoff = 0.5

        val_metrics = mymetrics.from_numpy(
            np.asarray(preds["hyps"]),
            np.asarray(preds["refs"]),
            pred_cutoff=pred_cutoff,
        )
        if len(preds["hyps_aux"]) > 0:
            val_metrics_aux = mymetrics.from_numpy(
                np.concatenate(preds["hyps_aux"]),
                np.concatenate(preds["refs_aux"]),
                pred_cutoff=pred_cutoff,
            )
        else:
            val_metrics_aux = None

        if evaluate_temporal:
            cutoff_times = ["2d", "5d", "13d", "noDS"]
            val_metrics_temp = {
                time: mymetrics.from_numpy(
                    np.asarray(preds["hyps_temp"][time]),
                    np.asarray(preds["refs_temp"][time]),
                    pred_cutoff=pred_cutoff,
                )
                for time in cutoff_times
            }
        else:
            val_metrics_temp = None

        if qualitative_evaluation:
            json.dump(weights_per_class, open("weights_per_class_3.json", "w"))

    return val_metrics, val_metrics_temp, val_metrics_aux

