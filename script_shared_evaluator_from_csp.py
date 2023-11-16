
import argparse
import copy
import json
import os
from itertools import product

# import clip
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy.stats import hmean
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from Models.CspModel.clip_modules.interface import CLIPInterface
from Models.CspModel.clip_modules.model_loader import load
from Models.CspModel.datasets.composition_dataset import CompositionDataset
from Models.CspModel.datasets.read_datasets import DATASET_PATHS
from Models.CspModel.models.compositional_modules import get_model

from ProjUtils.misc import modify_save_path, load_args

cudnn.benchmark = True


DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# -----------------------
# given the weights:
# 1. we can enforce consistency
# 2. pair, attr, obj
# 3. used in predict_logits function
# -----------------------
weight_grid = [
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 1.0],
    [1.0, 0.5, 1.0],
    [1.0, 0.5, 0.5],
    [1.0, 0.1, 1.0],
    [1.0, 0.1, 0.1],
    [1.0, 0.1, 0.5],
    [1.0, 0.01, 0.01],
    [1.0, 0.001, 0.01],
    [1.0, 0.001, 0.001]
]

dir_Map = {
    "ut-zappos": "zappos",
    "mit-states": "mitstates"
}


class Evaluator:
    """
    Evaluator class, adapted from:
    https://github.com/Tushar-N/attributes-as-operators

    With modifications from:
    https://github.com/ExplainableML/czsl
    """

    def __init__(self, dset, model):

        self.dset = dset

        # ---------------------------------------------------------------
        # Convert text pairs to idx tensors: [('sliced', 'apple'), ('ripe',
        # 'apple'), ...] --> torch.LongTensor([[0,1],[1,1], ...])
        # ---------------------------------------------------------------
        all_pair_attr_obj_id = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.all_pairs]
        self.train_pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.train_pairs]
        self.all_pair_attr_obj_id = torch.LongTensor(all_pair_attr_obj_id)

        # Mask over pairs that occur in closed world
        # Select set based on phase
        if dset.phase == 'train':
            print('Evaluating with train pairs')
            test_pair_set = set(dset.train_pair_id_list)
            test_pair_gt = set(dset.train_pair_id_list)
        elif dset.phase == 'val':
            print('Evaluating with validation pairs')
            test_pair_set = set(dset.val_pairs + dset.train_pairs)
            test_pair_gt = set(dset.val_pairs)
        else:
            print('Evaluating with test pairs')
            test_pair_set = set(dset.test_pairs + dset.train_pairs)
            test_pair_gt = set(dset.test_pairs)

        self.test_pair_dict = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in test_pair_gt]
        self.test_pair_dict = dict.fromkeys(self.test_pair_dict, 0)

        # dict values are pair val, score, total
        for attr, obj in test_pair_gt:
            pair_val_id = dset.allPair2idx[(attr, obj)]
            key = (dset.attr2idx[attr], dset.obj2idx[obj])
            self.test_pair_dict[key] = [pair_val_id, 0, 0]

        # open world
        if dset.open_world:
            masks = [1 for _ in dset.all_pairs]
        else:
            masks = [1 if pair in test_pair_set else 0 for pair in dset.all_pairs]

        # masks = [1 if pair in test_pair_set else 0 for pair in dset.pairs]

        self.closed_mask = torch.BoolTensor(masks)
        # Mask of seen concepts
        seen_pair_set = set(dset.train_pairs)
        mask = [1 if pair in seen_pair_set else 0 for pair in dset.all_pairs]
        self.seen_mask = torch.BoolTensor(mask)

        # Object specific mask over which pairs occur in the object oracle
        # setting
        oracle_obj_mask = []
        for _obj in dset.objs:
            mask = [1 if _obj == obj else 0 for attr, obj in dset.all_pairs]
            oracle_obj_mask.append(torch.BoolTensor(mask))
        self.oracle_obj_mask = torch.stack(oracle_obj_mask, 0)

        # Decide if the model under evaluation is a manifold model or not
        self.score_model = self.score_manifold_model

    # Generate mask for each settings, mask scores, and get prediction labels
    def generate_predictions(self, scores, obj_truth, bias=0.0, topk=1):  # (Batch, #pairs)
        '''
        Inputs
            scores: Output scores
            obj_truth: Ground truth object
        Returns
            results: dict of results in 3 settings
        '''

        def get_pred_from_scores(_scores, topk):
            """
            Given list of scores, returns top 10 attr and obj predictions
            Check later
            """
            _, pair_pred = _scores.topk(topk, dim=1)  # sort returns indices of k largest values
            pair_pred = pair_pred.contiguous().view(-1)
            attr_pred, obj_pred = self.all_pair_attr_obj_id[pair_pred][:, 0].view(-1, topk), \
                                  self.all_pair_attr_obj_id[pair_pred][:, 1].view(-1, topk)
            return (attr_pred, obj_pred)

        results = {}
        orig_scores = scores.clone()
        mask = self.seen_mask.repeat(
            scores.shape[0], 1
        )  # Repeat mask along pairs dimension
        scores[~mask] += bias  # Add bias to test pairs

        # Unbiased setting

        # Open world setting --no mask, all pairs of the dataset
        results.update({"open": get_pred_from_scores(scores, topk)})
        results.update({"unbiased_open": get_pred_from_scores(orig_scores, topk)})
        # Closed world setting - set the score for all Non test pairs to -1e10,
        # this excludes the pairs from set not in evaluation
        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10
        closed_orig_scores = orig_scores.clone()
        closed_orig_scores[~mask] = -1e10
        results.update({"closed": get_pred_from_scores(closed_scores, topk)})
        results.update(
            {"unbiased_closed": get_pred_from_scores(closed_orig_scores, topk)}
        )

        return results

    def score_clf_model(self, scores, obj_truth, topk=1):
        '''
        Wrapper function to call generate_predictions for CLF models
        '''
        attr_pred, obj_pred = scores

        # Go to CPU
        attr_pred, obj_pred, obj_truth = attr_pred.to(
            'cpu'), obj_pred.to('cpu'), obj_truth.to('cpu')

        # Gather scores (P(a), P(o)) for all relevant (a,o) pairs
        # Multiply P(a) * P(o) to get P(pair)
        # Return only attributes that are in our pairs
        attr_subset = attr_pred.index_select(1, self.all_pair_attr_obj_id[:, 0])
        obj_subset = obj_pred.index_select(1, self.all_pair_attr_obj_id[:, 1])
        scores = (attr_subset * obj_subset)  # (Batch, #pairs)

        results = self.generate_predictions(scores, obj_truth)
        results['biased_scores'] = scores

        return results

    def score_manifold_model(self, scores, obj_truth, bias=0.0, topk=1):
        '''
        Wrapper function to call generate_predictions for manifold models
        '''
        # Go to CPU
        scores = {k: v.to('cpu') for k, v in scores.items()}
        obj_truth = obj_truth.to(device)

        # Gather scores for all relevant (a,o) pairs
        scores = torch.stack(
            [scores[(attr, obj)] for attr, obj in self.dset.all_pairs], 1
        )  # (Batch, #pairs)
        orig_scores = scores.clone()
        results = self.generate_predictions(scores, obj_truth, bias, topk)
        results['scores'] = orig_scores
        return results

    def score_fast_model(self, scores, obj_truth, bias=0.0, topk=1):
        '''
        Wrapper function to call generate_predictions for manifold models
        '''

        results = {}
        # Repeat mask along pairs dimension
        mask = self.seen_mask.repeat(scores.shape[0], 1)
        scores[~mask] += bias  # Add bias to test pairs

        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10

        # sort returns indices of k largest values
        _, pair_pred = closed_scores.topk(topk, dim=1)
        # _, pair_pred = scores.topk(topk, dim=1)  # sort returns indices of k
        # largest values
        pair_pred = pair_pred.contiguous().view(-1)
        attr_pred, obj_pred = self.all_pair_attr_obj_id[pair_pred][:, 0].view(-1, topk), \
                              self.all_pair_attr_obj_id[pair_pred][:, 1].view(-1, topk)

        results.update({'closed': (attr_pred, obj_pred)})
        return results

    def evaluate_predictions(
            self,
            predictions,
            attr_truth,
            obj_truth,
            pair_truth,
            allpred,
            topk=1):
        # Go to CPU
        attr_truth, obj_truth, pair_truth = (
            attr_truth.to("cpu"),
            obj_truth.to("cpu"),
            pair_truth.to("cpu"),
        )

        pairs = list(zip(list(attr_truth.numpy()), list(obj_truth.numpy())))

        seen_ind, unseen_ind = [], []
        for i in range(len(attr_truth)):
            if pairs[i] in self.train_pairs:
                seen_ind.append(i)
            else:
                unseen_ind.append(i)

        seen_ind, unseen_ind = torch.LongTensor(seen_ind), torch.LongTensor(
            unseen_ind
        )

        def _process(_scores):
            # Top k pair accuracy
            # Attribute, object and pair
            attr_match = (
                attr_truth.unsqueeze(1).repeat(1, topk) == _scores[0][:, :topk]
            )
            obj_match = (
                obj_truth.unsqueeze(1).repeat(1, topk) == _scores[1][:, :topk]
            )

            # Match of object pair
            match = (attr_match * obj_match).any(1).float()
            attr_match = attr_match.any(1).float()
            obj_match = obj_match.any(1).float()
            # Match of seen and unseen pairs
            seen_match = match[seen_ind]
            unseen_match = match[unseen_ind]
            # Calculating class average accuracy

            seen_score, unseen_score = torch.ones(512, 5), torch.ones(512, 5)

            return attr_match, obj_match, match, seen_match, unseen_match, torch.Tensor(
                seen_score + unseen_score), torch.Tensor(seen_score), torch.Tensor(unseen_score)

        def _add_to_dict(_scores, type_name, stats):
            base = [
                "_attr_match",
                "_obj_match",
                "_match",
                "_seen_match",
                "_unseen_match",
                "_ca",
                "_seen_ca",
                "_unseen_ca",
            ]
            for val, name in zip(_scores, base):
                stats[type_name + name] = val

        stats = dict()

        # Closed world
        closed_scores = _process(predictions["closed"])
        unbiased_closed = _process(predictions["unbiased_closed"])
        _add_to_dict(closed_scores, "closed", stats)
        _add_to_dict(unbiased_closed, "closed_ub", stats)

        # Calculating AUC
        scores = predictions["scores"]
        # getting score for each ground truth class
        correct_scores = scores[torch.arange(scores.shape[0]), pair_truth][
            unseen_ind
        ]

        # Getting top predicted score for these unseen classes
        max_seen_scores = predictions['scores'][unseen_ind][:, self.seen_mask].topk(topk, dim=1)[
            0][:, topk - 1]

        # Getting difference between these scores
        unseen_score_diff = max_seen_scores - correct_scores

        # Getting matched classes at max bias for diff
        unseen_matches = stats["closed_unseen_match"].bool()
        correct_unseen_score_diff = unseen_score_diff[unseen_matches] - 1e-4

        # sorting these diffs
        correct_unseen_score_diff = torch.sort(correct_unseen_score_diff)[0]
        magic_binsize = 20
        # getting step size for these bias values
        bias_skip = max(len(correct_unseen_score_diff) // magic_binsize, 1)
        # Getting list
        biaslist = correct_unseen_score_diff[::bias_skip]

        seen_match_max = float(stats["closed_seen_match"].mean())
        unseen_match_max = float(stats["closed_unseen_match"].mean())
        seen_accuracy, unseen_accuracy = [], []

        # Go to CPU
        base_scores = {k: v.to("cpu") for k, v in allpred.items()}
        obj_truth = obj_truth.to("cpu")

        # Gather scores for all relevant (a,o) pairs
        base_scores = torch.stack(
            [allpred[(attr, obj)] for attr, obj in self.dset.all_pairs], 1
        )  # (Batch, #pairs)

        for bias in biaslist:
            scores = base_scores.clone()
            results = self.score_fast_model(
                scores, obj_truth, bias=bias, topk=topk)
            results = results['closed']  # we only need biased
            results = _process(results)
            seen_match = float(results[3].mean())
            unseen_match = float(results[4].mean())
            seen_accuracy.append(seen_match)
            unseen_accuracy.append(unseen_match)

        seen_accuracy.append(seen_match_max)
        unseen_accuracy.append(unseen_match_max)
        seen_accuracy, unseen_accuracy = np.array(seen_accuracy), np.array(unseen_accuracy)
        area = np.trapz(seen_accuracy, unseen_accuracy)

        for key in stats:
            stats[key] = float(stats[key].mean())

        try:
            harmonic_mean = hmean([seen_accuracy, unseen_accuracy], axis=0)
        except BaseException:
            harmonic_mean = 0

        max_hm = np.max(harmonic_mean)
        idx = np.argmax(harmonic_mean)
        if idx == len(biaslist):
            bias_term = 1e3
        else:
            bias_term = biaslist[idx]
        stats["biasterm"] = float(bias_term)
        stats["best_unseen"] = np.max(unseen_accuracy)
        stats["best_seen"] = np.max(seen_accuracy)
        stats["AUC"] = area
        stats["hm_unseen"] = unseen_accuracy[idx]
        stats["hm_seen"] = seen_accuracy[idx]
        stats["best_hm"] = max_hm
        return stats


def compute_representations(model, test_dataset, config, device):
    """Function computes the attribute-object representations using
    the text encoder.
    Args:
        model (nn.Module): model
        test_dataset (CompositionDataset): CompositionDataset object
            with phase = 'test'
        config (argparse.ArgumentParser): config/args
        device (str): device type cpu/cuda:0
    Returns:
        torch.Tensor: returns the tensor with the attribute-object
            representations
    """
    obj2idx = test_dataset.obj2idx
    attr2idx = test_dataset.attr2idx

    attr_idx = torch.tensor(list(range(len(attr2idx)))).to(device)
    obj_idx = torch.tensor(list(range(len(obj2idx)))).to(device)

    pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                          for attr, obj in test_dataset.all_pairs]).to(device)

    batch_test_pairs = np.array_split(pairs, len(pairs) // config.text_encoder_batch_size)
    if len(attr_idx) // config.text_encoder_batch_size > 0:
        test_attrs = np.array_split(attr_idx, len(attr_idx) // config.text_encoder_batch_size)
    else:
        test_attrs = attr_idx
    if len(obj_idx) // config.text_encoder_batch_size > 0:
        test_objs = np.array_split(obj_idx, len(obj_idx) // config.text_encoder_batch_size)
    else:
        test_objs = obj_idx

    model.eval()


    if config.experiment_name in ["clip", "csp", 'pair_soft_emb_and_soft_prompt', 'graph_prompt']:
        rep = torch.Tensor().to(device).type(model.dtype)
        with torch.no_grad():
            for batch_pair_attr_obj_sep_idx in tqdm(batch_test_pairs):
                batch_pair_attr_obj_sep_idx = batch_pair_attr_obj_sep_idx.to(device)
                token_tensors = model.construct_pair_txt_emb(batch_pair_attr_obj_sep_idx)
                text_features = model.text_encoder(
                    model.template_token_id,
                    token_tensors,
                    enable_pos_emb=model.enable_pos_emb,
                )

                text_features = text_features / text_features.norm(
                    dim=-1, keepdim=True
                )

                rep = torch.cat([rep, text_features], dim=0)

        return rep

    elif config.experiment_name in ["sep_soft_emb", 'sep_soft_prompt', 'shared_soft_prompt']:
        pair_rep = torch.Tensor().to(device).type(model.dtype)
        attr_rep = torch.Tensor().to(device).type(model.dtype)
        obj_rep = torch.Tensor().to(device).type(model.dtype)

        with torch.no_grad():
            for batch_pair_attr_obj_sep_idx in tqdm(batch_test_pairs):     # uz-zappos: 116 = 39 + 39 + 38
                batch_pair_attr_obj_sep_idx = batch_pair_attr_obj_sep_idx.to(device)
                pair_emb = model.construct_pair_txt_emb(batch_pair_attr_obj_sep_idx)
                pair_feats = model.text_encoder(
                    model.template_token_id,
                    pair_emb,
                    enable_pos_emb=model.enable_pos_emb,
                )
                _pair_features = pair_feats
                norm_pair_feat = _pair_features / _pair_features.norm(dim=-1, keepdim=True)
                pair_rep = torch.cat([pair_rep, norm_pair_feat], dim=0)

            if isinstance(test_attrs, list):
                for batch_attr in tqdm(test_attrs):
                    batch_attr = batch_attr.to(device)
                    attr_emb = model.construct_attr_txt_emb(batch_attr)
                    attr_feats = model.text_encoder(
                        model.attr_txt_token_id,
                        attr_emb,
                        enable_pos_emb=model.enable_pos_emb,
                    )
                    _attr_features = attr_feats
                    norm_attr_feat = _attr_features / _attr_features.norm(dim=-1, keepdim=True)
                    attr_rep = torch.cat([attr_rep, norm_attr_feat], dim=0)
            else:
                attr_emb = model.construct_attr_txt_emb(attr_idx)
                attr_feats = model.text_encoder(
                    model.attr_txt_token_id,
                    attr_emb,
                    enable_pos_emb=model.enable_pos_emb,
                )
                _attr_features = attr_feats
                attr_rep = _attr_features / _attr_features.norm(dim=-1, keepdim=True)

            if isinstance(test_objs, list):
                for batch_obj in tqdm(test_objs):
                    batch_obj = batch_obj.to(device)
                    obj_emb = model.construct_obj_txt_emb(batch_obj)
                    obj_feats = model.text_encoder(
                        model.obj_txt_token_id,
                        obj_emb,
                        enable_pos_emb=model.enable_pos_emb,
                    )
                    _obj_features = obj_feats
                    norm_obj_feat = _obj_features / _obj_features.norm(dim=-1, keepdim=True)
                    obj_rep = torch.cat([obj_rep, norm_obj_feat], dim=0)
            else:
                obj_emb = model.construct_obj_txt_emb(obj_idx)
                obj_feats = model.text_encoder(
                    model.obj_txt_token_id,
                    obj_emb,
                    enable_pos_emb=model.enable_pos_emb,
                )
                _obj_features = obj_feats
                obj_rep = _obj_features / _obj_features.norm(dim=-1, keepdim=True)
        return pair_rep, attr_rep, obj_rep


def clip_baseline(model, test_dataset, config, device):
    """Function to get the clip representations.

    Args:
        model (nn.Module): the clip model
        test_dataset (CompositionDataset): the test/validation dataset
        config (argparse.ArgumentParser): config/args
        device (str): device type cpu/cuda:0

    Returns:
        torch.Tensor: returns the tensor with the attribute-object
            representations with clip model.
    """
    pairs = test_dataset.all_pairs
    pairs = [(attr.replace(".", " ").lower(),
              obj.replace(".", " ").lower())
             for attr, obj in pairs]

    prompts = [f"a photo of {attr} {obj}" for attr, obj in pairs]
    tokenized_prompts = clip.tokenize(
        prompts, context_length=config.context_length)
    test_batch_tokens = np.array_split(
        tokenized_prompts,
        len(tokenized_prompts) //
        config.text_encoder_batch_size)
    rep = torch.Tensor().to(device).type(model.dtype)
    with torch.no_grad():
        for batch_tokens in test_batch_tokens:
            batch_tokens = batch_tokens.to(device)
            _text_features = model.text_encoder(
                batch_tokens, enable_pos_emb=True)
            text_features = _text_features / _text_features.norm(
                dim=-1, keepdim=True
            )
            rep = torch.cat((rep, text_features), dim=0)

    return rep


def predict_logits(model, pair_rep, dataset, device, config,
                   attr_rep = None, obj_rep = None, pair_weight = 1.0, attr_weight = 1.0, obj_weight = 1.0):
    """Function to predict the cosine similarities between the
    images and the attribute-object representations. The function
    also returns the ground truth for attributes, objects, and pair
    of attribute-objects.

    Args:
        model (nn.Module): the model
        pair_rep (nn.Tensor): the attribute-object representations.
        dataset (CompositionDataset): the composition dataset (validation/test)
        device (str): the device (either cpu/cuda:0)
        config (argparse.ArgumentParser): config/args

    Returns:
        tuple: the logits, attribute labels, object labels,
            pair attribute-object labels
    """
    model.eval()

    attr2idx = dataset.attr2idx
    obj2idx = dataset.obj2idx
    pair_sep_idx = torch.tensor([(attr2idx[attr], obj2idx[obj]) for attr, obj in dataset.all_pairs]).to(device)
    attr_index, obj_index = pair_sep_idx[:, 0], pair_sep_idx[:, 1]


    all_attr_gt, all_obj_gt, all_pair_gt = (
        [],
        [],
        [],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False)
    all_pair_logits = torch.Tensor()
    with torch.no_grad():
        for idx, data in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Testing"
        ):
            batch_img = data[0].to(device)
            batch_img_feat = model.encode_image(batch_img)
            normalized_img = batch_img_feat / batch_img_feat.norm(
                dim=-1, keepdim=True
            )

            pair_logits = (
                    model.clip_model.logit_scale.exp()
                    * normalized_img
                    @ pair_rep.t()
            )

            if attr_rep is not None:
                attr_logits = (
                        model.clip_model.logit_scale.exp()
                        * normalized_img
                        @ attr_rep.t()
                )
                attr_logits = attr_logits.cpu()

            if obj_rep is not None:
                obj_logits = (
                        model.clip_model.logit_scale.exp()
                        * normalized_img
                        @ obj_rep.t()
                )
                obj_logits = obj_logits.cpu()



            attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]

            pair_logits = pair_logits.cpu() * pair_weight
            if attr_rep is not None:
                pair_attr_logits = attr_logits[:, attr_index]
                pair_logits += pair_attr_logits * attr_weight

            if obj_rep is not None:
                pair_obj_logits = obj_logits[:, obj_index]
                pair_logits += pair_obj_logits * obj_weight

            all_pair_logits = torch.cat([all_pair_logits, pair_logits], dim=0)

            all_attr_gt.append(attr_truth)
            all_obj_gt.append(obj_truth)
            all_pair_gt.append(pair_truth)

    all_attr_gt, all_obj_gt, all_pair_gt = (
        torch.cat(all_attr_gt).to("cpu"),
        torch.cat(all_obj_gt).to("cpu"),
        torch.cat(all_pair_gt).to("cpu"),
    )

    return all_pair_logits, all_attr_gt, all_obj_gt, all_pair_gt


def threshold_with_feasibility(
        logits,
        seen_mask,
        threshold=None,
        feasiblity=None):
    """Function to remove infeasible compositions.

    Args:
        logits (torch.Tensor): the cosine similarities between
            the images and the attribute-object pairs.
        seen_mask (torch.tensor): the seen mask with binary
        threshold (float, optional): the threshold value.
            Defaults to None.
        feasiblity (torch.Tensor, optional): the feasibility.
            Defaults to None.

    Returns:
        torch.Tensor: the logits after filtering out the
            infeasible compositions.
    """
    score = copy.deepcopy(logits)
    # Note: Pairs are already aligned here
    mask = (feasiblity >= threshold).float()
    # score = score*mask + (1.-mask)*(-1.)
    score = score * (mask + seen_mask)

    return score


def test(
        test_dataset,
        evaluator,
        all_logits,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        config):
    """Function computes accuracy on the validation and
    test dataset.

    Args:
        test_dataset (CompositionDataset): the validation/test
            dataset
        evaluator (Evaluator): the evaluator object
        all_logits (torch.Tensor): the cosine similarities between
            the images and the attribute-object pairs.
        all_attr_gt (torch.tensor): the attribute ground truth
        all_obj_gt (torch.tensor): the object ground truth
        all_pair_gt (torch.tensor): the attribute-object pair ground
            truth
        config (argparse.ArgumentParser): the config

    Returns:
        dict: the result with all the metrics
    """
    predictions = {
        pair_name: all_logits[:, i]
        for i, pair_name in enumerate(test_dataset.all_pairs)
    }
    all_pred = [predictions]

    all_pred_dict = {}
    for k in all_pred[0].keys():
        all_pred_dict[k] = torch.cat(
            [all_pred[i][k] for i in range(len(all_pred))]
        ).float()

    results = evaluator.score_model(
        all_pred_dict, all_obj_gt, bias=config.bias, topk=config.topk
    )


    """
    results['open'][0][10]      # 18 460
    results['open'][1][10]

    test_dataset.attr2idx
    test_dataset.obj2idx['rubber']
    test_dataset.attrs
    test_dataset.objs
    test_dataset.test_data
    """

    # if config.save_for_analyze:
    phase = test_dataset.phase
    if phase == "val":
        item_list = test_dataset.val_data
    elif phase == "test":
        item_list = test_dataset.test_data
    img_id_list = [item[0] for item in item_list]

    dict_Save = {
        "raw_score": results["scores"],        # item_num * pair_num
        "item_list": item_list,
        "attr_list": test_dataset.attrs,
        "obj_list":  test_dataset.objs,
        "all_pair_list": test_dataset.all_pairs,
        "attr2idx": test_dataset.attr2idx,
        "obj2idx": test_dataset.obj2idx,
        "allPair2idx": test_dataset.allPair2idx,
        "train_pairs": test_dataset.train_pairs,
        "val_pairs": test_dataset.val_pairs,
        "test_pairs": test_dataset.test_pairs,
        "img_id_list": img_id_list
    }
    import pickle
    pkl_file_name = f"{phase}.pkl"
    with open(os.path.join(config.save_path, pkl_file_name), 'wb') as handle:
        pickle.dump(dict_Save, handle, protocol=pickle.HIGHEST_PROTOCOL)


    attr_acc = float(torch.mean(
        (results['unbiased_closed'][0].squeeze(-1) == all_attr_gt).float()))
    obj_acc = float(torch.mean(
        (results['unbiased_closed'][1].squeeze(-1) == all_obj_gt).float()))

    stats = evaluator.evaluate_predictions(
        results,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        all_pred_dict,
        topk=config.topk,
    )

    stats['attr_acc'] = attr_acc
    stats['obj_acc'] = obj_acc

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="name of the dataset", type=str, default="ut-zappos")
    parser.add_argument(
        "--lr", help="learning rate", type=float    )
    parser.add_argument(
        "--weight_decay", help="weight decay", type=float
    )
    parser.add_argument(
        "--clip_model", help="clip model type", type=str, default="ViT-B/16"
    )
    parser.add_argument(
        "--eval_batch_size", help="eval batch size", default=64, type=int
    )
    parser.add_argument(
        "--experiment_name",
        help="name of the experiment",
        type=str,
        default='csp'
    )
    parser.add_argument(
        "--evaluate_only",
        help="directly evaluate on the" "dataset without any training",
        action="store_true",
    )
    parser.add_argument(
        "--context_length",
        help="sets the context length of the clip model",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--soft_emb_dropout",
        help="add dropout to attributes",
        type=float,
    )

    parser.add_argument(
        "--open_world",
        help="evaluate on open world setup",
        action="store_true",
    )
    parser.add_argument(
        "--open_model",
        help="using open world graph",
        action="store_true",
    )

    parser.add_argument(
        "--bias",
        help="eval bias",
        type=float,
        default=1e3,
    )
    parser.add_argument(
        "--topk",
        help="eval topk",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--save_for_analyze",
        type=int,
        default=0
    )

    parser.add_argument(
        "--text_encoder_batch_size",
        help="batch size of the text encoder",
        default=16,
        type=int,
    )
    parser.add_argument(
        '--threshold',
        type=float,
        help="optional threshold"
    )
    parser.add_argument(
        '--threshold_trials',
        type=int,
        default=50,
        help="how many threshold values to try"
    )
    parser.add_argument(
        '--use_element',
        action='store_true'
    )
    parser.add_argument(
        '--epoch_num',
        type=int,
        # required=True
    )
    parser.add_argument(
        '--select_epoch_num',
        type=int,
        required=True
    )
    parser.add_argument("--attr_weight", default=1.0, type=float)
    parser.add_argument("--obj_weight", default=1.0, type=float)

    parser.add_argument(
        "--graph_gr_emb",
        type=str,
        default='d4096,d'
    )
    parser.add_argument(
        "--prompt_len",
        type=int,
        required=True

    )
    parser.add_argument(
        "--prompt_position",
        type=str,
        required=True
    )
    parser.add_argument(
        "--using_APhotoOf",
        type=str,
        default='True'
    )
    config = parser.parse_args()

    # load yaml based on args
    if config.dataset.startswith('mit'):
        yaml_file = "/localscratch2/xugy/VLPrompt/Cfg/common/mit.yml"
    if config.dataset.startswith('ut'):
        yaml_file = "/localscratch2/xugy/VLPrompt/Cfg/common/utzappos.yml"
    if config.dataset.startswith('cgqa'):
        yaml_file = "/localscratch2/xugy/VLPrompt/Cfg/common/cgqa.yml"
    load_args(yaml_file, config, suffix="")

    # add graph things
    if config.experiment_name.startswith("graph"):
        if config.dataset.startswith('mit'):
            config.graph_config_file = "/localscratch2/xugy/VLPrompt/Cfg/graph_prompt_embed/mit.yml"
        elif config.dataset.startswith('ut'):
            config.graph_config_file = "/localscratch2/xugy/VLPrompt/Cfg/graph_prompt_embed/utzappos.yml"
        elif config.dataset.startswith('cgqa'):
            config.graph_config_file = "/localscratch2/xugy/VLPrompt/Cfg/graph_prompt_embed/cgqa.yml"
        from ProjUtils.misc import load_args
        load_args(config.graph_config_file, config)


    # add saved dir
    modify_save_path(config)
    config.dataset_dir = DATASET_PATHS[config.dataset]

    # modify the trained value
    if config.dataset == "ut-zappos":
        if config.experiment_name == "clip":
            config.soft_embeddings = "/tank/space/xugy07/VLPrompt/RefCode/csp/ckpt/soft_embeddings_epoch_18.pt"
        elif config.experiment_name == 'sep_soft_emb':
            config.soft_embeddings = "/tank/space/xugy07/VLPrompt/ckpt/ut-zappos/sep_space_soft_emb/soft_embeddings_epoch_21.pt"
        elif config.experiment_name == "csp":
            config.soft_embeddings = "/tank/space/xugy07/VLPrompt/ckpt/ut-zappos/csp/ViTL14/soft_embeddings_epoch_20.pt"
        elif config.experiment_name in ['pair_soft_emb_and_soft_prompt', 'graph_prompt']:
            config.soft_embeddings = f'{config.save_path}/soft_embeddings_epoch_{config.select_epoch_num}.pt'
            # config.soft_embeddings = '/tank/space/xugy07/VLPrompt/ckpt/ut-zappos/pair_soft_emb_and_soft_prompt/ViT-L/14/dropout_0.3/soft_embeddings_epoch_19.pt'
        elif config.experiment_name == 'sep_soft_prompt':
            config.soft_embeddings = '/tank/space/xugy07/VLPrompt/ckpt/ut-zappos/sep_soft_prompt/ViT-L/14/soft_embeddings_epoch_20.pt'
        elif config.experiment_name == 'shared_soft_prompt':
            config.soft_embeddings = '/tank/space/xugy07/VLPrompt/ckpt/ut-zappos/shared_soft_prompt/ViT-L/14/Attr_1.0_Obj_1.0/soft_embeddings_epoch_20.pt'
    elif config.dataset == 'mit-states':
        if config.experiment_name in ['pair_soft_emb_and_soft_prompt', 'graph_prompt']:
            config.soft_embeddings = f'{config.save_path}/soft_embeddings_epoch_{config.select_epoch_num}.pt'
            # config.soft_embeddings = '/tank/space/xugy07/VLPrompt/ckpt/mit-states/pair_soft_emb_and_soft_prompt/soft_embeddings_epoch_20.pt'
        if config.experiment_name == 'csp':
            config.soft_embeddings = '/tank/space/xugy07/VLPrompt/ckpt/mit-states/csp/soft_embeddings_epoch_20.pt'
    elif config.dataset == 'cgqa':
        if config.experiment_name in ["pair_soft_emb_and_soft_prompt", "graph_prompt"]:
            config.soft_embeddings = f'{config.save_path}/soft_embeddings_epoch_{config.select_epoch_num}.pt'

    # set the seed value
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("evaluation details")
    print("----")
    print(f"dataset: {config.dataset}")
    print(f"experiment name: {config.experiment_name}")

    if config.experiment_name != 'clip':
        if not os.path.exists(config.soft_embeddings):
            print(f'{config.soft_embeddings} not found')
            print('code exiting!')
            exit(0)

    dataset_path = DATASET_PATHS[config.dataset]

    print('loading validation dataset')
    val_dataset = CompositionDataset(dataset_path,
                                     phase='val',
                                     split='compositional-split-natural',
                                     open_world=config.open_world)

    print('loading test dataset')
    test_dataset = CompositionDataset(dataset_path,
                                      phase='test',
                                      split='compositional-split-natural',
                                      open_world=config.open_world)


    # get the model and the text rep
    if config.experiment_name == 'clip':
        clip_model, preprocess = load(
            config.clip_model, device=device, context_length=config.context_length)

        model = CLIPInterface(
            clip_model,
            config,
            template_token_id=None,
            device=device,
            enable_pos_emb=True)
        val_text_rep = clip_baseline(model, val_dataset, config, device)
        test_text_rep = clip_baseline(model, test_dataset, config, device)

    elif config.experiment_name in ['sep_soft_emb']:
        model, optimizer = get_model(val_dataset, config, device)

        soft_embs = torch.load(config.soft_embeddings)['soft_embed']
        model.set_soft_embeddings(soft_embs)
        val_pair_rep, val_attr_rep, val_obj_rep = compute_representations(model, val_dataset, config, device)
        test_pair_rep, test_attr_rep, test_obj_rep = compute_representations(model, test_dataset, config, device)


    elif config.experiment_name in ['csp']:
        model, optimizer = get_model(val_dataset, config, device)

        soft_embs = torch.load(config.soft_embeddings)['soft_embed']
        model.set_soft_embeddings(soft_embs)
        val_pair_rep = compute_representations(model, val_dataset, config, device)
        test_pair_rep = compute_representations(model, test_dataset, config, device)


    elif config.experiment_name in ['sep_soft_prompt']:
        model, optimizer = get_model(val_dataset, config, device)

        pair_soft_emb = torch.load(config.soft_embeddings)['pair_soft_prompt']
        attr_soft_emb = torch.load(config.soft_embeddings)['attr_soft_prompt']
        obj_soft_emb = torch.load(config.soft_embeddings)['obj_soft_prompt']

        model.set_soft_embeddings(pair_soft_emb, attr_soft_emb, obj_soft_emb)

        val_pair_rep, val_attr_rep, val_obj_rep = compute_representations(model, val_dataset, config, device)
        test_pair_rep, test_attr_rep, test_obj_rep = compute_representations(model, test_dataset, config, device)


    elif config.experiment_name in ['pair_soft_emb_and_soft_prompt']:
        model, optimizer = get_model(val_dataset, config, device)

        soft_embs = torch.load(config.soft_embeddings)['soft_embed']
        soft_prompt = torch.load(config.soft_embeddings)['soft_prompt']

        model.set_soft_embeddings(soft_embs, soft_prompt)

        val_pair_rep = compute_representations(model, val_dataset, config, device)
        test_pair_rep = compute_representations(model, test_dataset, config, device)

    elif config.experiment_name in ['graph_prompt']:
        model, optimizer = get_model(val_dataset, config, device)

        # -----------------
        # 'embedding' = {Tensor: 144} tensor([[ 0.0109, -0.0066, -0.0030,
        # 'soft_prompt' = {Parameter: 3} Parameter containing:\ntensor([[-0.0093,  0.0148,
        # 'soft_embed' = {Tensor: 28} tensor([[ 0.0109, -0.0066, -0.0030,
        # 'gcn' = {OrderedDict: 4} OrderedDict([('gcn.conv1.layer.weight', tensor([[-2.1285e-02,  2.2323e-02
        # -----------------
        soft_embs = torch.load(config.soft_embeddings)['soft_embed']
        soft_prompt = torch.load(config.soft_embeddings)['soft_prompt']
        gcn = torch.load(config.soft_embeddings)['gcn']

        model.set_soft_embeddings(soft_embs, soft_prompt)
        model.set_gcn(gcn)

        val_pair_rep = compute_representations(model, val_dataset, config, device)
        test_pair_rep = compute_representations(model, test_dataset, config, device)


    elif config.experiment_name in ['shared_soft_prompt']:
        model, optimizer = get_model(val_dataset, config, device)

        shared_soft_prompt = torch.load(config.soft_embeddings)['soft_prompt']

        model.set_soft_embeddings(shared_soft_prompt)

        val_pair_rep, val_attr_rep, val_obj_rep = compute_representations(model, val_dataset, config, device)
        test_pair_rep, test_attr_rep, test_obj_rep = compute_representations(model, test_dataset, config, device)


    print('evaluating on the validation set')
    if config.open_world and config.threshold is None:
        evaluator = Evaluator(val_dataset, model=None)
        if config.dataset == 'mit-states':
            feasibility_path = os.path.join(f'{config.dataset_dir}/feasibility_{config.dataset}.pt')
        elif config.dataset == 'ut-zappos':
            feasibility_path = os.path.join(f'{config.dataset_dir}/feasibility_{config.dataset}.pt')
        elif config.dataset == 'cgqa':
            feasibility_path = os.path.join(f'{config.dataset_dir}/feasibility_{config.dataset}.pt')
        unseen_scores = torch.load(feasibility_path, map_location='cpu')['feasibility']

        seen_mask = val_dataset.seen_mask.to('cpu')
        min_feasibility = (unseen_scores + seen_mask * 10.).min()
        max_feasibility = (unseen_scores - seen_mask * 10.).max()
        threshol_list = np.linspace(
            min_feasibility,
            max_feasibility,
            num=config.threshold_trials)
        best_auc = 0.
        best_th = -10
        val_stats = None
        with torch.no_grad():
            all_logits, all_attr_gt, all_obj_gt, all_pair_gt = predict_logits(
                model,
                val_pair_rep,
                val_dataset,
                device,
                config)
            for th in threshol_list:
                temp_logits = threshold_with_feasibility(all_logits, val_dataset.seen_mask, threshold=th, feasiblity=unseen_scores)
                results = test(
                    val_dataset,
                    evaluator,
                    temp_logits,
                    all_attr_gt,
                    all_obj_gt,
                    all_pair_gt,
                    config
                )
                auc = results['AUC']
                if auc > best_auc:
                    best_auc = auc
                    best_th = th
                    print('New best AUC', best_auc)
                    print('Threshold', best_th)
                    val_stats = copy.deepcopy(results)

    else:
        best_th = config.threshold
        evaluator = Evaluator(val_dataset, model=None)
        # feasibility_path = os.path.join(f"/egr/research-hlr/xugy07/{dir_Map[config.dataset]}", f'feasibility_{config.dataset}.pt')
        feasibility_path = os.path.join(f"{config.dataset_dir}", f'feasibility_{config.dataset}.pt')
        unseen_scores = torch.load(feasibility_path, map_location='cpu')['feasibility']

        # --------------------------------------------------
        # diff: whether to pass the attr_rep and obj_rep
        # --------------------------------------------------
        with torch.no_grad():
            if config.experiment_name in ["csp", "clip", "pair_soft_emb_and_soft_prompt", 'graph_prompt']:
                all_logits, all_attr_gt, all_obj_gt, all_pair_gt = predict_logits(model,
                                                                                  val_pair_rep,
                                                                                  val_dataset,
                                                                                  device,
                                                                                  config)

                if config.open_world:
                    print('using threshold: ', best_th)
                    all_logits = threshold_with_feasibility(all_logits, val_dataset.seen_mask, threshold=best_th, feasiblity=unseen_scores)

            elif config.experiment_name in ["sep_soft_emb", "sep_soft_prompt", 'shared_soft_prompt', 'graph_prompt']:
                all_logits, all_attr_gt, all_obj_gt, all_pair_gt = predict_logits(model,
                                                                                  val_pair_rep,
                                                                                  val_dataset,
                                                                                  device,
                                                                                  config,
                                                                                  val_attr_rep,
                                                                                  val_obj_rep)
                if config.open_world:
                    print('using threshold: ', best_th)
                    all_logits = threshold_with_feasibility(all_logits, val_dataset.seen_mask, threshold=best_th, feasiblity=unseen_scores)

            results = test(
                val_dataset,
                evaluator,
                all_logits,
                all_attr_gt,
                all_obj_gt,
                all_pair_gt,
                config
            )
        val_stats = copy.deepcopy(results)
        result = ""
        for key in val_stats:
            result = result + key + "  " + str(round(val_stats[key], 4)) + "| "
        print(result)


    print('evaluating on the test set')
    with torch.no_grad():

        # unseen_scores = torch.load(feasibility_path, map_location='cpu')['feasibility']
        # best_th = config.threshold
        evaluator = Evaluator(test_dataset, model=None)

        if config.experiment_name in ["csp", "clip", "pair_soft_emb_and_soft_prompt", "graph_prompt"]:
            all_logits, all_attr_gt, all_obj_gt, all_pair_gt = predict_logits(model,
                                                                              test_pair_rep,
                                                                              test_dataset,
                                                                              device,
                                                                              config)
            if config.open_world and best_th is not None:
                print('using threshold: ', best_th)
                all_logits = threshold_with_feasibility(
                    all_logits,
                    test_dataset.seen_mask,
                    threshold=best_th,
                    feasiblity=unseen_scores)

            test_stats = test(
                test_dataset,
                evaluator,
                all_logits,
                all_attr_gt,
                all_obj_gt,
                all_pair_gt,
                config
            )

            result = ""
            for key in test_stats:
                result = result + key + "  " + \
                         str(round(test_stats[key], 4)) + "| "
            print(result)

        elif config.experiment_name in ["sep_soft_emb", "sep_soft_prompt", 'shared_soft_prompt']:
            for weight in weight_grid:
                pair_weight, attr_weight, obj_weight = weight
                all_logits, all_attr_gt, all_obj_gt, all_pair_gt = predict_logits(model,
                                                                                  test_pair_rep,
                                                                                  test_dataset,
                                                                                  device,
                                                                                  config,
                                                                                  test_attr_rep,
                                                                                  test_obj_rep,
                                                                                  pair_weight=pair_weight,
                                                                                  attr_weight=attr_weight,
                                                                                  obj_weight=obj_weight)
                if config.open_world and best_th is not None:
                    print('using threshold: ', best_th)
                    all_logits = threshold_with_feasibility(
                        all_logits,
                        test_dataset.seen_mask,
                        threshold=best_th,
                        feasiblity=unseen_scores)
                test_stats = test(
                    test_dataset,
                    evaluator,
                    all_logits,
                    all_attr_gt,
                    all_obj_gt,
                    all_pair_gt,
                    config
                )

                result = ""
                print("---------",   weight,  "---------")
                for key in test_stats:
                    result = result + key + "  " + \
                        str(round(test_stats[key], 4)) + "| "
                print(result)

    """
    results = {
        'val': val_stats,
        'test': test_stats,
    }
    """

    if best_th is not None:
        results['best_threshold'] = best_th

    if config.experiment_name != 'clip':
        if config.open_world:
            result_path = config.soft_embeddings[:-2] + "open.calibrated.json"
        else:
            result_path = config.soft_embeddings[:-2] + "closed.json"

        with open(result_path, 'w+') as fp:
            json.dump(results, fp)

    print("done!")
