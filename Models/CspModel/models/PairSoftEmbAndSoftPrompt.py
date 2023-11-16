"""
1. soft_emb
2. soft_prompt
combine these two
"""


import os

import clip
import pandas as pd
import torch
import torch.nn as nn
from Models.CspModel.clip_modules.interface import CLIPInterface
from Models.CspModel.clip_modules.model_loader import load

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class PairSoftEmbAndSoftInterface(CLIPInterface):
    """ soft two things """
    def __init__(
        self,
        clip_model,
        config,
        offset,
        soft_element_embeddings,
        soft_prompt_emebddings,
        pair_txt_token_ids,
        device="cuda",
        enable_pos_emb=True,
        soft_emb_dropout=0.0,
    ):
        super().__init__(
            clip_model,
            config,
            pair_txt_token_ids,
            soft_element_embeddings,
            device=device,
            enable_pos_emb=enable_pos_emb,
        )

        # attr and obj split
        self.offset = offset

        # additional soft_emb_dropout
        self.soft_emb_dropout = nn.Dropout(soft_emb_dropout)

        # soft_prompt
        self.soft_prompt_embeddings = soft_prompt_emebddings



    def construct_pair_txt_emb(self, pair_attr_obj_idx):
        """
        Function creates the token tensor for further inference.

        Args:
            pair_attr_obj_idx (torch.Tensor): Shape [N x 2], where N is the number
                of pairs of attr and obj

        Returns:
            torch.Tensor: token tensor passed to the text encoder;
                shape [N x context_length x 512]
        """
        # id is just for get the
        attr_idx, obj_idx = pair_attr_obj_idx[:, 0], pair_attr_obj_idx[:, 1]
        all_pair_txt_token_id = self.template_token_id.repeat(len(pair_attr_obj_idx), 1)
        all_pair_txt_token_emb = self.clip_model.token_embedding(all_pair_txt_token_id.to(self.device)).type(self.clip_model.dtype)

        if self.config.prompt_position == "front":
            # step 1: text embedding by replacing soft_prompt
            all_pair_txt_token_emb[:, 1: len(self.soft_prompt_embeddings) + 1, :] = self.soft_prompt_embeddings.type(self.clip_model.dtype)
            # step 2: replace by soft_element_embedding
            eos_idx = int(self.template_token_id[0].argmax())
            soft_embeddings = self.soft_emb_dropout(self.soft_element_embeddings).to(self.device)
            all_pair_txt_token_emb[:, eos_idx - 2, :] = soft_embeddings[attr_idx].type(self.clip_model.dtype)
            all_pair_txt_token_emb[:, eos_idx - 1, :] = soft_embeddings[obj_idx + self.offset].type(self.clip_model.dtype)
        elif self.config.prompt_position == "end":
            # step 1: text embedding by replacing soft_prompt
            all_pair_txt_token_emb[:, 3: len(self.soft_prompt_embeddings) + 3, :] = self.soft_prompt_embeddings.type(self.clip_model.dtype)
            # step 2:
            soft_embeddings = self.soft_emb_dropout(self.soft_element_embeddings)
            all_pair_txt_token_emb[:, 1, :] = soft_embeddings[attr_idx].type(self.clip_model.dtype)
            all_pair_txt_token_emb[:, 2, :] = soft_embeddings[obj_idx + self.offset].type(self.clip_model.dtype)
        elif self.config.prompt_position == "middle":       # split embeddings
            eos_idx = int(self.template_token_id[0].argmax())
            soft_embeddings = self.soft_emb_dropout(self.soft_element_embeddings)
            all_pair_txt_token_emb[:, 1, :] = soft_embeddings[attr_idx].type(self.clip_model.dtype)
            all_pair_txt_token_emb[:, eos_idx-1, :] = soft_embeddings[obj_idx + self.offset].type(self.clip_model.dtype)
            all_pair_txt_token_emb[:, 2: len(self.soft_prompt_embeddings) + 2, :] = self.soft_prompt_embeddings.type(self.clip_model.dtype)

        return all_pair_txt_token_emb


    def set_soft_embeddings(self, soft_emb, soft_prompt):
        if soft_emb.shape == self.soft_element_embeddings.shape and soft_prompt.shape == self.soft_prompt_embeddings.shape:
            self.state_dict()['soft_element_embeddings'].copy_(soft_emb)
            self.state_dict()['soft_prompt_embeddings'].copy_(soft_prompt)
        else:
            raise RuntimeError(f"Error: Incorrect Soft Embedding Shape {soft_emb.shape}, Expecting {self.soft_element_embeddings.shape}!")



    def forward(self, batch_img, train_pair_sep_attr_obj_idx):
        # ------------------
        # normalized img
        # ------------------
        batch_img = batch_img.to(self.device)
        normalized_img = batch_img / batch_img.norm(dim=-1, keepdim=True)

        # ------------------------------------------------
        # step 1: get all embedding of pair attr and obj
        # ------------------------------------------------
        pair_txt_emb = self.construct_pair_txt_emb(train_pair_sep_attr_obj_idx)

        # ---------------------------------------
        # step 2: bert for pair, attr and obj
        # ---------------------------------------
        pair_feats = self.text_encoder(
            self.template_token_id,
            pair_txt_emb,
            enable_pos_emb=self.enable_pos_emb,
        )

        # ---------------------------------------
        # step 3: normalization
        # ---------------------------------------
        _pair_features = pair_feats
        norm_pair_feat = _pair_features / _pair_features.norm(dim=-1, keepdim=True)

        # ---------------------------------------
        # step 4: calcuate logits
        # ---------------------------------------
        pair_logits = (self.clip_model.logit_scale.exp() * normalized_img @ norm_pair_feat.t())

        return pair_logits



def soft_emb_and_soft_prompt_init(
    train_dataset,
    config,
    device,
):
    """
    csp:
    1. soft_emb
    2. fixed prompt_emb
    """
    # ---------------
    # clip model
    # ---------------
    clip_model, preprocess_func = load(
        config.clip_model, device=device, context_length=config.context_length
    )
    # pytorch_total_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)

    # -------------------
    # element embedding
    # -------------------
    allattrs = train_dataset.attrs
    allobj = train_dataset.objs

    # cleaning the classes and the attributes
    obj_str_list = [obj.replace(".", " ").lower() for obj in allobj]
    attr_str_list = [attr.replace(".", " ").lower() for attr in allattrs]
    tokenized_id = torch.cat(
        [
            clip.tokenize(tok, context_length=config.context_length)
            for tok in attr_str_list + obj_str_list
        ]
    )
    orig_token_embedding = clip_model.token_embedding(tokenized_id.to(device))
    soft_element_embedding = torch.zeros(
        (len(attr_str_list) + len(obj_str_list), orig_token_embedding.size(-1)),
    )
    for idx, rep in enumerate(orig_token_embedding):
        eos_idx = tokenized_id[idx].argmax()
        soft_element_embedding[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)
    soft_element_embedding = nn.Parameter(soft_element_embedding)

    # -------------------
    # prompt embedding
    # -------------------
    if config.using_APhotoOf == 'True':
        ctx_init = "a photo of"
    else:
        ctx_init = " ".join(["X" for _ in range(config.prompt_len)])
    n_ctx = len(ctx_init.split())
    prompt_token_id = clip.tokenize(ctx_init, context_length=config.context_length).to(device)
    with torch.no_grad():
        prompt_token_embedding = clip_model.token_embedding(prompt_token_id)
    init_prompt_embedding = prompt_token_embedding[0, 1: 1 + n_ctx, :]
    soft_prompt_embedding = nn.Parameter(init_prompt_embedding).to(device)

    # --------------
    # off set
    # --------------
    offset = len(attr_str_list)

    # -----------------------
    # pair txt token ids
    # -----------------------
    # pair_prompt_template = "a photo of X X",
    pair_prompt_template = ctx_init + " X X"
    template_pair_prompt_token_ids = clip.tokenize(pair_prompt_template, context_length=config.context_length).to(device)

    return (
        clip_model,
        soft_element_embedding,
        soft_prompt_embedding,
        template_pair_prompt_token_ids,   # we must need ID to identify the token_idx to extract EOS emebddings
        offset
    )


def get_pair_soft_emb_and_soft_prompt(train_dataset, config, device):
    """   """
    # clip model, prompt and element
    (
        clip_model,
        soft_element_embedding,
        soft_prompt_embedding,
        pair_txt_token_ids,
        offset
    ) = soft_emb_and_soft_prompt_init(train_dataset, config, device)

    # optimizer
    optimizer = torch.optim.Adam(
        [soft_element_embedding] + [soft_prompt_embedding],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # using soft emb or prompt to initlize the CLIPInterface
    interface = PairSoftEmbAndSoftInterface(
        clip_model,
        config,
        offset,
        soft_element_embedding,
        soft_prompt_embedding,
        pair_txt_token_ids,
        device,
        soft_emb_dropout=config.soft_emb_dropout
    )

    return interface, optimizer
