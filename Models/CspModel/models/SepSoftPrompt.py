import os

import clip
import pandas as pd
import torch
import torch.nn as nn
from Models.CspModel.clip_modules.interface import CLIPInterface
from Models.CspModel.clip_modules.model_loader import load

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class SepSoftPromptInterface(CLIPInterface):
    def __init__(
        self,
        clip_model,
        config,
        offset,
        soft_element_embeddings,
        pair_soft_prompt_emb,
        attr_soft_prompt_emb,
        obj_soft_prompt_emb,
        pair_txt_input_ids,
        attr_txt_input_ids,
        obj_txt_input_ids,
        device="cuda",
        enable_pos_emb=True,
        soft_emb_dropout=0.0,
    ):
        super().__init__(
            clip_model,
            config,
            pair_txt_input_ids,
            soft_element_embeddings,
            device=device,
            enable_pos_emb=enable_pos_emb,
        )


        self.offset = offset
        self.soft_emb_dropout = nn.Dropout(soft_emb_dropout)

        self.attr_txt_token_id = attr_txt_input_ids
        self.obj_txt_token_id = obj_txt_input_ids

        self.pair_soft_prompt_emb = pair_soft_prompt_emb
        self.attr_soft_prompt_emb = attr_soft_prompt_emb
        self.obj_soft_prompt_emb = obj_soft_prompt_emb



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
        # step 1: text embedding
        attr_idx, obj_idx = pair_attr_obj_idx[:, 0], pair_attr_obj_idx[:, 1]
        pair_txt_token_id = self.template_token_id.repeat(len(pair_attr_obj_idx), 1)
        pair_txt_token_emb = self.clip_model.token_embedding(pair_txt_token_id.to(self.device)).type(self.clip_model.dtype)

        # step 2: replace element embedding
        eos_idx = int(self.template_token_id[0].argmax())
        soft_embeddings = self.soft_emb_dropout(self.soft_element_embeddings)
        pair_txt_token_emb[:, eos_idx - 2, :] = soft_embeddings[attr_idx].type(self.clip_model.dtype)
        pair_txt_token_emb[:, eos_idx - 1, :] = soft_embeddings[obj_idx + self.offset].type(self.clip_model.dtype)

        # step 3: replace soft prompt
        pair_txt_token_emb[:, 1 : len(self.pair_soft_prompt_emb) + 1, :] = self.pair_soft_prompt_emb.type(self.clip_model.dtype)

        return pair_txt_token_emb


    def construct_attr_txt_emb(self, attr_idx):
        """ Function creates the token tensor for further inference. """
        # ------------------
        # we just embed the prompt:
        # 1. a photo of X X
        # 2. not realted to soft embedding
        # ------------------
        # step 1: text embedding
        attr_txt_token_ids = self.attr_txt_token_id.repeat(len(attr_idx), 1)
        attr_txt_token_emb = self.clip_model.token_embedding(
            attr_txt_token_ids.to(self.device)
        ).type(self.clip_model.dtype)

        # step 2: replace element embedding
        eos_idx = int(self.attr_txt_token_id[0].argmax())
        soft_embeddings = self.soft_emb_dropout(self.soft_element_embeddings)
        attr_txt_token_emb[:, eos_idx - 1, :] = soft_embeddings[attr_idx].type(self.clip_model.dtype)

        # step 3: replace soft prompt
        attr_txt_token_emb[:, 1: len(self.attr_soft_prompt_emb) + 1, :] = self.attr_soft_prompt_emb.type(self.clip_model.dtype)

        return attr_txt_token_emb



    def construct_obj_txt_emb(self, obj_idx):
        """ Function creates the token tensor for further inference. """
        # step 1: text embedding
        obj_txt_token_ids = self.obj_txt_token_id.repeat(len(obj_idx), 1)
        obj_txt_token_emb = self.clip_model.token_embedding(obj_txt_token_ids.to(self.device)).type(self.clip_model.dtype)

        # step 2: replace element embedding
        eos_idx = int(self.template_token_id[0].argmax())
        soft_embeddings = self.soft_emb_dropout(self.soft_element_embeddings)
        obj_txt_token_emb[:, eos_idx - 1, :] = soft_embeddings[obj_idx + self.offset].type(self.clip_model.dtype)

        # step 3: replace soft prompt
        obj_txt_token_emb[:, 1: len(self.attr_soft_prompt_emb) + 1, :] = self.obj_soft_prompt_emb.type(self.clip_model.dtype)

        return obj_txt_token_emb


    def set_soft_embeddings(self, pair_soft_emb, attr_soft_emb, obj_soft_emb):
        """ """

        if pair_soft_emb.shape == self.pair_soft_prompt_emb.shape \
                and attr_soft_emb.shape == self.attr_soft_prompt_emb.shape \
                and obj_soft_emb.shape == self.obj_soft_prompt_emb.shape:
            self.state_dict()['pair_soft_prompt_emb'].copy_(pair_soft_emb)
            self.state_dict()['attr_soft_prompt_emb'].copy_(attr_soft_emb)
            self.state_dict()['obj_soft_prompt_emb'].copy_(obj_soft_emb)
        else:
            raise RuntimeError(
                f"Error: Sep_Soft_Prompt shape not matching !")

    def forward(self, batch_img, pair_attr_obj_idx, attr_idx, obj_idx):
        batch_img = batch_img.to(self.device)
        normalized_img = batch_img / batch_img.norm(dim=-1, keepdim=True)

        # ------------------------------------------------
        # step 1: get all embedding of pair attr and obj
        # ------------------------------------------------
        pair_emb = self.construct_pair_txt_emb(pair_attr_obj_idx)
        attr_emb = self.construct_attr_txt_emb(attr_idx)
        obj_emb = self.construct_obj_txt_emb(obj_idx)

        # ---------------------------------------
        # step 2: bert for pair, attr and obj
        # ---------------------------------------
        pair_feats = self.text_encoder(
            self.template_token_id,
            pair_emb,
            enable_pos_emb=self.enable_pos_emb,
        )

        attr_feats = self.text_encoder(
            self.attr_txt_token_id,
            attr_emb,
            enable_pos_emb=self.enable_pos_emb,
        )

        obj_feats = self.text_encoder(
            self.obj_txt_token_id,
            obj_emb,
            enable_pos_emb=self.enable_pos_emb,
        )

        # ---------------------------------------
        # step 3: normalization
        # ---------------------------------------
        #_text_features = text_features[idx, :]
        _pair_features = pair_feats
        norm_pair_feat = _pair_features / _pair_features.norm(dim=-1, keepdim=True)

        _attr_features = attr_feats
        norm_attr_feat = _attr_features / _attr_features.norm(dim=-1, keepdim=True)

        _obj_features = obj_feats
        norm_obj_feat = _obj_features / _obj_features.norm(dim=-1, keepdim=True)

        # ---------------------------------------
        # step 4: calcuate logits
        # ---------------------------------------
        pair_logits = (self.clip_model.logit_scale.exp() * normalized_img @ norm_pair_feat.t())
        attr_logits = (self.clip_model.logit_scale.exp() * normalized_img @ norm_attr_feat.t())
        obj_logits = (self.clip_model.logit_scale.exp() * normalized_img @ norm_obj_feat.t())

        return pair_logits, attr_logits, obj_logits



def sep_soft_prompt_init(
    train_dataset,
    config,
    device,
    pair_prompt_template = "a photo of X X",
    attr_prompt_template = "a photo of X",
    obj_prompt_template = "a photo of X",
):
    """
    csp:
    1. soft_emb
    2. fixed prompt_emb
    """

    # clip model
    clip_model, preprocess = load(
        config.clip_model, device=device, context_length=config.context_length
    )

    # -------------------
    # encode and fix elements
    # -------------------
    allattrs = train_dataset.attrs
    allobj = train_dataset.objs

    # cleaning the classes and the attributes
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    tokenized = torch.cat(
        [
            clip.tokenize(tok, context_length=config.context_length)
            for tok in attributes + classes
        ]
    )
    orig_token_embedding = clip_model.token_embedding(tokenized.to(device))
    soft_embedding = torch.zeros(
        (len(attributes) + len(classes), orig_token_embedding.size(-1)),
    )
    for idx, rep in enumerate(orig_token_embedding):
        eos_idx = tokenized[idx].argmax()
        soft_embedding[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)
    soft_embedding = nn.Parameter(soft_embedding)


    # --------------------------
    # soft prompt construction
    # --------------------------

    # prompts shared by pair, attr and obj
    shared_ctx_init = "a photo of "
    n_ctx = len(shared_ctx_init.split())
    prompt = clip.tokenize(shared_ctx_init, context_length=config.context_length).to(device)
    with torch.no_grad():
        embedding = clip_model.token_embedding(prompt)
    ctx_vectors = embedding[0, 1: 1 + n_ctx, :]

    # sep prompt token emb initialized by "a photo of" and length is 3
    pair_soft_prompt_embedding = nn.Parameter(ctx_vectors).to(device)
    attr_soft_prompt_embedding = nn.Parameter(ctx_vectors).to(device)
    obj_soft_prompt_embedding = nn.Parameter(ctx_vectors).to(device)

    # sep txt input idx including [prompt, class_idx]
    pair_txt_input_ids = clip.tokenize(
        [pair_prompt_template],
        context_length=config.context_length,
    ).to(device)
    attr_txt_input_ids = clip.tokenize(
        [attr_prompt_template],
        context_length=config.context_length,
    ).to(device)
    obj_txt_input_ids = clip.tokenize(
        [obj_prompt_template],
        context_length=config.context_length,
    ).to(device)

    offset = len(attributes)

    return (
        clip_model,
        soft_embedding,
        pair_soft_prompt_embedding,
        attr_soft_prompt_embedding,
        obj_soft_prompt_embedding,
        pair_txt_input_ids,
        attr_txt_input_ids,
        obj_txt_input_ids,
        offset
    )


def get_sep_soft_prompt(train_dataset, config, device):
    """   """
    # clip model, prompt and element
    (
        clip_model,
        soft_ele_embedding,
        soft_pair_prompt_embedding,
        soft_attr_prompt_embedding,
        soft_obj_prompt_embedding,
        pair_txt_input_ids,
        attr_txt_input_ids,
        obj_txt_input_ids,
        offset
    ) = sep_soft_prompt_init(train_dataset, config, device)

    # optimizer
    optimizer = torch.optim.Adam(
        [soft_pair_prompt_embedding] + [soft_attr_prompt_embedding] + [soft_obj_prompt_embedding],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # using soft emb or prompt to initlize the CLIPInterface
    interface = SepSoftPromptInterface(
        clip_model,
        config,
        offset,
        soft_ele_embedding,
        soft_pair_prompt_embedding,
        soft_attr_prompt_embedding,
        soft_obj_prompt_embedding,
        pair_txt_input_ids,
        attr_txt_input_ids,
        obj_txt_input_ids,
        device,
        soft_emb_dropout=config.soft_emb_dropout
    )

    return interface, optimizer
