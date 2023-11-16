import os

import clip
import pandas as pd
import torch
import torch.nn as nn
from Models.CspModel.clip_modules.interface import CLIPInterface
from Models.CspModel.clip_modules.model_loader import load

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class SoftEmbAndSoftInterface(CLIPInterface):
    """ soft two things """
    def __init__(
        self,
        clip_model,
        config,
        offset,
        soft_element_embeddings,
        soft_prompt_emebddings,
        pair_prompt_token_ids,
        device="cuda",
        enable_pos_emb=True,
        soft_emb_dropout=0.0,
    ):
        super().__init__(
            clip_model,
            config,
            pair_prompt_token_ids,
            soft_element_embeddings,
            device=device,
            enable_pos_emb=enable_pos_emb,
        )

        # attr and obj split
        self.offset = offset
        self.soft_emb_dropout = nn.Dropout(soft_emb_dropout)

        # important
        self.soft_prompt_embeddings = soft_prompt_emebddings


    def construct_pair_txt_input(self, pair_attr_obj_idx):
        """
        Function creates the token tensor for further inference.

        Args:
            pair_attr_obj_idx (torch.Tensor): Shape [N x 2], where N is the number
                of pairs of attr and obj

        Returns:
            torch.Tensor: token tensor passed to the text encoder;
                shape [N x context_length x 512]
        """
        # step 1: text embedding by replacing soft_prompt
        attr_idx, obj_idx = pair_attr_obj_idx[:, 0], pair_attr_obj_idx[:, 1]
        prompt_token_id = self.template_token_id.repeat(len(pair_attr_obj_idx), 1)
        prompt_token_embed = self.clip_model.token_embedding(prompt_token_id.to(self.device)).type(self.clip_model.dtype)
        prompt_token_embed[:, 1: len(self.soft_prompt_embeddings) + 1, :] = self.soft_prompt_embeddings.type(self.clip_model.dtype)

        # step 2: replace by soft_element_embedding
        eos_idx = int(self.template_token_id[0].argmax())
        soft_embeddings = self.soft_emb_dropout(self.soft_element_embeddings)
        prompt_token_embed[:, eos_idx - 2, :] = soft_embeddings[attr_idx].type(self.clip_model.dtype)
        prompt_token_embed[:, eos_idx - 1, :] = soft_embeddings[obj_idx + self.offset].type(self.clip_model.dtype)

        return prompt_token_embed


    def construct_attr_txt_input(self, attr_idx):
        """ Function creates the token tensor for further inference. """
        # ------------------
        # we just embed the prompt:
        # 1. a photo of X X
        # 2. not realted to soft embedding
        # ------------------
        prompt_attr_token_id = self.fixed_attr_prompt_token_ids.repeat(len(attr_idx), 1)
        prompt_attr_token_embed = self.clip_model.token_embedding(
            prompt_attr_token_id.to(self.device)
        ).type(self.clip_model.dtype)

        eos_idx = int(self.template_token_id[0].argmax())
        soft_embeddings = self.soft_emb_dropout(self.soft_element_embeddings)
        prompt_attr_token_embed[:, eos_idx - 1, :] = soft_embeddings[attr_idx].type(self.clip_model.dtype)

        return prompt_attr_token_embed



    def construct_obj_txt_input(self, obj_idx):
        """ Function creates the token tensor for further inference. """
        #
        prompt_token_id = self.fixed_obj_prompt_token_ids.repeat(len(obj_idx), 1)
        prompt_token_embed = self.clip_model.token_embedding(prompt_token_id.to(self.device)).type(self.clip_model.dtype)
        eos_idx = int(self.template_token_id[0].argmax())
        soft_embeddings = self.soft_emb_dropout(self.soft_element_embeddings)
        prompt_token_embed[:, eos_idx - 1, :] = soft_embeddings[obj_idx + self.offset].type(self.clip_model.dtype)
        return prompt_token_embed




    def forward(self, batch_img, pair_attr_obj_idx, attr_idx, obj_idx):
        batch_img = batch_img.to(self.device)
        normalized_img = batch_img / batch_img.norm(dim=-1, keepdim=True)

        # ------------------------------------------------
        # step 1: get all embedding of pair attr and obj
        # ------------------------------------------------
        pair_emb = self.construct_pair_txt_input(pair_attr_obj_idx)
        # attr_emb = self.construct_attr_txt_input(attr_idx)
        # obj_emb = self.construct_obj_txt_input(obj_idx)

        # ---------------------------------------
        # step 2: bert for pair, attr and obj
        # ---------------------------------------
        pair_feats = self.text_encoder(
            self.template_token_id,
            pair_emb,
            enable_pos_emb=self.enable_pos_emb,
        )

        """
        attr_feats = self.text_encoder(
            self.fixed_attr_prompt_token_ids,
            attr_emb,
            enable_pos_emb=self.enable_pos_emb,
        )

        obj_feats = self.text_encoder(
            self.fixed_obj_prompt_token_ids,
            obj_emb,
            enable_pos_emb=self.enable_pos_emb,
        )
        """

        # ---------------------------------------
        # step 3: normalization
        # ---------------------------------------
        #_text_features = text_features[idx, :]
        _pair_features = pair_feats
        norm_pair_feat = _pair_features / _pair_features.norm(dim=-1, keepdim=True)

        #_attr_features = attr_feats
        #norm_attr_feat = _attr_features / _attr_features.norm(dim=-1, keepdim=True)

        #_obj_features = obj_feats
        #norm_obj_feat = _obj_features / _obj_features.norm(dim=-1, keepdim=True)

        # ---------------------------------------
        # step 4: calcuate logits
        # ---------------------------------------
        pair_logits = (self.clip_model.logit_scale.exp() * normalized_img @ norm_pair_feat.t())
        # attr_logits = (self.clip_model.logit_scale.exp() * normalized_img @ norm_attr_feat.t())
        # obj_logits = (self.clip_model.logit_scale.exp() * normalized_img @ norm_obj_feat.t())

        # return pair_logits, attr_logits, obj_logits
        return pair_logits, None, None



def soft_emb_and_soft_prompt_init(
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
    # encode elements
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
    soft_element_embedding = torch.zeros(
        (len(attributes) + len(classes), orig_token_embedding.size(-1)),
    )
    for idx, rep in enumerate(orig_token_embedding):
        eos_idx = tokenized[idx].argmax()
        soft_element_embedding[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)
    soft_element_embedding = nn.Parameter(soft_element_embedding)


    # -----------------
    # prompt
    # -----------------
    ctx_init = "a photo of "
    n_ctx = len(ctx_init.split())
    prompt = clip.tokenize(ctx_init, context_length=config.context_length).to(device)
    with torch.no_grad():
        embedding = clip_model.token_embedding(prompt)
    ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
    prompt_token_ids = clip.tokenize(pair_prompt_template, context_length=config.context_length).to(device)
    soft_prompt_embedding = nn.Parameter(ctx_vectors).to(device)

    # --------------
    # off set
    # --------------
    offset = len(attributes)

    return (
        clip_model,
        soft_element_embedding,
        soft_prompt_embedding,
        prompt_token_ids,   # we must need ID to identify the token_idx to extract EOS emebddings
        offset
    )


def get_sep_soft_emb_and_soft_prompt(train_dataset, config, device):
    """   """
    # clip model, prompt and element
    (
        clip_model,
        soft_element_embedding,
        soft_prompt_embedding,
        prompt_token_ids,
        offset
    ) = soft_emb_and_soft_prompt_init(train_dataset, config, device)

    # optimizer
    optimizer = torch.optim.Adam(
        [soft_element_embedding] + [soft_prompt_embedding],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # using soft emb or prompt to initlize the CLIPInterface
    interface = SoftEmbAndSoftInterface(
        clip_model,
        config,
        offset,
        soft_element_embedding,
        soft_prompt_embedding,
        prompt_token_ids,
        device,
        soft_emb_dropout=config.soft_emb_dropout
    )

    return interface, optimizer
