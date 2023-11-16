import os

import clip
import pandas as pd
import torch
import torch.nn as nn
from Models.CspModel.clip_modules.interface import CLIPInterface
from Models.CspModel.clip_modules.model_loader import load

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class CSPInterface(CLIPInterface):
    def __init__(
        self,
        clip_model,
        config,
        offset,
        soft_element_embeddings,
        fixed_class_token_ids,
        device="cuda",
        enable_pos_emb=True,
        attr_dropout=0.0,
    ):
        super().__init__(
            clip_model,
            config,
            fixed_class_token_ids,
            soft_element_embeddings,
            device=device,
            enable_pos_emb=enable_pos_emb,
        )

        self.offset = offset
        self.soft_element_emb_dropout = nn.Dropout(attr_dropout)


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
        attr_idx, obj_idx = pair_attr_obj_idx[:, 0], pair_attr_obj_idx[:, 1]
        prompt_token_id = self.template_token_id.repeat(len(pair_attr_obj_idx), 1)
        prompt_token_embed = self.clip_model.token_embedding(
            prompt_token_id.to(self.device)
        ).type(self.clip_model.dtype)

        eos_idx = int(self.template_token_id[0].argmax())
        soft_embeddings = self.soft_element_emb_dropout(self.soft_element_embeddings)
        prompt_token_embed[:, eos_idx - 2, :] = soft_embeddings[
            attr_idx
        ].type(self.clip_model.dtype)
        prompt_token_embed[:, eos_idx - 1, :] = soft_embeddings[
            obj_idx + self.offset
        ].type(self.clip_model.dtype)

        return prompt_token_embed


    def construct_attr_token_tensors(self, attr_idx):
        """Function creates the token tensor for further inference.

        Args:
            pair_attr_obj_idx (torch.Tensor): Shape [N x 2], where N is the number
                of pairs of attr and obj

        Returns:
            torch.Tensor: token tensor passed to the text encoder;
                shape [N x context_length x 512]
        """
        # ------------------
        # we just embed the prompt:
        # 1. a photo of X X
        # 2. not realted to soft embedding
        # ------------------
        prompt_attr_token_id = self.prompt_attr_token_id.repeat(len(attr_idx), 1)
        prompt_attr_token_embed = self.clip_model.token_embedding(
            prompt_attr_token_id.to(self.device)
        ).type(self.clip_model.dtype)

        eos_idx = int(self.template_token_id[0].argmax())
        soft_embeddings = self.soft_element_emb_dropout(self.soft_element_embeddings)
        prompt_attr_token_embed[:, eos_idx - 1, :] = soft_embeddings[attr_idx].type(self.clip_model.dtype)

        return prompt_attr_token_embed



    def construct_obj_token_tensors(self, obj_idx):
        """Function creates the token tensor for further inference.

        Args:
            pair_attr_obj_idx (torch.Tensor): Shape [N x 2], where N is the number
                of pairs of attr and obj

        Returns:
            torch.Tensor: token tensor passed to the text encoder;
                shape [N x context_length x 512]
        """
        prompt_token_id = self.prompt_obj_token_id.repeat(len(obj_idx), 1)
        prompt_token_embed = self.clip_model.token_embedding(prompt_token_id.to(self.device)).type(self.clip_model.dtype)

        eos_idx = int(self.template_token_id[0].argmax())
        soft_embeddings = self.soft_element_emb_dropout(self.soft_element_embeddings)

        prompt_token_embed[:, eos_idx - 1, :] = soft_embeddings[obj_idx + self.offset].type(self.clip_model.dtype)

        return prompt_token_embed


def csp_init(
    train_dataset,
    config,
    device,
    pair_prompt_template="a photo of X X",
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
    soft_embedding = torch.zeros(
        (len(attributes) + len(classes), orig_token_embedding.size(-1)),
    )
    for idx, rep in enumerate(orig_token_embedding):
        eos_idx = tokenized[idx].argmax()
        soft_embedding[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)
    soft_embedding = nn.Parameter(soft_embedding)


    # -----------------
    # prompt
    # -----------------
    pair_prompt_token_ids = clip.tokenize(
        [pair_prompt_template],
        context_length=config.context_length,
    )


    offset = len(attributes)

    return (
        clip_model,
        soft_embedding,
        pair_prompt_token_ids,
        offset
    )



def get_csp(train_dataset, config, device):

    (
        clip_model,
        soft_element_embedding,
        fixed_prompt_pair_token_id,
        offset
    ) = csp_init(train_dataset, config, device)

    optimizer = torch.optim.Adam(
        [soft_element_embedding],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    interface = CSPInterface(
        clip_model,
        config,
        offset,
        soft_element_embedding,
        fixed_prompt_pair_token_id,
        device,
        attr_dropout=config.soft_emb_dropout
    )

    return interface, optimizer


def get_mix_csp(train_dataset, config, device):
    """
    not sure what mix means：
    1.
    """
    (
        clip_model,
        soft_embedding,
        class_token_ids,
        offset
    ) = csp_init(train_dataset, config, device)

    with torch.no_grad():
        subset_soft_embeddings = soft_embedding[train_dataset.indices, :]

    subset_soft_embeddings.requires_grad = True

    optimizer = torch.optim.Adam(
        [subset_soft_embeddings],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # reduce the offset to selected offset
    offset = len(train_dataset.attr_indices)
    interface = CSPInterface(
        clip_model,
        config,
        offset,
        subset_soft_embeddings,
        class_token_ids,
        device,
        attr_dropout=config.soft_emb_dropout
    )

    return interface, optimizer


def get_sep_space_soft_emb(train_dataset, config, device):
    """   """
    # clip model, prompt and element
    (
        clip_model,
        soft_element_embedding,
        fixed_prompt_pair_token_id,
        offset
    ) = csp_init(train_dataset, config, device)

    # optimizer
    optimizer = torch.optim.Adam(
        [soft_element_embedding],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # using soft emb or prompt to initlize the CLIPInterface
    interface = CSPInterface(
        clip_model,
        config,
        offset,
        soft_element_embedding,
        fixed_prompt_pair_token_id,
        device,
        attr_dropout=config.soft_emb_dropout
    )

    return interface, optimizer
