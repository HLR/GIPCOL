## Remove this later
import argparse
import os

import clip
import torch
import torch.nn as nn
from Models.CspModel.clip_modules.interface import CLIPInterface
from Models.CspModel.clip_modules.model_loader import load

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def get_coop(train_dataset, config, device, prompt_template="a photo of x x"):
    """
    only two main pars:
    1. emebdding
    2. prompt
    we can update these two parts simutaneously.
    """

    # ------------------------
    # get clip and transform
    # ------------------------
    clip_model, preprocess = load(
        config.clip_model, device=device, context_length=config.context_length
    )


    # ------------------------
    # part 1: soft embedding
    # ------------------------
    # encode and fix element embedding
    allattrs = train_dataset.attrs
    allobj = train_dataset.objs

    # cleaning the classes and the attributes
    objs = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]

    # token_id
    tokenized = torch.cat(
        [
            clip.tokenize(tok, context_length=config.context_length)
            for tok in attributes + objs
        ]
    )

    # original embedding as text embedding
    orig_token_embedding = clip_model.token_embedding(tokenized.to(device))

    # element concept embedding as average embedding
    with torch.no_grad():
        frozen_element_embedding = torch.zeros(
            (len(attributes) + len(objs), clip_model.token_embedding.weight.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            frozen_element_embedding[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

    # -------------------
    # part 2: soft prompt
    # -------------------
    # init and fix soft prompt
    ctx_init = "a photo of "
    n_ctx = len(ctx_init.split())
    prompt = clip.tokenize(ctx_init, context_length=config.context_length).to(device)
    with torch.no_grad():
        embedding = clip_model.token_embedding(prompt)
    ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
    prompt_token_ids = clip.tokenize(prompt_template, context_length=config.context_length).to(device)
    soft_prompt_embedding = nn.Parameter(ctx_vectors).to(device)


    # -----------------------
    # soft embedding is the optimizer
    # -----------------------
    optimizer = torch.optim.Adam(
        [soft_prompt_embedding],
        lr=config.lr,
        weight_decay=config.weight_decay)
    offset = len(attributes)


    # ----------------------
    # init the coop model
    # ----------------------
    coop = COOP(
        clip_model,
        config,
        offset,
        soft_prompt_embedding,
        frozen_element_embedding,
        prompt_token_ids,
        device=device,
        enable_pos_emb=True,
    )

    # ----------------------
    # return 2 things:
    # 1.
    # ----------------------
    return coop, optimizer


class COOP(CLIPInterface):
    def __init__(
        self,
        clip_model,
        config: argparse.ArgumentParser,
        offset,
        soft_prompt_embeddings: torch.nn.Parameter,
        frozen_element_embeddings: torch.nn.Parameter,
        prompt_token_ids: torch.tensor,
        device: torch.device = "cuda:0",
        enable_pos_emb: bool = False,
    ):
        super().__init__(
            clip_model,
            config,
            prompt_token_ids,
            soft_prompt_embeddings=soft_prompt_embeddings,
            device=device,
            enable_pos_emb=enable_pos_emb,
        )
        self.frozen_element_embeddings = frozen_element_embeddings
        self.offset = offset

    def construct_token_tensors(self, pair_idx):
        """
        we don't have anything to update in the model
        1. we initilize ouside
        2. then we used the parameters to initilize the model
        another new logic

        all class token embedding
        """
        # attr idx and obj idx for emebdding
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]

        # a photo of X X
        template_token_id = self.template_token_ids.repeat(len(pair_idx), 1)

        # we get embedding
        template_token_emb = self.clip_model.token_embedding(template_token_id.to(self.device)).type(self.clip_model.dtype)

        # replate the attr and obj
        eos_idx = int(self.template_token_ids[0].argmax())
        template_token_emb[:, eos_idx - 2, :] = self.frozen_element_embeddings[attr_idx].type(self.clip_model.dtype)
        template_token_emb[:, eos_idx - 1, :] = self.frozen_element_embeddings[obj_idx + self.offset].type(self.clip_model.dtype)

        # adding the correct learnable context
        template_token_emb[:, 1 : len(self.soft_prompt_embeddings) + 1, :] = self.soft_prompt_embeddings.type(self.clip_model.dtype)

        return template_token_emb
