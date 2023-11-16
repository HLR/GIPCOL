import argparse

import torch
from clip.model import CLIP

from .text_encoder import CustomTextEncoder


class CLIPInterface(torch.nn.Module):
    def __init__(
        self,
        clip_model: CLIP,
        config: argparse.ArgumentParser,
        template_token_id: torch.tensor,
        soft_element_embeddings: torch.nn.Parameter = None,
        dtype: torch.dtype = None,
        device: torch.device = "cuda",
        enable_pos_emb: bool = False
    ):
        """CLIP interface for our custom modules.

        Args:
            clip_model (CLIP): the clip model
            config (argparse.ArgumentParser): arguments used for
                training
            template_token_id (torch.tensor): the input token ids to the text
                encoder
            soft_element_embeddings (torch.nn.Parameter, optional): the only
                parameter that we finetune in the experiment.
                Defaults to None.
            dtype (torch.dtype, optional): torch dtype for the
                transformer. This allows the half precision option.
                Defaults to None.
            device (torch.device, optional): the device where the model
                should be loaded. Defaults to "cuda:0".
            enable_pos_emb (bool, optional): if true, adds the learned
                positional embeddings. Defaults to False.
        """
        super().__init__()

        self.config = config

        self.clip_model = clip_model

        if dtype is None and device == "cpu":
            self.dtype = torch.float32
        elif dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype

        self.device = device

        self.enable_pos_emb = enable_pos_emb

        self.text_encoder = CustomTextEncoder(clip_model, self.dtype)
        for params in self.text_encoder.parameters():
            params.requires_grad = False
        self.clip_model.text_projection.requires_grad = False

        self.template_token_id = template_token_id
        self.soft_element_embeddings = soft_element_embeddings

    def encode_image(self, imgs):
        return self.clip_model.encode_image(imgs)

    def encode_text(self, text, enable_pos_emb=True):
        return self.text_encoder.encode_text(
            text, enable_pos_emb=enable_pos_emb
        )

    def tokenize(self, text):
        return self.text_encoder.tokenize(text)

    def set_soft_embeddings(self, se):
        if se.shape == self.soft_element_embeddings.shape:
            self.state_dict()['soft_element_embeddings'].copy_(se)
        else:
            raise RuntimeError(f"Error: Incorrect Soft Embedding Shape {se.shape}, Expecting {self.soft_element_embeddings.shape}!")

    def construct_token_tensors(self, idx):
        """ pair_cls_num * 77(context_length) * 512 """
        if self.soft_element_embeddings is None:
            return None
        else:
            # Implement a custom version
            raise NotImplementedError


    def construct_attr_token_tensors(self, attr_idx):
        """ attr_cls_num * 77(context_length) * 512 """
        if self.soft_element_embeddings is None:
            return None
        else:
            # Implement a custom version
            raise NotImplementedError


    def construct_obj_token_tensors(self, obj_idx):
        """ obj_cls_num * 77(context_length) * 512 """
        if self.soft_element_embeddings is None:
            return None
        else:
            # Implement a custom version
            raise NotImplementedError



    def forward(self, batch_img, pair_attr_obj_idx):

        # -----------------
        # image encoding
        # -----------------
        batch_img = batch_img.to(self.device)
        normalized_img = batch_img / batch_img.norm(dim=-1, keepdim=True)


        # ------------------------------------------------
        # step 1: get all embedding of pair attr and obj
        # ------------------------------------------------
        pair_emb = self.construct_token_tensors(pair_attr_obj_idx)

        # ---------------------------------------
        # step 2: bert for pair, attr and obj
        # ---------------------------------------
        pair_feats = self.text_encoder(
            self.template_token_id,
            pair_emb,
            enable_pos_emb=self.enable_pos_emb,
        )

        # ---------------------------------------
        # step 3: normalization
        # ---------------------------------------
        #_text_features = text_features[idx, :]
        _pair_features = pair_feats
        norm_pair_feat = _pair_features / _pair_features.norm(dim=-1, keepdim=True)

        # ---------------------------------------
        # step 4: calcuate logits
        # ---------------------------------------
        pair_logits = (self.clip_model.logit_scale.exp() * normalized_img @ norm_pair_feat.t())

        return pair_logits
