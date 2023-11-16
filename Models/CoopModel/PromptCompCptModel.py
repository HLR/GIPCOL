import torch
import torch.nn as nn

import clip
from Models.CoopModel.PromptLearner import SoftPromptLearner
from Models.CoopModel.TextEncoder import TxtEncoder

class PromptCompCptModel(nn.Module):
    """
    acturally it has three elements:
    1. prompt_leaner, not sure how to implement now
    2. CLIP
        2.1 img_encoder (ViT, viusal Transformer)
        2.2 text_encoder (Transformer)
    """

    def __init__(self, cfg_node, attr_cls_str_list, obj_cls_str_list, pair_cls_str_list, clip_model, pair_sep_idx):
        # parenet init
        super().__init__()
        self.cfg_node = cfg_node

        # having cls names
        self.attr_cls_str_list = attr_cls_str_list
        self.obj_cls_str_list = obj_cls_str_list
        self.pair_cls_str_list = pair_cls_str_list
        pair_cls_str_list = [" ".join(pairTuple) for pairTuple in self.pair_cls_str_list]

        # freeze embedding: 83 (train) or 116 (all) pairs
        self.frozen_element_embedding, self.attr_obj_split = self.build_element_embedding(attr_cls_str_list, obj_cls_str_list, clip_model)

        # prompt_learner
        self.attr_prompt_learner = SoftPromptLearner(cfg_node, attr_cls_str_list, clip_model, pair_sep_idx, self.frozen_element_embedding, self.attr_obj_split, suffix_type = "attr")
        self.obj_prompt_learner = SoftPromptLearner(cfg_node, obj_cls_str_list, clip_model, pair_sep_idx, self.frozen_element_embedding, self.attr_obj_split, suffix_type = "obj")
        self.pair_prompt_learner = SoftPromptLearner(cfg_node, pair_cls_str_list, clip_model, pair_sep_idx, self.frozen_element_embedding, self.attr_obj_split, suffix_type = "pair")

        # important, because we have different ways to extract txt embedding
        if self.cfg_node.UseAvgEmbedding:
            self.attr_prompt_token_ids = self.attr_prompt_learner.element_cls_prompt_token_id  # cls_size * context_length
            self.obj_prompt_token_ids = self.obj_prompt_learner.element_cls_prompt_token_id  # cls_size * context_length
            self.pair_prompt_token_ids = self.pair_prompt_learner.element_cls_prompt_token_id  # cls_size * context_length
        else:
            self.attr_prompt_token_ids = self.attr_prompt_learner.raw_cls_prompt_token_id         # cls_size * context_length
            self.obj_prompt_token_ids = self.obj_prompt_learner.raw_cls_prompt_token_id           # cls_size * context_length
            self.pair_prompt_token_ids = self.pair_prompt_learner.raw_cls_prompt_token_id         # cls_size * context_length

        # img encoder and txt encoder
        self.img_encoder = clip_model.visual
        self.text_encoder = TxtEncoder(clip_model)

        # other meta information
        self.logit_scale = clip_model.logit_scale   # scalar (4.6052) for contrastive learning
        self.dtype = clip_model.dtype    # torch.float16

    def build_element_embedding(self, attr_cls_str_list, obj_cls_str_list, clip_model):
        """
        we have non, but csp doesn't have:
        1. avarage attr and obj embedding
        2. 3 context lengs
        """
        # cleaning the classes and the attributes
        attr_list = [attr.replace(".", " ").lower() for attr in attr_cls_str_list]
        obj_list = [obj.replace(".", " ").lower() for obj in obj_cls_str_list]

        element_cpt_token_id = torch.cat(
            [
                # clip_model.tokenize(tok, context_length=self.cfg_node.context_length)
                clip.tokenize(element_cpt)
                for element_cpt in attr_list + obj_list
            ]
        )
        orig_token_embedding = clip_model.token_embedding(element_cpt_token_id)

        with torch.no_grad():
            frozen_embedding = torch.zeros(
                (len(attr_list) + len(obj_list), clip_model.token_embedding.weight.size(-1)),
            )
            for idx, rep in enumerate(orig_token_embedding):
                eos_idx = element_cpt_token_id[idx].argmax()
                frozen_embedding[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        attr_obj_split = len(attr_cls_str_list)
        return frozen_embedding, attr_obj_split


    def pairTuple2pairStr(self, list_PairTuple):
        list_PairStr = []
        for pairTuple in list_PairTuple:
            pairStr = " ".joint(pairTuple)
            list_PairStr.append(pairStr)


    def forward(self, image):
        """
        input is just image
        """

        # ----------------------------------
        # extract and norm image feat
        #   3 * 224 * 224 --> 512
        # ----------------------------------
        img_feat = self.img_encoder(image.type(self.dtype))
        image_features = img_feat / img_feat.norm(dim=-1, keepdim=True)

        # -------------------------------
        # extract and norm txt feat
        # -------------------------------
        logit_scale = self.logit_scale.exp()
        attr_logits, obj_logits, pair_logits = 0.0, 0.0, 0.0

        # attr feat
        if self.cfg_node.Trainer.CalAttrLoss:
            attr_prompt_token_ids = self.attr_prompt_token_ids
            attr_prompt_token_embedding = self.attr_prompt_learner()
            attr_prompt_embedding = self.text_encoder(attr_prompt_token_embedding, attr_prompt_token_ids)
            attr_features = attr_prompt_embedding / attr_prompt_embedding.norm(dim=-1, keepdim=True)
            attr_logits = logit_scale * image_features @ attr_features.t()  # 2 * 196, just 2 * batch_size

        # obj feat
        if self.cfg_node.Trainer.CalObjLoss:
            obj_prompt_token_ids = self.obj_prompt_token_ids
            obj_prompt_token_embedding = self.obj_prompt_learner()
            obj_feat = self.text_encoder(obj_prompt_token_embedding, obj_prompt_token_ids)
            obj_features = obj_feat / obj_feat.norm(dim=-1, keepdim=True)
            obj_logits = logit_scale * image_features @ obj_features.t()  # 2 * 196, just 2 * batch_size

        # pair feat
        if self.cfg_node.Trainer.CalPairLoss:
            pair_prompt_token_ids = self.pair_prompt_token_ids
            pair_prompt_token_embedding = self.pair_prompt_learner()
            pair_feat = self.text_encoder(pair_prompt_token_embedding, pair_prompt_token_ids)
            pair_features = pair_feat / pair_feat.norm(dim=-1, keepdim=True)
            pair_logits = logit_scale * image_features @ pair_features.t()  # 2 * 196, just 2 * batch_size

        return attr_logits, obj_logits, pair_logits
