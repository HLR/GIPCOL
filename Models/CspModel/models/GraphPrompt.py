"""
1. soft_emb and soft_prompt
2. graph convolusion to encode all pairs embedding
"""


import os

import clip
import torch
import torch.nn as nn
from Models.CspModel.clip_modules.interface import CLIPInterface
from Models.CspModel.clip_modules.model_loader import load
from Models.CspModel.models.GraphModel.graph_method import GraphFull

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class GraphPromptInterface(CLIPInterface):
    """ soft two things """
    def __init__(
        self,
        train_ds,
        graph_model,
        clip_model,
        config,
        attr_obj_offset,
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
            enable_pos_emb=enable_pos_emb
        )
        self.train_ds = train_ds

        # attr and obj split
        self.attr_obj_offset = attr_obj_offset
        self.soft_emb_dropout = nn.Dropout(soft_emb_dropout)

        # additional parts comapred with csp
        self.soft_prompt_embeddings = soft_prompt_emebddings
        self.graph_model = graph_model

        # extract the embedding
        self.embeddings = self.init_embeddings(soft_element_embeddings).to(device).detach()
        # self.register_buffer("embeddings", embeddings)
        # print()



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
        # step 1: text embedding by replacing soft_prompt
        template_token_id = self.template_token_id.repeat(len(pair_attr_obj_idx), 1)
        template_token_emb = self.clip_model.token_embedding(template_token_id.to(self.device)).type(self.clip_model.dtype)
        template_token_emb[:, 1: len(self.soft_prompt_embeddings) + 1, :] = self.soft_prompt_embeddings.type(self.clip_model.dtype)

        # step 2: go through GCN to update embeddings
        current_embeddings = self.graph_model(self.embeddings)

        # step 2: replace by soft_element_embedding
        attr_idx, obj_idx = pair_attr_obj_idx[:, 0], pair_attr_obj_idx[:, 1]
        eos_idx = int(self.template_token_id[0].argmax())
        # soft_embeddings = self.soft_emb_dropout(self.soft_element_embeddings)
        current_embeddings = self.soft_emb_dropout(current_embeddings)
        template_token_emb[:, eos_idx - 2, :] = current_embeddings[attr_idx].type(self.clip_model.dtype)
        template_token_emb[:, eos_idx - 1, :] = current_embeddings[obj_idx + self.attr_obj_offset].type(self.clip_model.dtype)

        return template_token_emb


    def set_soft_embeddings(self, soft_emb, soft_prompt):
        if soft_emb.shape == self.soft_element_embeddings.shape and soft_prompt.shape == self.soft_prompt_embeddings.shape:
            # self.state_dict()['soft_element_embeddings'].copy_(soft_emb)
            self.state_dict()['soft_prompt_embeddings'].copy_(soft_prompt)
        else:
            raise RuntimeError(f"Error: Incorrect Soft Embedding Shape {soft_emb.shape}, Expecting {self.soft_element_embeddings.shape}!")

    def set_gcn(self, gcn):
        self.graph_model.load_state_dict(gcn)


    def init_embeddings(self, element_embedding):

        def get_compositional_embeddings(embeddings, pairs):
            # Getting compositional embeddings from base embeddings
            composition_embeds = []
            for (attr, obj) in pairs:
                attr_embed = embeddings[self.train_ds.attr2idx[attr]]
                obj_embed = embeddings[self.train_ds.obj2idx[obj] + len(self.train_ds.attrs)]
                composed_embed = (attr_embed + obj_embed) / 2
                composition_embeds.append(composed_embed)
            composition_embeds = torch.stack(composition_embeds)
            print('Compositional Embeddings are ', composition_embeds.shape)
            return composition_embeds

        # init with word embeddings
        composition_embeds = get_compositional_embeddings(element_embedding, self.train_ds.all_pairs)
        full_embeddings = torch.cat([element_embedding, composition_embeds], dim=0)

        return full_embeddings



    def forward(self, batch_img, pair_attr_obj_idx):
        # ------------------
        # normalized img
        # ------------------
        batch_img = batch_img.to(self.device)
        normalized_img = batch_img / batch_img.norm(dim=-1, keepdim=True)

        # ------------------------------------------------
        # step 1: get all embedding of pair attr and obj
        # ------------------------------------------------
        # pair_txt_emb = self.construct_pair_txt_emb(pair_attr_obj_idx, self.embeddings)
        # embeddings = self.embeddings
        pair_txt_emb = self.construct_pair_txt_emb(pair_attr_obj_idx)
        # pair_txt_emb = torch.randn([1262,10,768]).to(self.device)

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



def graph_and_prompt_init(
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
    clip_model, clip_img_preprocessor = load(
        config.clip_model, device=device, context_length=config.context_length
    )

    # -------------------
    # element embedding
    # -------------------
    allattrs = train_dataset.attrs
    allobj = train_dataset.objs

    # cleaning the classes and the attributes
    objects = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    tokenized = torch.cat(
        [
            clip.tokenize(tok, context_length=config.context_length)
            for tok in attributes + objects
        ]
    )
    orig_token_embedding = clip_model.token_embedding(tokenized.to(device))
    soft_element_embedding = torch.zeros(
        (len(attributes) + len(objects), orig_token_embedding.size(-1)),
    )
    for idx, rep in enumerate(orig_token_embedding):
        eos_idx = tokenized[idx].argmax()
        soft_element_embedding[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)
    # soft_element_embedding = nn.Parameter(soft_element_embedding)

    # -------------------
    # prompt embedding
    # -------------------
    ctx_init = "a photo of"
    n_ctx = len(ctx_init.split())
    prompt_token_id = clip.tokenize(ctx_init, context_length=config.context_length).to(device)
    with torch.no_grad():
        prompt_token_embedding = clip_model.token_embedding(prompt_token_id)
    init_prompt_embedding = prompt_token_embedding[0, 1: 1 + n_ctx, :]
    soft_prompt_embedding = nn.Parameter(init_prompt_embedding).to(device)

    # --------------
    # off set
    # --------------
    offset = len(attributes)

    # -----------------------
    # pair txt token ids
    # -----------------------
    pair_prompt_template = "a photo of X X",
    template_text_token_ids = clip.tokenize(pair_prompt_template, context_length=config.context_length).to(device)


    # ---------------------
    # GCN related
    # ---------------------


    return (
        clip_model,
        soft_element_embedding,
        soft_prompt_embedding,
        template_text_token_ids,   # we must need ID to identify the token_idx to extract EOS emebddings
        offset
    )




def get_graph_and_prompt(train_dataset, config, device):
    """   """
    # ---------------------------------
    # clip model, prompt and element
    # ---------------------------------
    (
        clip_model,
        soft_element_embedding,
        soft_prompt_embedding,
        template_token_ids,
        att_obj_offset
    ) = graph_and_prompt_init(train_dataset, config, device)

    # ----------------------------
    # graph model
    # ----------------------------
    graph_model = GraphFull(train_dataset, config, soft_element_embedding).to(device)

    # -------------------------
    # prepare two sets of params:
    # 1. GCN
    # 2. soft_emb + soft_prompt
    # -------------------------
    optim_params = [
        {
            'params': [soft_prompt_embedding],
            'lr': config.lr,
            'weight_decay': config.weight_decay
        },
        {
            'params': graph_model.parameters(),
            'lr': config.graph_lr,
            'weight_decay': config.graph_weight_decay
        }
    ]

    # optimizer
    optimizer = torch.optim.Adam(optim_params)

    # using soft emb or prompt to initlize the CLIPInterface
    clip_interface = GraphPromptInterface(
        train_dataset,
        graph_model,
        clip_model,
        config,
        att_obj_offset,
        soft_element_embedding,
        soft_prompt_embedding,
        template_token_ids,
        device,
        soft_emb_dropout=config.soft_emb_dropout
    )

    return clip_interface, optimizer
