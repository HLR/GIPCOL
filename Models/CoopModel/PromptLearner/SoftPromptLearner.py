import torch
import torch.nn as nn
import Models.CoopModel.clip.clip as clip
from Models.CoopModel.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

class SoftPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, all_pair_sep_idx, frozen_element_embedding, attr_obj_split, suffix_type):
        super().__init__()

        # for avg_element_emebdding
        self.frozen_element_embedding = frozen_element_embedding
        self.attr_obj_split =attr_obj_split
        self.suffix_type = suffix_type

        self.dtype = clip_model.dtype

        # use raw string or element embedding
        self.use_avg_embedding = cfg.UseAvgEmbedding

        # attr and obj seperately
        self.all_pair_sep_idx = all_pair_sep_idx

        # cls number
        cls_num = len(classnames)

        # txt input
        n_ctx = cfg.Prompt.ContextLen
        ctx_init = cfg.Prompt.ContextInitStr
        ctx_dim = clip_model.ln_final.weight.shape[0]

        self.cls_num = cls_num
        self.ctx_len = n_ctx
        self.class_token_position = cfg.Prompt.ClassTokenPosition

        # assert: img input
        clip_img_size = clip_model.visual.input_resolution
        cfg_img_size = cfg.CLIP.ImgSize[0]
        assert cfg_img_size == clip_img_size, f"cfg_imsize ({cfg_img_size}) must equal to clip_imsize ({clip_img_size})"

        # ------------------------
        # ctx has its own
        # ------------------------
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                cls_prompt_token_embedding = clip_model.token_embedding(prompt).type(self.dtype)
            soft_ctx_embedding = cls_prompt_token_embedding[0, 1 : 1 + n_ctx, :]
            prompt_token_prefix = ctx_init
        else:
            # random initialization
            if cfg.Prompt.ClassSpecific:
                print("Initializing class-specific contexts")
                soft_ctx_embedding = torch.empty(cls_num, n_ctx, ctx_dim, dtype=self.dtype)
            else:
                print("Initializing a generic context")
                soft_ctx_embedding = torch.empty(n_ctx, ctx_dim, dtype=self.dtype)
            nn.init.normal_(soft_ctx_embedding, std=0.02)

            if self.suffix_type == 'pair':
                self.prompt_txt_template_str = " ".join(["A"] * (n_ctx)) + ' X X'
            else:
                self.prompt_txt_template_str = " ".join(["A"] * (n_ctx)) + ' X'
            self.element_cls_prompt_token_id = clip.tokenize(self.prompt_txt_template_str)
            # self.element_cls_prompt_token_id = torch.cat([clip.tokenize(p) for p in self.prompt_txt_template_str])

        print(f'Initial context: "{self.prompt_txt_template_str}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # ------------------------------------------------
        # copy csp and initilize the soft_prompt using
        # ------------------------------------------------
        ctx_init = "a photo of "
        n_ctx = len(ctx_init.split())
        prompt = clip.tokenize(ctx_init)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(self.dtype)
        ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
        self.soft_prompt_embedding = nn.Parameter(ctx_vectors)  # to be optimized

        token_avg_emb_suffix = self.construct_cls_prompt_avg_element_suffix(all_pair_sep_idx, clip_model)
        self.register_buffer("token_avg_ele_suffix", token_avg_emb_suffix)

        # --------------------------------
        # use cls_name instead of cls_id
        # --------------------------------
        # cls label as text
        classnames = [name.replace(".", " ") for name in classnames]
        cls_name_len = [len(_tokenizer.encode(name)) for name in classnames]       # split into more tokens
        self.cls_name_len = cls_name_len

        # x x x + cls_name

        self.raw_cls_prompt_txt = ["X X X" + " " + name + "." for name in classnames]
        self.raw_cls_prompt_token_id = torch.cat([clip.tokenize(p) for p in self.raw_cls_prompt_txt])
        with torch.no_grad():
            self.raw_cls_prompt_token_embedding = clip_model.token_embedding(self.raw_cls_prompt_token_id).type(self.dtype)


        # ----------------------------------------------------------------
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        # ----------------------------------------------------------------
        self.register_buffer("token_prefix", self.raw_cls_prompt_token_embedding[:, :1, :])  # SOS
        self.register_buffer("token_str_suffix", self.raw_cls_prompt_token_embedding[:, 1 + n_ctx:, :])  # CLS, EOS



    def construct_cls_prompt_avg_element_suffix(self, pair_sep_idx, clip_model):
        """
        prompt:
        1. str_format: sos soft_embedding cls_label, [SOS, X, X, ... X, Red, Car]
        2. token_format: 40966 ... 40967
        3. token_embedding:
        """
        attr_idx, obj_idx = pair_sep_idx[:, 0], pair_sep_idx[:, 1]
        cls_prompt_token_id = self.element_cls_prompt_token_id.repeat(self.cls_num, 1)
        cls_prompt_token_embedding = clip_model.token_embedding(cls_prompt_token_id).type(clip_model.dtype)

        eos_idx = int(self.element_cls_prompt_token_id[0].argmax())
        if self.suffix_type == "pair":
            cls_prompt_token_embedding[:, eos_idx - 2, :] = self.frozen_element_embedding[attr_idx].type(clip_model.dtype)
            cls_prompt_token_embedding[:, eos_idx - 1, :] = self.frozen_element_embedding[obj_idx + self.attr_obj_split].type(clip_model.dtype)
        elif self.suffix_type == 'attr':
            cls_prompt_token_embedding[:, eos_idx - 1, :] = self.frozen_element_embedding[list(range(self.cls_num))].type(clip_model.dtype)
        elif self.suffix_type == 'obj':
            obj_idx = [self.attr_obj_split + x  for x in list(range(self.cls_num))]
            cls_prompt_token_embedding[:, eos_idx - 1, :] = self.frozen_element_embedding[obj_idx].type(clip_model.dtype)

        # copy the learnable context
        # cls_prompt_token_embedding[:, 1: len(self.soft_embeddings) + 1, :] = self.soft_embeddings.type(self.clip_model.dtype)
        suffix = cls_prompt_token_embedding[:, 1 + self.ctx_len:, :]
        return suffix

    def construct_cls_prompt_avg_element(self):
        """
        In CSP, 83 training pair IDX
        1. extract attr and obj
        2. replace the emebdding
        Can we change to all pairs?
        """

        # prefix: sos embedding
        prefix = self.token_prefix
        soft_prompt_embedding = self.soft_prompt_embedding
        suffix = self.token_avg_ele_suffix

        if soft_prompt_embedding.dim() == 2:
            soft_prompt_embedding = soft_prompt_embedding.unsqueeze(0).expand(self.cls_num, -1, -1)

        # simple way:
        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    soft_prompt_embedding,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        return prompts



    def construct_cls_prompt_raw_string(self):
        soft_prompt_embedding = self.soft_prompt_embedding
        if soft_prompt_embedding.dim() == 2:
            soft_prompt_embedding = soft_prompt_embedding.unsqueeze(0).expand(self.cls_num, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_str_suffix


        # -------------------------
        # cls_token_pos matters
        # -------------------------
        # simple way:
        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    soft_prompt_embedding,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.ctx_len // 2
            prompts = []
            for i in range(self.cls_num):
                name_len = self.cls_name_len[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = soft_prompt_embedding[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = soft_prompt_embedding[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.cls_num):
                name_len = self.cls_name_len[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = soft_prompt_embedding[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


    def forward(self):
        """
        1. pair_idx --> attr_idx and obj_idx, and replace embedding by element embedding
        2. raw, but not working in
        """
        if self.use_avg_embedding:
            return self.construct_cls_prompt_avg_element()
        else:
            return self.construct_cls_prompt_raw_string()