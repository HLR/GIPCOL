"""
Just keep it simple
1. not hierarchical


in coop:
SimpleTrainer --> TrainerX --> CoOp
"""
import os
import numpy as np
import torch
from torch.nn import functional as F
import Models.CoopModel.clip.clip as clip
from Models.CoopModel.PromptCompCptModel import PromptCompCptModel
from Models.CoopModel.Optimizer import build_optimizer
from Models.CoopModel.Scheduler import build_lr_scheduler
from DataSet.AttrOpCompDataSet import CompositionDatasetActivations
from DataSet.CSPCompDataset import CompositionDataset
from tqdm import tqdm
from Evaluator.Metrics import compute_accuracy
from ProjUtils.CkptUtils import CheckpointerFromCfg
from ProjUtils.meters import MetricMeter
from torch.utils.tensorboard import SummaryWriter

def load_clip_to_cpu(cfg):
    """
    clip is a file (python module) including
    1. global variables
    2. class definiton
    """
    # could be RN50, RN101, RN50*4, RN50*16, ViT-B/32, ViT-B/16
    backbone_name = cfg.CLIP.ImgBackBone
    # RN50 --> url
    url = clip._MODELS[backbone_name]
    # download the model
    model_path = clip._download(url)

    # --------------------
    # load the model
    # --------------------
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    # ----------------
    # two ways:
    # 1. using dict directly
    # 2. using model's state_dict
    # ----------------
    model = clip.build_model(state_dict or model.state_dict())

    return model



class PromptCompTrainer():

    def __init__(self, cfg_node):
        """
        at least 2 things should be in the trainer:
        1. model
        2. optimizer (maybe scheduler)
        3. data_loader
        4. evaluator
        """
        # setting check pointere

        # setting device
        if torch.cuda.is_available() and cfg_node.UseCUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.use_neural_split = cfg_node.UseNaturalSplit

        # training parameters
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg_node.Optim.MAX_EPOCH
        self.output_dir = cfg_node.output_dir

        # cfgNode
        self.cfg_node = cfg_node

        # -----------------------
        # build all the things in the trainer:
        # 1. call from init
        # 2. then init
        # -----------------------
        # build data_loader
        self.build_data_loader()
        self.batch_num = len(self.trainloader)
        # build model
        self.build_prompt_model(self.trainloader.dataset)
        # build evaluator
        self.evaluator = None
        # init best result
        self.best_result = -np.inf


        # --------------------
        # add two things:
        # 1. logger
        # 2. checkpointer
        # --------------------
        # ckpt
        self.ckpt = CheckpointerFromCfg(
            self.cfg_node,
            self.prompt_model,
            None,       # not saving optimizer
            None,       # not saving scheduler
            self.cfg_node.ckpt_dir,
            save_to_disk=True
        )
        # meter
        self.meters = MetricMeter()
        # logger
        self.writer = SummaryWriter()


    def build_prompt_model(self, dataset):
        """
        CLIP is the model we want to explore the knowledge.
        1. build CLIP model
        2. init prompt model using clip:
            2.1 init txt encoder
            2.2 init img encoder
            2.3 define prompt learner ( )
        """
        # step 1: load attr and obj list from dataset
        attr_cls_list, obj_cls_list, pair_cls_list, train_pair_cls_list\
            = dataset.attr_str_list, dataset.obj_str_list, dataset.all_pair_str_list, dataset.train_pair_str_list

        # step 2: load pre-trained clip
        print(f"Loading CLIP (backbone: {self.cfg_node.CLIP.ImgBackBone})")
        clip_model = load_clip_to_cpu(self.cfg_node)
        if self.cfg_node.Trainer.Prec == "fp32" or self.cfg_node.Trainer.Prec == "amp":
            # CLIP's default precision is fp16
            clip_model.float()


        # step 3:
        self.train_pair_sep_idx = self.build_pair_sep_idx(dataset)

        # step 4: using the pretrained clip to initilize the prompt model
        #
        self.prompt_model = PromptCompCptModel(self.cfg_node, attr_cls_list, obj_cls_list, train_pair_cls_list, clip_model, self.train_pair_sep_idx)

        # ---------------------------------
        # step 5: freeze clip
        # attr_prompt_learner.ctx
        # obj_prompt_learner.ctx
        # pair_prompt_learner.ctx
        # ----------------------------------
        #
        # img_encoder.conv1.weight
        # img_encoder.ln_pre.weight
        # img_encoder.ln_pre.bias
        # img_encoder.transformer.resblocks.0.attn.in_proj_weight
        #
        # weight and bias are important
        #
        # -----------------------------------
        #
        # txt_encoder
        # img_encoder
        # final_ln
        # (attr_prompt_learner): PromptLearner()
        # (obj_prompt_learner): PromptLearner()
        # (pair_prompt_learner): PromptLearner()
        #
        # -----------------------------------
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.prompt_model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        for name, param in self.prompt_model.named_parameters():
            if param.requires_grad == True:
                print("trainable params", name)

        # step 7: to GPU
        self.prompt_model.to(self.device)

        # step 8: directly build opt

        # 8.1: prepare the param_list
        if self.cfg_node.Trainer.CalAttrLoss and self.cfg_node.Trainer.CalObjLoss:
            all_soft_prompt = [self.prompt_model.pair_prompt_learner.soft_prompt_embedding] + [self.prompt_model.attr_prompt_learner.soft_prompt_embedding] + \
                              [self.prompt_model.obj_prompt_learner.soft_prompt_embedding]
        else:
            all_soft_prompt = self.prompt_model.pair_prompt_learner.soft_prompt_embedding

        # 8.2 prepare the
        self.optim = torch.optim.Adam(
            [all_soft_prompt],
            lr=self.cfg_node.Optim.LR,
            weight_decay=self.cfg_node.Optim.WEIGHT_DECAY)

        # step 9: build scheduler
        # self.scheduler = build_lr_scheduler(self.optim, self.cfg_node.Optim)

        # step 10:
        self.scaler = None

        # ------------------------------
        # step 11: data parallel
        # Note that multi-gpu training could be slow because CLIP's size is big, which slows down the copy operation in DataParallel
        # ------------------------------
        """
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
        """


    def build_pair_sep_idx(self, train_dataset):
        attr2idx = train_dataset.dict_Attr2IDX
        obj2idx = train_dataset.dict_Obj2IDX
        train_pair_sep_idx = torch.tensor([(attr2idx[attr], obj2idx[obj]) for attr, obj in train_dataset.train_pair_str_list])
        return train_pair_sep_idx


    def build_data_loader(self):
        """
        copy from AttrAsOpt and CSP
        """
        if self.use_neural_split:
            # train
            trainset = CompositionDataset(root=self.cfg_node.DataSet.data_dir,
                                          phase='train',
                                          split='compositional-split-natural')
            self.trainloader = torch.utils.data.DataLoader(trainset,
                                                           batch_size=self.cfg_node.DataLoader.Train.batch_size,
                                                           shuffle=True,
                                                           num_workers=self.cfg_node.DataLoader.num_worker)
            # val
            valset = CompositionDataset(root=self.cfg_node.DataSet.data_dir,
                                        phase='val',
                                        split='compositional-split-natural')
            self.valloader = torch.utils.data.DataLoader(valset,
                                                         batch_size=self.cfg_node.DataLoader.Test.batch_size,
                                                         shuffle=False,
                                                         num_workers=self.cfg_node.DataLoader.num_worker)
            # test
            testset = CompositionDataset(root=self.cfg_node.DataSet.data_dir,
                                         phase='test',
                                         split='compositional-split-natural')
            self.testloader = torch.utils.data.DataLoader(testset,
                                                          batch_size=self.cfg_node.DataLoader.Test.batch_size,
                                                          shuffle=False,
                                                          num_workers=self.cfg_node.DataLoader.num_worker)
        else:
            # train
            trainset = CompositionDatasetActivations(root=self.cfg_node.DataSet.data_dir,
                                                     phase='train',
                                                     split='compositional-split')
            self.trainloader = torch.utils.data.DataLoader(trainset,
                                                           batch_size=self.cfg_node.DataLoader.Train.batch_size,
                                                           shuffle=True,
                                                           num_workers=self.cfg_node.DataLoader.num_worker)

            # test
            testset = CompositionDatasetActivations(root=self.cfg_node.DataSet.data_dir,
                                                    phase='test',
                                                    split='compositional-split')
            self.testloader = torch.utils.data.DataLoader(testset,
                                                          batch_size=self.cfg_node.DataLoader.Test.batch_size,
                                                          shuffle=False,
                                                          num_workers=self.cfg_node.DataLoader.num_worker)

    def train(self):
        """
        for batch:
            forward to get loss
            backward to update parameters
        """
        torch.autograd.set_detect_anomaly(True)
        self.prompt_model.train()

        for epoch_idx in range(self.max_epoch):
            for batch_idx, batch in tqdm(enumerate(self.trainloader), total=len(self.trainloader)):
                image, attr_label, obj_label, pair_label = batch[0].to(self.device), batch[1].to(self.device), \
                                                           batch[2].to(self.device), batch[3].to(self.device)

                # ---------------
                # forward
                # ---------------
                attr_logit, obj_logit, pair_logit = self.prompt_model(image)

                total_loss = 0.0
                if self.cfg_node.Trainer.CalAttrLoss:
                    attr_loss = F.cross_entropy(attr_logit, attr_label)
                    total_loss += attr_loss
                if self.cfg_node.Trainer.CalObjLoss:
                    obj_loss = F.cross_entropy(obj_logit, obj_label)
                    total_loss += obj_loss
                pair_loss = F.cross_entropy(pair_logit, pair_label)
                total_loss += pair_loss

                # ---------------
                # backward
                # ---------------
                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()

                print(self.optim.param_groups[0])

                # global step and save to tb_writer
                """
                train_summary = {
                    "attr_loss": attr_loss.item(),
                    "obj_loss": obj_loss.item(),
                    "pair_loss": pair_loss.item(),
                    "total_loss": total_loss.item(),
                    "attr_acc": compute_accuracy(attr_logit, attr_label)[0].item(),
                    "obj_acc": compute_accuracy(obj_logit, obj_label)[0].item(),
                    "pair_acc": compute_accuracy(pair_logit, pair_label)[0].item(),
                }
                self.meters.update(train_summary)
                global_step = epoch_idx * self.batch_num + batch_idx
                for name, meter in self.meters.meters.items():
                    self.writer.add_scalar(os.path.join(f"{self.cfg_node.log_dir}/train/" + name), meter.avg, global_step)
                self.writer.add_scalar(f"{self.cfg_node.log_dir}/train/lr", self.get_current_lr(), global_step)
                """
                print(total_loss.item(), self.get_current_lr())


            # ------------------------
            # post epoch operations
            # ------------------------
            # update optim after one epoch
            # self.scheduler.step()


    def get_current_lr(self):
        """return opt name"""
        return self.optim.param_groups[0]["lr"]


