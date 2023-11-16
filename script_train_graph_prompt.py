#
import argparse
import os
import pickle
import pprint

import numpy as np
import torch
import tqdm
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import StepLR

import sys

sys.path.append('/tank/space/xugy07/VLPrompt')

from Models.CspModel.datasets.composition_dataset import CompositionDataset
from Models.CspModel.datasets.read_datasets import DATASET_PATHS
from Models.CspModel.models.compositional_modules import get_model
from Models.CspModel.utils import set_seed

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


from ProjUtils.Constants import SavePath
def modify_save_path(config):
    dataset = config.dataset
    exp = config.experiment_name
    ori_save_path = config.save_path
    weight_str = f"Attr_{config.attr_weight}_Obj_{config.obj_weight}"
    new_save_path = os.path.join(SavePath, dataset, exp, config.clip_model, weight_str)
    config.save_path = new_save_path



def train_model(model, optimizer, train_dataset, config, device,
                attr_weight = 1.0, obj_weight = 1.0):
    """
    shared by all

    -----------------------

    Function to train the model to predict attributes with cross entropy loss.

    Args:
        model (nn.Module): the model to compute the similarity score with the images.
        optimizer (nn.optim): the optimizer with the learnable parameters.
        train_dataset (CompositionDataset): the train dataset
        config (argparse.ArgumentParser): the config
        device (...): torch device

    Returns:
        tuple: the trained model (or the best model) and the optimizer
    """
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True
    )

    model.train()

    loss_fn = CrossEntropyLoss()

    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    train_pair_sep_idx = torch.tensor([(attr2idx[attr], obj2idx[obj])for attr, obj in train_dataset.train_pairs]).to(device)
    train_attr_idx = torch.tensor(list(range(len(attr2idx)))).to(device)
    train_obj_idx = torch.tensor(list(range(len(obj2idx)))).to(device)


    train_losses = []
    train_pair_loss_list = []
    train_attr_loss_list = []
    train_obj_loss_list = []
    train_total_loss_list = []

    torch.autograd.set_detect_anomaly(True)

    # -----------------
    # prepare the
    # -----------------
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

    save_soft_embeddings(model, config, epoch=1)

    for i in range(config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )

        epoch_train_losses = []
        for batch_idx, batch in enumerate(train_dataloader):
            batch_img, batch_attr, batch_obj, batch_train_pair_target = batch[0], batch[1], batch[2], batch[3]
            batch_attr = batch_attr.to(device)
            batch_obj = batch_obj.to(device)
            batch_train_pair_target = batch_train_pair_target.to(device)
            batch_img = batch_img.to(device)

            # img feats
            batch_feat = model.encode_image(batch_img)

            # go through model
            if config.experiment_name in ["csp", "pair_soft_emb_and_soft_prompt"]:
                pair_logit = model(batch_feat, train_pair_sep_idx)
                attr_logit, obj_logit = None, None
            elif config.experiment_name in ['sep_soft_prompt', 'sep_soft_emb', 'shared_soft_prompt']:
                pair_logit, attr_logit, obj_logit = model(batch_feat, train_pair_sep_idx, train_attr_idx, train_obj_idx)

            attr_loss, obj_loss = torch.tensor(0.), torch.tensor(0.)
            pair_loss = loss_fn(pair_logit, batch_train_pair_target)
            if attr_logit is not None:
                attr_loss = loss_fn(pair_logit, batch_attr)
            if obj_logit is not None:
                obj_loss = loss_fn(pair_logit, batch_obj)

            total_loss = pair_loss + attr_loss * attr_weight + obj_loss * obj_weight

            # normalize loss to account for batch accumulation
            total_loss = total_loss / config.gradient_accumulation_steps

            # backward pass
            total_loss.backward()

            # weights update
            if ((batch_idx + 1) % config.gradient_accumulation_steps == 0) or (batch_idx + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()

            epoch_train_losses.append(total_loss.item())
            #progress_bar.set_postfix(
            #    {"train loss": np.mean(epoch_train_losses[-50:])}
            #)

            print("Loss for pair, attr obj:", pair_loss.item(), attr_loss.item(), obj_loss.item(), total_loss.item() * config.gradient_accumulation_steps)
            # print("LR:", optimizer.state_dict()['param_groups'][0]['lr'])
            train_pair_loss_list.append(pair_loss.item())
            train_attr_loss_list.append(attr_loss.item())
            train_obj_loss_list.append(obj_loss.item())
            train_total_loss_list.append(total_loss.item())

            # update progress
            progress_bar.update()


        progress_bar.close()
        progress_bar.write(
            f"epoch {i +1} train loss {np.mean(epoch_train_losses)}"
        )
        train_losses.append(np.mean(epoch_train_losses))

        if (i + 1) % config.save_every_n == 0:
            save_soft_embeddings(model, config, epoch=i + 1)

        # update scheduler
        # scheduler.step()

    dict_TrainStat = {
        "pair_loss": train_pair_loss_list,
        "attr_loss": train_attr_loss_list,
        "obj_loss": train_obj_loss_list,
        "total_loss": train_total_loss_list
    }
    with open(os.path.join(config.save_path, "train_stats.pkl"), "wb") as fp:
        pickle.dump(dict_TrainStat, fp)

    return model, optimizer


def save_soft_embeddings(model, config, epoch=None):
    """Function to save soft embeddings.

    Args:
        model (nn.Module): the CSP/COOP module
        config (argparse.ArgumentParser): the config
        epoch (int, optional): epoch number for the soft embedding.
            Defaults to None.

    -----------------------------------


    could be:
    1. soft_emb
    2. soft_prompt
    """
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    # save the soft embedding
    with torch.no_grad():
        if epoch:
            soft_emb_path = os.path.join(
                config.save_path, f"soft_embeddings_epoch_{epoch}.pt"
            )
        else:
            soft_emb_path = os.path.join(
                config.save_path, "soft_embeddings.pt"
            )

        if config.experiment_name in ["sep_soft_emb", "csp"]:
            torch.save(
                {
                    "soft_embed": model.soft_element_embeddings
                },
                soft_emb_path
            )

        if config.experiment_name in ["coop"]:
            torch.save(
                {
                    "soft_prompt": model.soft_prompt_embeddings
                },
                soft_emb_path
            )

        if config.experiment_name in ["sep_soft_prompt"]:
            torch.save(
                {
                    "pair_soft_prompt": model.pair_soft_prompt_emb,
                    "attr_soft_prompt": model.attr_soft_prompt_emb,
                    "obj_soft_prompt": model.obj_soft_prompt_emb
                },
                soft_emb_path
            )

        if config.experiment_name in ["sep_soft_emb_and_soft_prompt", "pair_soft_emb_and_soft_prompt"]:
            torch.save(
                {
                    "soft_prompt": model.soft_prompt_embeddings,
                    "soft_embed": model.soft_element_embeddings
                },
                soft_emb_path
            )

        if config.experiment_name in ['shared_soft_prompt']:
            torch.save(
                {
                    "soft_prompt": model.shared_soft_prompt_emb,
                },
                soft_emb_path
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        help="name of the experiment",
        type=str,
        default="coop"
    )
    parser.add_argument("--dataset", help="name of the dataset", type=str, default="ut-zappos")
    parser.add_argument(
        "--lr", help="learning rate", type=float, default=5e-05
    )
    parser.add_argument(
        "--weight_decay", help="weight decay", type=float, default=1e-05
    )
    parser.add_argument(
        "--clip_model", help="clip model type", type=str, default="ViT-L/14"
    )
    parser.add_argument(
        "--epochs", help="number of epochs", default=10, type=int
    )
    parser.add_argument(
        "--train_batch_size", help="train batch size", default=64, type=int
    )
    parser.add_argument(
        "--eval_batch_size", help="eval batch size", default=1024, type=int
    )
    parser.add_argument(
        "--evaluate_only",
        help="directly evaluate on the" "dataset without any training",
        action="store_true",
    )
    parser.add_argument(
        "--context_length",
        help="sets the context length of the clip model",
        default=77,
        type=int,
    )
    parser.add_argument(
        "--soft_emb_dropout",
        help="add dropout to attributes",
        type=float,
        default=0.0,
    )
    parser.add_argument("--save_path", help="save path", type=str, default='/tank/space/xugy07/VLPrompt/ckpt')
    parser.add_argument(
        "--save_every_n",
        default=1,
        type=int,
        help="saves the model every n epochs; "
        "this is useful for validation/grid search",
    )
    parser.add_argument(
        "--save_model",
        help="indicate if you want to save the model state dict()",
        action="store_true",
    )
    parser.add_argument("--seed", help="seed value", default=0, type=int)
    parser.add_argument("--attr_weight", default=1.0, type=float)
    parser.add_argument("--obj_weight", default=1.0, type=float)

    parser.add_argument(
        "--gradient_accumulation_steps",
        help="number of gradient accumulation steps",
        default=1,
        type=int
    )

    config = parser.parse_args()

    # set the seed value
    set_seed(config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("training details")
    pprint.pprint(config)

    # modify save path
    modify_save_path(config)

    # This should work for mit-states, ut-zappos, and maybe c-gqa.
    dataset_path = DATASET_PATHS[config.dataset]

    # init train dataset
    train_dataset = CompositionDataset(dataset_path,
                                       phase='train',
                                       split='compositional-split-natural')

    # -----------------
    # model:
    #   1. coop
    #   2. csp
    #   3. mix-csp
    # opt:
    #   promt_opt
    # -----------------
    model, optimizer = get_model(train_dataset, config, device)

    print("model dtype", model.dtype)
    print("soft embedding dtype", model.soft_element_embeddings.dtype)

    if not config.evaluate_only:
        model, optimizer = train_model(
            model,
            optimizer,
            train_dataset,
            config,
            device,
            config.attr_weight,
            config.obj_weight
        )


    save_soft_embeddings(
        model,
        config,
    )

    with open(os.path.join(config.save_path, "config.pkl"), "wb") as fp:
        pickle.dump(config, fp)

    print("done!")
