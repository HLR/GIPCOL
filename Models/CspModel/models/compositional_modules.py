import os

from Models.CspModel.models.coop import get_coop
from Models.CspModel.models.csp import get_csp, get_mix_csp
from Models.CspModel.models.SepSoftEmb import get_sep_soft_emb
from Models.CspModel.models.SepSoftEmbAndSoftPrompt import get_sep_soft_emb_and_soft_prompt
from Models.CspModel.models.PairSoftEmbAndSoftPrompt import get_pair_soft_emb_and_soft_prompt
from Models.CspModel.models.SepSoftPrompt import get_sep_soft_prompt
from Models.CspModel.models.SharedSoftPrompt import get_shared_soft_prompt
from Models.CspModel.models.GraphPrompt import get_graph_and_prompt

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def get_model(train_dataset, config, device):
    if config.experiment_name == "coop":
        return get_coop(train_dataset, config, device)

    elif config.experiment_name == "csp":
        return get_csp(train_dataset, config, device)

    # special experimental setup
    elif config.experiment_name == "mix_csp":
        return get_mix_csp(train_dataset, config, device)

    # added by myself
    elif config.experiment_name == "sep_soft_emb":
        """ Exp 1:  Element embedding should consistent with compositonal """
        return get_sep_soft_emb(train_dataset, config, device)

    elif config.experiment_name == "sep_soft_prompt":
        """ Exp 2: Post filter our prediction """
        return get_sep_soft_prompt(train_dataset, config, device)

    elif config.experiment_name == "sep_soft_emb_and_soft_prompt":
        """ Exp 3: Both comp soft emb and soft prompt """
        return get_sep_soft_emb_and_soft_prompt(train_dataset, config, device)

    elif config.experiment_name == "pair_soft_emb_and_soft_prompt":
        """ Exp 3: Both comp soft emb and soft prompt """
        return get_pair_soft_emb_and_soft_prompt(train_dataset, config, device)

    elif config.experiment_name == 'shared_soft_prompt':
        # one clip recognize all
        return get_shared_soft_prompt(train_dataset, config, device)

    elif config.experiment_name == 'graph_prompt':
        # graph embedding + CLIP
        return get_graph_and_prompt(train_dataset, config, device)


    else:
        raise NotImplementedError(
            "Error: Unrecognized Experiment Name {:s}.".format(
                config.experiment_name
            )
        )
