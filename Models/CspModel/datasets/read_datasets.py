import os
import socket
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

HostName = socket.gethostname()

if HostName.startswith("turing"):
    DATASET_PATHS = {
        "mit-states": '/egr/research-hlr/xugy07/mitstates',
        "ut-zappos": '/egr/research-hlr/xugy07/zappos',
        "cgqa": "/egr/research-hlr/xugy07/cgqa"
    }
if HostName.startswith("ladyada"):
    DATASET_PATHS = {
        "mit-states": '/egr/research-hlr/xugy07/mitstates',
        "ut-zappos": '/egr/research-hlr/xugy07/zappos',
        "cgqa": "/egr/research-hlr/xugy07/cgqa"
    }
else:
    DATASET_PATHS = {
        "mit-states": '/home/xu/CompDataset/mitstates',
        "ut-zappos": '/home/xu/CompDataset/zappos',
        "cgqa": '/home/xu/CompDataset/cgqa'
    }