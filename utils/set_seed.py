# -----------------------------------------------------------------------------
# Autor: Paul Schreiber, Mai 2025
# Diese Datei enthält vollständig persönliche Implementierungen.
# -----------------------------------------------------------------------------
import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(13)