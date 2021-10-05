from .dota_obb import DotaOBBDataset
from .registry import DATASETS


@DATASETS.register_module
class UCAS_AODDataset(DotaOBBDataset):
    CLASSES = ('airplane', 'car')
