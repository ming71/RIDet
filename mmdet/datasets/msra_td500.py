from .dota_obb import DotaOBBDataset
from .registry import DATASETS


@DATASETS.register_module
class MSRA_TD500Dataset(DotaOBBDataset):
    CLASSES = ('text',)