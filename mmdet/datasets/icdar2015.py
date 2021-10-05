from .dota_obb import DotaOBBDataset
from .registry import DATASETS


@DATASETS.register_module
class ICDAR2015Dataset(DotaOBBDataset):
    CLASSES = ('text',)