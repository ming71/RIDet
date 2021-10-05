from .dota_obb import DotaOBBDataset
from .registry import DATASETS


@DATASETS.register_module
class GF2020PlaneDataset(DotaOBBDataset):
    CLASSES = ('Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 
    'A220', 'A321', 'A330', 'A350', 'other', 'ARJ21')

@DATASETS.register_module
class GF2020ShipDataset(DotaOBBDataset):
    CLASSES = ('ship',)