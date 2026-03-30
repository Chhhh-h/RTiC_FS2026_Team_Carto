from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MapBackgroundDataset(CustomDataset):
    """Map dataset for land cover classification.
    
    Classes: river, forest, lake, wetland, stream, building, road
    Annotation values: 0, 1, 2, 3, 4, 5, 6,7
    Ignore index: 255
    """
    CLASSES = ('background','river', 'forest', 'lake', 'wetland', 'stream', 'building', 'road')
    
    PALETTE = [
        [0, 0, 0],        # background - black
        [0, 0, 255],      # river - blue
        [0, 128, 0],      # forest - green
        [0, 255, 255],    # lake - cyan
        [128, 0, 128],    # wetland - purple
        [128, 255, 255],  # stream - light cyan
        [255, 0, 0],      # building - red
        [128, 128, 128]   # road - gray
    ]

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.png',
                 ann_dir=None,
                 seg_map_suffix='.tif',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 **kwargs):
        super(MapBackgroundDataset, self).__init__(
            pipeline=pipeline,
            img_dir=img_dir,
            img_suffix=img_suffix,
            ann_dir=ann_dir,
            seg_map_suffix=seg_map_suffix,
            split=split,
            data_root=data_root,
            test_mode=test_mode,
            ignore_index=ignore_index,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
