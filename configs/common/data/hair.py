from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances
register_coco_instances("hair_train", {}, "/home/data2/dongtrinh/ViTDet/data/coco/annotations/instances_train.json", "/home/data2/dongtrinh/ViTDet/data/coco/train")
register_coco_instances("hair_test", {}, "/home/data2/dongtrinh/ViTDet/data/coco/annotations/instances_val.json", "/home/data2/dongtrinh/ViTDet/data/coco/val")


dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="hair_train"),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=(416, 448, 480, 512, 544, 576, 608, 640),
                sample_style="choice",
                max_size=640,
            ),
            L(T.RandomFlip)(horizontal=True),
        ],
        image_format="BGR",
        use_instance_mask=False,
        recompute_boxes=False
    ),
    total_batch_size=1,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="hair_test", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
        ],
        image_format="${...train.mapper.image_format}",
        use_instance_mask=False,
        recompute_boxes=False
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
