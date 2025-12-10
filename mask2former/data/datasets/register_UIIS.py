from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json

def custom_dataset_loader(json_file, image_root):
    """
    """
    dataset_dicts = load_coco_json(json_file, image_root)
    for record in dataset_dicts:
        for ann in record.get("annotations", []):
            ann["category_id"] = ann["category_id"] - 1
    return dataset_dicts

def register_UIIS_instance_dataset():
    classes = [
        'fish',
        'reefs',
        'aquatic plants',
        'wrecks/ruins',
        'human divers',
        'robots',
        'sea-floor'
    ]

    train_json     = "data/UIIS/annotations/train.json"
    train_img_root = "data/UIIS/train"
    val_json       = "data/UIIS/annotations/val.json"
    val_img_root   = "data/UIIS/val"

    DatasetCatalog.register(
        "UIIS_train",
        lambda: custom_dataset_loader(train_json, train_img_root)
    )
    MetadataCatalog.get("UIIS_train").set(
        json_file=train_json,
        image_root=train_img_root,
        evaluator_type="coco",
        thing_classes=classes,
        thing_dataset_id_to_contiguous_id={i+1: i for i in range(len(classes))}
    )

    DatasetCatalog.register(
        "UIIS_val",
        lambda: custom_dataset_loader(val_json, val_img_root)
    )
    MetadataCatalog.get("UIIS_val").set(
        json_file=val_json,
        image_root=val_img_root,
        evaluator_type="coco",
        thing_classes=classes,
        thing_dataset_id_to_contiguous_id={i+1: i for i in range(len(classes))}
    )

# 执行注册
register_UIIS_instance_dataset()
