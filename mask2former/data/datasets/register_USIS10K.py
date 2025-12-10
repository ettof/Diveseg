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

def register_USIS10K_instance_dataset():
    classes = ['wrecks/ruins', 'fish', 'reefs', 'aquatic plants', 'human divers', 'robots', 'sea-floor']
    class_agnostic = ["foreground"]

    # multi-class data path
    train_json = "data/USIS10K/multi_class_annotations/multi_class_train_annotations.json"
    train_img_root = "data/USIS10K/train"
    val_json   = "data/USIS10K/multi_class_annotations/multi_class_val_annotations.json"
    val_img_root = "data/USIS10K/val"
    test_json  = "data/USIS10K/multi_class_annotations/multi_class_test_annotations.json"
    test_img_root = "data/USIS10K/test"

    # class-agnostic data path
    ca_train_json = "data/USIS10K/foreground_annotations/foreground_train_annotations.json"
    ca_train_img_root = "data/USIS10K/train"
    ca_val_json   = "data/USIS10K/foreground_annotations/foreground_val_annotations.json"
    ca_val_img_root = "data/USIS10K/val"
    ca_test_json  = "data/USIS10K/foreground_annotations/foreground_test_annotations.json"
    ca_test_img_root = "data/USIS10K/test"

    # Register multi-class training/validation/test sets
    DatasetCatalog.register(
        "USIS10K_train",
        lambda: custom_dataset_loader(train_json, train_img_root)
    )
    MetadataCatalog.get("USIS10K_train").set(
        json_file=train_json,
        image_root=train_img_root,
        evaluator_type="coco",
        thing_classes=classes,
        thing_dataset_id_to_contiguous_id={i+1: i for i in range(len(classes))}
    )

    DatasetCatalog.register(
        "USIS10K_val",
        lambda: custom_dataset_loader(val_json, val_img_root)
    )
    MetadataCatalog.get("USIS10K_val").set(
        json_file=val_json,
        image_root=val_img_root,
        evaluator_type="coco",
        thing_classes=classes,
        thing_dataset_id_to_contiguous_id={i+1: i for i in range(len(classes))}
    )

    DatasetCatalog.register(
        "USIS10K_test",
        lambda: custom_dataset_loader(test_json, test_img_root)
    )
    MetadataCatalog.get("USIS10K_test").set(
        json_file=test_json,
        image_root=test_img_root,
        evaluator_type="coco",
        thing_classes=classes,
        thing_dataset_id_to_contiguous_id={i+1: i for i in range(len(classes))}
    )

    # Register class-agnostic training/validation/test sets
    DatasetCatalog.register(
        "USIS10K_Class_Agnostic_train",
        lambda: custom_dataset_loader(ca_train_json, ca_train_img_root)
    )
    MetadataCatalog.get("USIS10K_Class_Agnostic_train").set(
        json_file=ca_train_json,
        image_root=ca_train_img_root,
        evaluator_type="coco",
        thing_classes=class_agnostic,
        thing_dataset_id_to_contiguous_id={1: 0}
    )

    DatasetCatalog.register(
        "USIS10K_Class_Agnostic_val",
        lambda: custom_dataset_loader(ca_val_json, ca_val_img_root)
    )
    MetadataCatalog.get("USIS10K_Class_Agnostic_val").set(
        json_file=ca_val_json,
        image_root=ca_val_img_root,
        evaluator_type="coco",
        thing_classes=class_agnostic,
        thing_dataset_id_to_contiguous_id={1: 0}
    )

    DatasetCatalog.register(
        "USIS10K_Class_Agnostic_test",
        lambda: custom_dataset_loader(ca_test_json, ca_test_img_root)
    )
    MetadataCatalog.get("USIS10K_Class_Agnostic_test").set(
        json_file=ca_test_json,
        image_root=ca_test_img_root,
        evaluator_type="coco",
        thing_classes=class_agnostic,
        thing_dataset_id_to_contiguous_id={1: 0}
    )

register_USIS10K_instance_dataset()
