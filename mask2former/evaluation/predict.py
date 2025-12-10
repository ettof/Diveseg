# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation.coco_evaluation import COCOEvaluator, _evaluate_predictions_on_coco
from detectron2.evaluation.fast_eval_api import COCOeval_opt
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table

class InstanceSegEvaluator(COCOEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.
    The metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means
    the metric cannot be computed (e.g. due to no predictions made).

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """
    def __init__(self, dataset_name, tasks=None, distributed=True, output_dir=None, *, kpt_oks_sigmas=()):
        super().__init__(dataset_name, tasks, distributed, output_dir, kpt_oks_sigmas=kpt_oks_sigmas)
        self._dataset_name = dataset_name
        self._output_dir = output_dir

    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        Additionally, save visualization of instance segmentation results and intermediate 'I' tensor as PNG images.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                assert category_id in reverse_id_mapping, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has class ids in {dataset_id_to_contiguous_id}."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

            # Save visualizations if output_dir is specified
            self._logger.info("Saving instance segmentation and intermediate 'I' visualizations to {} ...".format(self._output_dir))
            self.save_visualizations(predictions)

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions with official COCO API...")
        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                    img_ids=img_ids,
                    max_dets_per_image=self._max_dets_per_image,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res

    def save_visualizations(self, predictions):
        """
        Save instance segmentation predictions and intermediate 'I' tensor as PNG images.
        Different colors are used for different classes based on metadata.
        Only masks with confidence scores >= 0.5 are visualized.
        """
        dataset_dicts = DatasetCatalog.get(self._dataset_name)
        metadata = MetadataCatalog.get(self._dataset_name)
        for prediction in predictions:
            img_id = prediction["image_id"]
            # Find the image info from the dataset
            img_info = [d for d in dataset_dicts if d["image_id"] == img_id][0]
            img_path = img_info["file_name"]
            img = cv2.imread(img_path)
            if img is None:
                self._logger.warning(f"Failed to load image: {img_path}")
                continue

            # Initialize Visualizer with the image (convert BGR to RGB)
            visualizer = Visualizer(img[:, :, ::-1], metadata=metadata)
            instances = prediction["instances"]

            # Convert predictions to Instances object for visualization
            from detectron2.structures import Instances
            height, width = img.shape[:2]
            vis_instances = Instances(image_size=(height, width))
            masks = []
            classes = []
            scores = []

            for instance in instances:
                if "segmentation" in instance and instance.get("score", 1.0) >= 0.5:
                    mask = mask_util.decode(instance["segmentation"])
                    masks.append(mask)
                    classes.append(instance["category_id"])
                    scores.append(instance.get("score", 1.0))

            if masks:
                vis_instances.set("pred_masks", torch.tensor(np.array(masks), dtype=torch.bool))
                vis_instances.set("pred_classes", torch.tensor(classes, dtype=torch.int64))
                vis_instances.set("scores", torch.tensor(scores, dtype=torch.float32))

                # Draw instance predictions with class-specific colors
                vis = visualizer.draw_instance_predictions(vis_instances)
            else:
                vis = visualizer.output  # No masks, use plain image

            # Get the visualized image and convert back to BGR for saving
            vis_img = vis.get_image()[:, :, ::-1]
            save_path = os.path.join(self._output_dir, f"vis_seg_{img_id}.png")
            cv2.imwrite(save_path, vis_img)
            self._logger.info(f"Saved segmentation visualization to {save_path}")

            # Visualize and save the intermediate 'I' tensor
            if "intermediate_outputs" in prediction and "I" in prediction["intermediate_outputs"]:
                I_tensor = prediction["intermediate_outputs"]["I"]
                # Ensure I_tensor is 3-channel (C, H, W) and move to CPU
                if I_tensor.shape[0] != 3:
                    self._logger.warning(f"Expected 3-channel tensor for 'I', got shape {I_tensor.shape}")
                    continue
                I_tensor = I_tensor.permute(1, 2, 0).cpu().numpy()  # Convert to H, W, C
                # Normalize to [0, 255] for visualization
                I_tensor = (I_tensor - I_tensor.min()) / (I_tensor.max() - I_tensor.min() + 1e-8) * 255
                I_tensor = I_tensor.astype(np.uint8)
                # Save as PNG
                I_save_path = os.path.join(self._output_dir, f"vis_I_{img_id}.png")
                cv2.imwrite(I_save_path, I_tensor[:, :, ::-1])  # Convert RGB to BGR for OpenCV
                self._logger.info(f"Saved intermediate 'I' visualization to {I_save_path}")
            else:
                self._logger.warning(f"No 'I' tensor found in intermediate_outputs for image {img_id}")