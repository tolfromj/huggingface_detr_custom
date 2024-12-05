import torch
import mlflow
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers.image_transforms import center_to_corners_format
from torch.nn.functional import softmax

batch_metrics = []
id2label = {
    0: "veh_go",
    1: "veh_goLeft",
    2: "veh_noSign",
    3: "veh_stop",
    4: "veh_stopLeft",
    5: "veh_stopWarning",
    6: "veh_warning",
    7: "ped_go",
    8: "ped_noSign",
    9: "ped_stop",
    10: "bus_go",
    11: "bus_noSign",
    12: "bus_stop",
    13: "bus_warning",
}
# def coco_to_pascal_bbox(boxes, orig_width, orig_height):
#     """
#     Convert bounding boxes from COCO format (x_min, y_min, width, height) in range [0, 1]
#     to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

#     Args:
#         boxes (torch.Tensor): Bounding boxes in YOLO format
#         width (int): width of original image size
#         height (int): height of original image size


#     Returns:
#         torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
#     """
#     x_min = boxes[0]
#     x_max = boxes[0] + boxes[2]
#     y_min = boxes[1]
#     y_max = boxes[1] + boxes[3]
#     if x_min < 0:
#         x_min = float(0)
#     if y_min < 0:
#         y_min = float(0)
#     if x_max > orig_width:
#         x_max = float(orig_width)
#     if y_max > orig_height:
#         y_max = float(orig_height)
# return [x_min, y_min, x_max, y_max]
def yolo_to_pascal_bbox(boxes, width, height, device):
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        width (int): width of original image size
        height (int): height of original image size

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
    """
    # convert center to corners format
    boxes = center_to_corners_format(boxes)

    # convert to absolute coordinates
    boxes = boxes * torch.tensor([[width, height, width, height]]).to(device)
    return boxes


def compute_metrics(scores, pred_boxes, labels, compute_result, device):
    """_summary_

    Args:
        scores (_type_): _description_
        pred_boxes (Tensor[[float]): cx,cy,w,h (yolo format)
        labels (_type_): _description_
        compute_result (_type_): _description_

    Returns:
        _type_: _description_
    """
    global batch_metrics, id2label

    image_sizes = []
    target = []
    for label in labels:

        image_sizes.append(label["orig_size"])
        width, height = label["orig_size"]
        denormalized_boxes = yolo_to_pascal_bbox(label["boxes"], width, height, device)
        target.append(
            {
                "boxes": denormalized_boxes,
                "labels": label[
                    "class_labels"
                ],  # 'class_labels' = tensor([0, 0], device='cuda:0')
            }
        )
    predictions = []
    for score, box, target_sizes in zip(scores, pred_boxes, image_sizes):
        # Extract the bounding boxes, labels, and scores from the model's output
        pred_scores = score[:, :-1]  # Exclude the no-object class
        pred_scores = softmax(pred_scores, dim=-1)
        width, height = target_sizes
        denormalized_pred_boxes = yolo_to_pascal_bbox(box, width, height, device)
        pred_labels = torch.argmax(pred_scores, dim=-1)

        # Get the scores corresponding to the predicted labels
        pred_scores_for_labels = torch.gather(
            pred_scores, 1, pred_labels.unsqueeze(-1)
        ).squeeze(-1)
        predictions.append(
            {
                "boxes": denormalized_pred_boxes,
                "scores": pred_scores_for_labels,
                "labels": pred_labels,
            }
        )

    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)

    if not compute_result:
        # Accumulate batch-level metrics
        batch_metrics.append({"preds": predictions, "target": target})
        return {}
    else:
        # Compute final aggregated metrics
        # Aggregate batch-level metrics (this should be done based on your metric library's needs)
        all_preds = []
        all_targets = []
        for batch in batch_metrics:
            all_preds.extend(batch["preds"])
            all_targets.extend(batch["target"])

        # Update metric with all accumulated predictions and targets
        metric.update(preds=all_preds, target=all_targets)
        metrics = metric.compute()

        # print(all_preds[0], all_targets[0])
        # Convert and format metrics as needed
        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        # print(all_preds[0], all_targets[0])
        # mar_100_per_class = metrics.pop("mar_100_per_class")
        total_map = metrics.pop("map")
        map_50 = metrics.pop("map_50")
        map_75 = metrics.pop("map_75")

        with mlflow.start_span(name="mAP", span_type="AP") as span:
            for class_id, class_map in zip(classes, map_per_class):
                class_name = (
                    id2label[class_id.item()]
                    if id2label is not None
                    else class_id.item()
                )
                span.set_inputs({"class_name": class_name})
                metrics[f"mAP_{class_name}"] = class_map
                span.set_outputs({"mAP": class_map})
                # print(span.span_type)

        # Round metrics for cleaner output
        # metrics = {k: round(v.item(), 4) for k, v in metrics.items()}
        metrics["mAP"] = total_map
        metrics["mAP_50"] = map_50
        metrics["mAP_75"] = map_75

        # Clear batch metrics for next evaluation
        batch_metrics = []

        return metrics
