from typing import Callable, Tuple, Dict
from transformers import AutoImageProcessor
import torch


def get_collate(model_name: str) -> Callable:
    if model_name in ["facebook/detr-resnet-50", "facebook/detr-resnet-101"]:
        return detr_collate_fn
    else:
        return lambda x: tuple(zip(*x))


def detr_collate_fn(batch) -> Tuple[torch.Tensor, Dict]:
    checkpoint = "facebook/detr-resnet-50"  # self.get_model_checkpoint()
    # TODO: checkpoint 입력 구조 수정 필요. resnet-101도 입력받을 수 있게!
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1]["labels"] for item in batch]
    batch_dict = {}
    batch_dict["pixel_mask"] = encoding["pixel_mask"]
    batch_dict["labels"] = labels

    # image_id = [label["image_id"] for label in labels]
    # print("image_id", image_id)
    # for i in range(len(batch_dict["labels"])):
    #     for bbox_e in batch_dict["labels"][i]["boxes"]:
    #         print("collate_batch: ", bbox_e)
    # print("", batch["labels"])

    return encoding["pixel_values"], batch_dict


# def base_collate_fn (batch):
#     pixel_values = [item[0] for item in batch]
