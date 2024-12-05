import torch
import timm

from torch import nn
from typing import Dict, List, Optional, Tuple, Union

import mlflow
from transformers import DetrConfig, DetrModel#, DetrForObjectDetection
from transformers.loss.loss_for_object_detection import ImageLoss, HungarianMatcher
from transformers.models.detr.modeling_detr import (
    # DetrLoss,
    DetrMLPPredictionHead,
    DetrPreTrainedModel,
    # DetrHungarianMatcher,
)

# from ultralytics import YOLO
from transformers import Swinv2Config, Swinv2Model

# Initializing a Swinv2 microsoft/swinv2-tiny-patch4-window8-256 style configuration
configuration = Swinv2Config()

# Initializing a model (with random weights) from the microsoft/swinv2-tiny-patch4-window8-256 style configuration
model = Swinv2Model(configuration)

# Accessing the model configuration
configuration = model.config


class HuggingfaceSwinV2Model(nn.Module):
    def __init__(self, chkpt, device):
        super().__init__()
        self.configuration = Swinv2Config()
        self.model = Swinv2Model(configuration)
        configuration = model.config


def get_model(model_name: str, device: str) -> nn.Module:
    if model_name in ["facebook/detr-resnet-50", "facebook/detr-resnet-101"]:
        config = DetrConfig()
        return HuggingfaceDetrModel(config, model_name, device)


class HuggingfaceDetrModel(DetrPreTrainedModel):
    def __init__(self, config: DetrConfig, ckpt, device):
        super().__init__(config)
        self.ckpt = ckpt
        self.model = DetrModel.from_pretrained(self.ckpt)

        self.class_labels_classifier = nn.Linear(
            in_features=256, out_features=15  # , bias=True
        )
        self.bbox_predictor = DetrMLPPredictionHead(
            input_dim=256, hidden_dim=256, output_dim=4, num_layers=3
        )
        self.learning_rate = 1e-6
        self.optimizer = self.get_optimizer()
        mlflow.log_param("learning_rate", self.learning_rate)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        targets: dict,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict:
        """
        pixel_values [B, C, H, W] Normalize IageNetv1 mean std -~+
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):

        Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
        following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
        respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
        in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        DETR resizes the input images such that the shortest side is at least a certain amount of pixels while the longest
        is at most 1333 pixels. At training time, scale augmentation is used such that the shortest side is randomly set to
        at least 480 and at most 800 pixels. At inference time, the shortest side is set to 800. One can use
        DetrImageProcessor to prepare images (and optional annotations in COCO format) for the model. Due to this resizing,
        images in a batch can have different sizes. DETR solves this by padding images up to the largest size in a batch,
        and by creating a pixel mask that indicates which pixels are real/which are padding. Alternatively, one can also
        define a custom collate_fn in order to batch images together, using ~transformers.DetrImageProcessor.pad_and_create_pixel_mask.
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):

        Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
        following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
        respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
        in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        Returns:{
                "loss":  loss,
                "loss_dict":  loss_dict,
                "logits":  logits,
                "pred_boxes":  pred_boxes,
                "outputs": outputs,
                }
        """
        pixel_mask = targets["pixel_mask"]
        if "labels" in targets.keys():
            labels = targets["labels"]
        else:
            labels = None
        # First, sent images through DETR base model to obtain encoder + decoder outputs
        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # class logits + predicted bounding boxes
        logits = self.class_labels_classifier(sequence_output)
        pred_boxes = self.bbox_predictor(sequence_output).sigmoid()

        loss, loss_dict = None, None  # , auxiliary_outputs = None, None, None
        if labels is not None:
            # First: create the matcher
            matcher = HungarianMatcher(
                class_cost=self.config.class_cost,
                bbox_cost=self.config.bbox_cost,
                giou_cost=self.config.giou_cost,
            )
            # Second: create the criterion
            losses = ["labels", "boxes", "cardinality"]
            criterion = ImageLoss(
                matcher=matcher,
                num_classes=14,
                eos_coef=self.config.eos_coefficient,
                losses=losses,
            )
            criterion.to(self.device)
            # Third: compute the losses, based on outputs and labels
            outputs_loss = {}
            outputs_loss["logits"] = logits
            outputs_loss["pred_boxes"] = pred_boxes

            if torch.isnan(pred_boxes[0][0][0]):
                print(labels[0]["image_id"])
            # if self.config.auxiliary_loss:
            #     intermediate = outputs.intermediate_hidden_states if return_dict else outputs[4]
            #     outputs_class = self.class_labels_classifier(intermediate)
            #     outputs_coord = self.bbox_predictor(intermediate).sigmoid()
            #     auxiliary_outputs = self._set_aux_loss(outputs_class, outputs_coord)
            #     outputs_loss["auxiliary_outputs"] = auxiliary_outputs

            loss_dict = criterion(outputs_loss, labels)

            # Fourth: compute total loss, as a weighted sum of the various losses
            weight_dict = {"loss_ce": 1, "loss_bbox": self.config.bbox_loss_coefficient}
            weight_dict["loss_giou"] = self.config.giou_loss_coefficient
            # if self.config.auxiliary_loss:
            #     aux_weight_dict = {}
            #     for i in range(self.config.decoder_layers - 1):
            #         aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            #     weight_dict.update(aux_weight_dict)
            loss = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            )

        return {
            "loss": loss,
            "loss_dict": loss_dict,  # loss_dict : 'loss_ce', 'loss_bbox', 'loss_giou', 'cardinality_error'
            "logits": logits,
            "pred_boxes": pred_boxes,
            "outputs": outputs,  # 'logits', 'pred_boxes', 'last_hidden_state', 'encoder_last_hidden_state'
        }

    def get_optimizer(self, optim="sgd"):
        """
        Optimizer 선언
        """
        lr = self.learning_rate  # learning_rate
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if "bias" in key:
                    params += [{"params": [value], "lr": lr * 2, "weight_decay": 0}]
                else:
                    params += [{"params": [value], "lr": lr, "weight_decay": 0.0005}]
        if optim == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), momentum=0.9)
        # elif optim == ''
        return optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= decay
        return self.optimizer


# class UltralyticsModel:
#     def __init__(self):
#         self.model = YOLO("yolov10s.pt")

#     def forward(self, images: torch.Tensor, targets: list[dict] = None):
#         self.model(images)
