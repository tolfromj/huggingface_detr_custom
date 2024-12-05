import os

import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
import argparse
from PIL import Image
import albumentations as A
from transformers import AutoImageProcessor

from models import get_model


def get_file_extension(filename):
    _, extension = os.path.splitext(filename)
    return extension.lstrip(".")


def main(model_name, ckpt_path, result_dir, img_dir, threshold=0.5):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = get_model(model_name, device).to(device)
    model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
    model.eval()

    # make a result directory
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # load img_files
    img_files = [img for img in os.listdir(img_dir) if ".jpg" in img]
    img_files.sort()

    # load model and image_processor
    image_processor = AutoImageProcessor.from_pretrained(model_name)

    # transform
    # test_augments_transform = A.Compose(
    #     [
    #         A.Resize(height=800, width=800, p=1.0),
    #     ]
    # )

    # make result files
    with torch.no_grad():
        for img_file in tqdm(img_files):
            # result directory
            img_ext = get_file_extension(img_file)
            txt_filename = img_file.replace(img_ext, "txt")
            txt_file = os.path.join(result_dir, txt_filename)

            # load a image
            img_path = os.path.join(img_dir, img_file)
            image = Image.open(img_path)
            width, height = image.size
            image = np.array(image)
            # image = test_augments_transform(image=image)["image"]

            # image pre-processing
            input = image_processor(images=image)
            pixel_values = torch.tensor(input["pixel_values"]).to(device)
            target = {"pixel_mask": torch.tensor(input["pixel_mask"]).to(device)}

            # inference
            outputs = model(pixel_values, target)  # pixel_value, pixel_mask
            logits = outputs["logits"]
            pred_boxes = outputs["pred_boxes"]

            # post-processing
            prob = F.softmax(logits, -1)
            scores, labels = prob[..., :-1].max(-1)  # cx, cy, w, h : [0,1]
            # scale_fct = torch.tensor([width, height, width, height]).to(device)
            # boxes = pred_boxes * scale_fct

            # filtering with threshold
            mask = (scores > threshold).to(device)
            scores = scores[mask]
            labels = labels[mask]
            boxes = pred_boxes[mask]

            num_obj = len(scores)

            with open(txt_file, "w") as f:
                for score, label, box in zip(scores, labels, boxes):
                    f.write(
                        "%d %lf %lf %lf %lf %lf\n"
                        % (label, box[0], box[1], box[2], box[3], score)
                    )

            if num_obj == 0:
                print(f"'{img_file}' has no boxes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/detr-resnet-101")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/workspace/traffic_light/output/facebook/detr-resnet-101/v1/best_23.pth",
    )
    parser.add_argument(
        "--result_dir", type=str, default="./Result/detect/predictions/v1"
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="/workspace/traffic_light/data/detection/test/images",
    )
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    threshold = 0.5
    main(args.model_name, args.ckpt_path, args.result_dir, args.img_dir, args.threshold)
