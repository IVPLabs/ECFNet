import os
import argparse
import numpy as np
import torch
from PIL import Image
from ultralytics import SAM
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def apply_mask_to_image(image_path, mask, color=(0, 255, 0), alpha=0.5):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image from {image_path}")

    mask_rgb = (np.stack([mask] * 3, axis=-1) * np.array(color, dtype=np.uint8)).astype(np.uint8)

    overlay = cv2.addWeighted(image, 1 - alpha, mask_rgb, alpha, 0)
    return overlay

def accumulate_masks(mask_data, image_shape):
    accumulated_mask = np.zeros(image_shape, dtype=np.float32)


    for mask in mask_data:
        accumulated_mask += mask.astype(np.float32)

    accumulated_mask = np.clip(accumulated_mask, 0, 1)

    return accumulated_mask

def read_yolo_labels(file_path, img_width, img_height):
    boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            class_id, x_center, y_center, width, height = map(float, values)
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            boxes.append([x_min, y_min, x_max, y_max])
    return boxes


def save_combined_mask(image_path, label_path, model, output_dir, img_shape, tau=10):
    image = Image.open(image_path).convert("RGB")
    img_width, img_height = image.size
    #print(img_width, img_height)

    boxes = read_yolo_labels(label_path, img_width, img_height)

    results = model(np.array(image), bboxes=boxes)
    mask_data = results[0].masks.data.cpu().numpy()

    sam_mask = accumulate_masks(mask_data, (512, 640))
    print(sam_mask.shape)

    #sam_mask = accumulate_masks(mask_data, (512, 640))

    masked_image = apply_mask_to_image(image_path, sam_mask)
    #


    final_mask = np.zeros(img_shape, dtype=np.float32)

    #
    gaussian_mask = generate_gaussian_masks(boxes, img_shape, tau)

    combined_mask = np.where(sam_mask == 0, gaussian_mask * sam_mask, gaussian_mask)
    final_mask = np.maximum(final_mask, combined_mask)

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + ".png")

    cv2.imwrite(output_path, sam_mask)
    print(f"标签已保存到 {output_path}")



def generate_gaussian_masks(boxes, img_shape, tau=0.5):
    h_img, w_img = img_shape
    gaussian_mask = np.zeros((h_img, w_img), dtype=np.float32)

    for box in boxes:
        x_min, y_min, x_max, y_max = box


        xc, yc = (x_min + x_max) / 2, (y_min + y_max) / 2
        w, h = x_max - x_min, y_max - y_min

        x_grid, y_grid = np.meshgrid(np.arange(w_img), np.arange(h_img))

        mask = np.exp(-0.5 * (((x_grid - xc) / (w / 2)) ** 2 + ((y_grid - yc) / (h / 2)) ** 2) * np.log(tau))

        gaussian_mask = np.maximum(gaussian_mask, mask)


    print(f"高斯核的掩码形状为 {gaussian_mask.shape}")

    return gaussian_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SAM masks for a dataset split.")
    parser.add_argument('--dataset_path', type=str, default='./data', help='dataset root')
    parser.add_argument('--weight_path', type=str, default='./weights/sam_b.pt', help='SAM checkpoint')
    parser.add_argument('--split', type=str, default='val', help='dataset split')
    args = parser.parse_args()

    split_path = os.path.join(args.dataset_path, args.split)
    image_dir = os.path.join(split_path, "images")
    label_dir = os.path.join(split_path, "labels")
    output_dir = os.path.join(split_path, "masks")

    model = SAM(args.weight_path)

    for img_file in sorted(os.listdir(image_dir)):
        if img_file.endswith(".png") or img_file.endswith(".jpg"):

            image_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")

            if not os.path.exists(label_path):
                print(f"标签文件不存在，跳过: {label_path}")
                continue

            save_combined_mask(image_path, label_path, model, output_dir, (512, 640))

