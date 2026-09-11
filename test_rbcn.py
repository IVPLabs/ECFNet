import os
import argparse
import torch
from torch import nn
import numpy as np
from model.rbcn import RBCN
from PIL import Image
import cv2
import time


torch.set_printoptions(profile="full")


def calculate_slice_metrics(gt_boxes, slice_boxes, threshold=0.9):
    """Match slices and GT boxes, then return TP, FP and FN.

    Each slice can match at most one GT and each GT can contribute at most one
    TP. Extra slices covering an already matched GT are ignored. A slice that
    does not cover any GT by ``threshold`` is an FP; every unmatched GT is an
    FN.
    """
    candidates = []
    for slice_box in slice_boxes:
        covered_gt = []
        for gt_index, gt_box in enumerate(gt_boxes):
            gt_area = max(
                0.0, (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
            )
            intersection = calculate_intersection(*slice_box, *gt_box)
            if intersection / max(gt_area, 1e-8) >= threshold:
                covered_gt.append(gt_index)
        candidates.append(covered_gt)

    # Maximum bipartite matching avoids making the result depend on slice/GT
    # traversal order when one 3x3 slice happens to cover multiple targets.
    gt_to_slice = {}

    def match_slice(slice_index, visited_gt):
        for gt_index in candidates[slice_index]:
            if gt_index in visited_gt:
                continue
            visited_gt.add(gt_index)
            if gt_index not in gt_to_slice or match_slice(
                gt_to_slice[gt_index], visited_gt
            ):
                gt_to_slice[gt_index] = slice_index
                return True
        return False

    for slice_index in range(len(slice_boxes)):
        match_slice(slice_index, set())

    tp = len(gt_to_slice)
    fp = sum(not covered_gt for covered_gt in candidates)
    fn = len(gt_boxes) - tp
    return tp, fp, fn

def reconstruct_image(x, selected_ids, args):

    img_height, img_width = x.shape[:2]
    patch_height = img_height // args.num_patch
    patch_width = img_width // args.num_patch

    reconstructed_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    for i in range(args.num_patch * args.num_patch):
        row_start = (i // args.num_patch) * patch_height
        row_end = (i // args.num_patch + 1) * patch_height
        col_start = (i % args.num_patch) * patch_width
        col_end = (i % args.num_patch + 1) * patch_width

        if i in selected_ids:
            patch = x[row_start:row_end, col_start:col_end, :]
            reconstructed_img[row_start:row_end, col_start:col_end, :] = patch
        else:
            reconstructed_img[row_start:row_end, col_start:col_end, :] = 0

    reconstructed_img_pil = Image.fromarray(reconstructed_img)
    return reconstructed_img_pil



def overlay_scores_on_image(image, scores, grid_size, save_path):
    """
    Superimpose scores on the image, with each score corresponding to a grid block
    """
    image = np.array(image)
    h, w, _ = image.shape
    block_h, block_w = h // grid_size, w // grid_size
    scores = scores.squeeze()


    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    text_color = (0, 0, 255)

    for i in range(grid_size):
        for j in range(grid_size):
            score = scores[i, j]
            text = f"{float(score):.2f}"
            center_x = int(j * block_w + block_w / 2)
            center_y = int(i * block_h + block_h / 2)

            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = int(center_x - text_size[0] / 2)
            text_y = int(center_y + text_size[1] / 2)

            cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

    cv2.imwrite(save_path, image)
    print(f"Score image saved at {save_path}")

def calculate_intersection(x1, y1, x2, y2, px1, py1, px2, py2):
    inter_x1 = max(x1, px1)
    inter_y1 = max(y1, py1)
    inter_x2 = min(x2, px2)
    inter_y2 = min(y2, py2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    return inter_w * inter_h

def get_3x3_patch_coords(center_i, center_j, num_patch):
    """Mirror ExpSlicer's fixed-size, boundary-shifted 3x3 crop."""
    region_size = min(3, num_patch)
    i_start = min(max(0, center_i - 1), num_patch - region_size)
    j_start = min(max(0, center_j - 1), num_patch - region_size)
    i_end = i_start + region_size
    j_end = j_start + region_size
    return i_start, i_end, j_start, j_end


def get_slice_boxes(selected_mask, image_size):
    """Create one possibly-overlapping 3x3 slice for every positive grid."""
    grid_h, grid_w = selected_mask.shape
    if grid_h != grid_w:
        raise ValueError(f"Expected a square selection mask, got {selected_mask.shape}")
    if image_size % grid_h != 0:
        raise ValueError(
            f"Image size {image_size} must be divisible by grid size {grid_h}"
        )

    patch_size = image_size // grid_h
    slice_boxes = []
    for i in range(grid_h):
        for j in range(grid_w):
            if selected_mask[i, j] != 1:
                continue

            i_start, i_end, j_start, j_end = get_3x3_patch_coords(i, j, grid_h)
            slice_boxes.append([
                j_start * patch_size,
                i_start * patch_size,
                j_end * patch_size,
                i_end * patch_size,
            ])
    return slice_boxes

def compute_mean_std(image):

    mean = np.zeros(3)
    std = np.zeros(3)

    for c in range(3):
        mean[c] = np.mean(image[..., c])
        std[c] = np.std(image[..., c])

    return mean, std


def profile_thop_gflops(model, imgsz, device):
    """Return THOP MACs in billions, the GFLOPs convention used by this project."""
    try:
        from thop import profile
    except ImportError:
        print("THOP is not installed; skipping GFLOPs profiling.")
        return None

    dummy_input = torch.zeros(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
    return macs / 1e9


def patch_select(args):
    print(args)

    root = args.dataset_path
    log_file = os.path.join(args.output_dir, f"patch_select_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    device = torch.device(args.dev if torch.cuda.is_available() else 'cpu')
    print('training device:', device)

    num_cls = 1 if args.loss == 'bce' else 2
    model = RBCN(args, num_cls, if_denoise=True)
    model = model.to(device)
    model = model.eval()



    if args.patch_selection_and_save:
        # after training, patch selection and save selected patches
        if args.weight_path:
            checkpoint = torch.load(args.weight_path, map_location=device)
            model.load_state_dict(checkpoint, strict=True)
            model.to(device)
            print(f"Weights loaded from {args.weight_path}")
        else:
            print("No weights supplied; using random initialization for the smoke test.")

        thop_gflops = profile_thop_gflops(model, args.imgsz, device)
        if thop_gflops is not None:
            print(
                f"[Start] RBCN computation (THOP convention): "
                f"{thop_gflops:.3f} GFLOPs"
            )


        print('============================ Patch Selection ============================')

        imgroot = os.path.join(root, 'images')

        selected_patchroot = os.path.join(args.output_dir, 'selected_patches')
        os.makedirs(selected_patchroot, exist_ok=True)

        if args.nonselection_save:
            nonselected_patchroot = os.path.join(args.output_dir, 'nonselected_patches')
            os.makedirs(nonselected_patchroot, exist_ok=True)

        if args.save_whole_img_after_select:
            save_whole_img_after_select_root = os.path.join(args.output_dir, 'save_whole_image/')
            os.makedirs(save_whole_img_after_select_root, exist_ok=True)

        id = np.arange(args.num_patch * args.num_patch)
        num = 0

        total_time = 0.0
        processed_images = 0
        model_use_total_time = 0.0


        # -----------------------------Traverse the image folder--------------------------
        with torch.no_grad():
            for dir in os.listdir(imgroot):
                final_TP, final_FP, final_FN = 0, 0, 0
                if dir != args.test_split:
                    continue
                for img in os.listdir(os.path.join(imgroot, dir)):
                    if args.max_images and processed_images >= args.max_images:
                        break
                    num += 1
                    name = img.split('.')[0]
                    # --------------------------Operate on the truth mask-------------------------------
                    masks_file = os.path.join(root, "masks", dir, name + ".png")
                    y = Image.open(masks_file)
                    y = np.asarray(y)
                    if len(y.shape) == 3:
                        y = y[..., 0]
                    y = y / (y.max() + 10e-8)
                    y = cv2.resize(y, [args.imgsz, args.imgsz], interpolation=cv2.INTER_NEAREST).astype(np.float32)
                    y = torch.from_numpy(y)
                    y = y.reshape((1, args.imgsz, args.imgsz))
                    ate = nn.MaxPool2d(kernel_size=(args.imgsz // args.num_patch), stride=(args.imgsz // args.num_patch))(y)
                    ate = (ate >= 0.5).float()

                    # --------------------------Inference-------------------------------
                    processed_images += 1
                    x = Image.open(os.path.join(imgroot, dir, img))
                    x = np.asarray(x).copy()
                    mean, std = compute_mean_std(x)
                    temp_x = (x - mean) / (std + 10e-8)
                    temp_x = temp_x / (temp_x.max() + 10e-8)
                    x = torch.from_numpy(x)
                    temp_x = cv2.resize(temp_x, [args.imgsz, args.imgsz], interpolation=cv2.INTER_LINEAR).astype(np.float32)
                    temp_x = torch.from_numpy(temp_x)
                    temp_x = temp_x.unsqueeze(0).permute(0, 3, 1, 2)
                    temp_x = temp_x.to(device)

                    pte, _ = model(temp_x)
                    pte = nn.Sigmoid()(pte)
                    score_map = pte.detach().cpu().numpy()
                    pte[pte >= args.th] = 1
                    pte[pte < args.th] = 0

                    selected_mask = pte.detach().cpu().numpy().astype(int).squeeze()
                    selected_ids = id[selected_mask.reshape(-1) == 1]

                    label_file = os.path.join(root, "labels", dir, name + ".txt")

                    if os.path.exists(label_file):
                        with open(label_file, "r") as f:
                            yolo_labels = [list(map(float, line.strip().split()[1:])) for line in f.readlines()]
                    else:
                        yolo_labels = []

                    gt_boxes = []
                    for label in yolo_labels:
                        cx, cy, w, h = label
                        x1 = max(0.0, (cx - w / 2) * args.imgsz)
                        y1 = max(0.0, (cy - h / 2) * args.imgsz)
                        x2 = min(float(args.imgsz), (cx + w / 2) * args.imgsz)
                        y2 = min(float(args.imgsz), (cy + h / 2) * args.imgsz)
                        gt_boxes.append([x1, y1, x2, y2])

                    slice_boxes = get_slice_boxes(selected_mask, args.imgsz)
                    TP, FP, FN = calculate_slice_metrics(gt_boxes, slice_boxes)
                    precision = TP / (TP + FP) if TP + FP else 0.0
                    recall = TP / (TP + FN) if TP + FN else 0.0
                    print(
                        f"Image: {name}, TP: {TP}, FP: {FP}, FN: {FN}, "
                        f"Precision: {precision:.4f}, Recall: {recall:.4f}"
                    )

                    final_TP += TP
                    final_FP += FP
                    final_FN += FN

                    if args.save_whole_img_with_scores:
                        score_dir_path = os.path.join(args.output_dir, "save_with_score")
                        os.makedirs(score_dir_path, exist_ok=True)
                        score_image_path = os.path.join(score_dir_path, name + "_score.jpg")
                        overlay_scores_on_image(x, score_map, args.num_patch, score_image_path)

                    if args.save_whole_img_after_select:
                        image_after_select = reconstruct_image(x, selected_ids, args)
                        image_after_select.save(os.path.join(save_whole_img_after_select_root, name + '.jpg'))

                    if args.max_images and processed_images >= args.max_images:
                        break


                final_precision = (
                    final_TP / (final_TP + final_FP)
                    if final_TP + final_FP else 0.0
                )
                final_recall = (
                    final_TP / (final_TP + final_FN)
                    if final_TP + final_FN else 0.0
                )
                print(
                    f"tp_final: {final_TP} ; fn_final: {final_FN} ; "
                    f"fp_final: {final_FP} ; CRP: {final_precision} ; "
                    f"CRR: {final_recall}"
                )

        if thop_gflops is not None:
            print(
                f"[End] RBCN computation (THOP convention): "
                f"{thop_gflops:.3f} GFLOPs"
            )



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--imgsz', type=int, help='image size', default=640)
    argparser.add_argument('--aug', action='store_true', help='data augmentation or not', default=False)
    argparser.add_argument('--norm', action='store_true', help='normalize data or not', default=True)
    argparser.add_argument('--num_patch', type=int, help='number of patches (eg. 4*4 or 8*8)', default=20)

    argparser.add_argument('--dim', type=int, help='attention embedding dimension for patch selection', default=128)
    argparser.add_argument('--tokensz', type=int, help='token size for image embedding', default=4)
    argparser.add_argument(
        '--rbcn_block',
        type=str,
        choices=('c2fp', 'c3k2'),
        default='c3k2',
        help='RBCN basic block: paper C2fP (default) or legacy C3k2',
    )
    argparser.add_argument('--th', type=float, help='threshold for attention or not', default=0.5)
    argparser.add_argument('--dev', type=str, help='cuda device', default='cuda:0')
    argparser.add_argument('--epoch', type=int, help='number of training epochs', default=1)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    argparser.add_argument('--batchsz', type=int, help='batch size', default=32)
    argparser.add_argument('--loss', type=str, help='loss function(bce/ce)', default='bce')
    argparser.add_argument('--beta', type=float, help='weighted cross entropy parameter', default=1)

    # patch selection
    argparser.add_argument('--patch_selection_and_save', action='store_true', help='patch selection or not', default=True)
    argparser.add_argument('--nonselection_save', action='store_true', help='save non-selected patches or not', default=False)
    argparser.add_argument('--save_whole_img_with_scores', action='store_true', help='save non-selected patches or not', default=False)
    argparser.add_argument('--save_whole_img_after_select', action='store_true',
                           help='save the whole image after select or not', default=False)
    argparser.add_argument(
        '--dataset_path',
        type=str,
        default='./data',
        help='dataset root containing images, masks, and labels',
    )
    argparser.add_argument('--test_split', type=str, default='test')
    argparser.add_argument(
        '--weight_path',
        type=str,
        default=None,
        help='optional RBCN state dict; random initialization is used when omitted',
    )
    argparser.add_argument('--max_images', type=int, default=0, help='0 processes the complete test set')
    argparser.add_argument('--output_dir', type=str, default='./runs/rbcn_test')
    args = argparser.parse_args()

    patch_select(args)
