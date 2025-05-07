import os
import json
import numpy as np
import skimage.io as sio
from pathlib import Path
from tqdm import tqdm
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pycocotools import mask as mask_utils
from PIL import Image
import gc

# --- Utility Functions ---
def decode_maskobj(mask_obj):
    return mask_utils.decode(mask_obj)

def encode_mask(binary_mask):
    arr = np.asfortranarray(binary_mask).astype(np.uint8)
    rle = mask_utils.encode(arr)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def read_maskfile(filepath):
    return sio.imread(filepath)

# --- Dataset ---
class CellInstanceDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_dirs = sorted(os.listdir(root_dir))

    def __getitem__(self, idx):
        image_name = self.image_dirs[idx]
        image_path = os.path.join(self.root_dir, image_name, 'image.tif')
        image = Image.open(image_path).convert("RGB")

        masks = []
        boxes = []
        labels = []

        for class_idx in range(1, 5):
            class_path = os.path.join(self.root_dir, image_name, f'class{class_idx}.tif')
            if not os.path.exists(class_path):
                continue
            try:
                class_mask = read_maskfile(class_path)
            except Exception:
                print(f"[WARN] CANNOT IDENTIFY：{class_path}")
                continue

            obj_ids = np.unique(class_mask)
            obj_ids = obj_ids[obj_ids != 0]

            for obj_id in obj_ids:
                binary_mask = class_mask == obj_id
                pos = np.where(binary_mask)
                if pos[0].size == 0 or pos[1].size == 0:
                    continue
                xmin, xmax = np.min(pos[1]), np.max(pos[1])
                ymin, ymax = np.min(pos[0]), np.max(pos[0])
                if xmax <= xmin or ymax <= ymin:
                    continue
                boxes.append([xmin, ymin, xmax, ymax])
                masks.append(binary_mask)
                labels.append(class_idx)

        if len(boxes) == 0:
            return None

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.stack(masks).astype(np.uint8), dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.image_dirs)

# --- Collate function ---
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return tuple(zip(*batch))


# --- Training Function ---
def train_one_epoch(model, optimizer, data_loader, device, scheduler=None):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for images, targets in tqdm(data_loader):
        try:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                continue

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            if scheduler:
                scheduler.step()

            total_loss += losses.item()
            num_batches += 1

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print("[CUDA OOM] Skip batch due to memory overflow.")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

        gc.collect()
        torch.cuda.empty_cache()

    return total_loss / max(num_batches, 1)


def evaluate(model, data_loader, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, targets) in enumerate(data_loader):
        try:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            total_loss += losses.item()
            num_batches += 1

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"[Validation OOM in Batch {batch_idx}] Skip.")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
        except Exception as e:
            print(f"[Validation Error in Batch {batch_idx}]: {e}")
            continue

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss

# --- Inference and Submission ---
def inference(model, test_dir, output_json, image_id_map):
    model.eval()
    torch.cuda.empty_cache()
    results = []
    with torch.no_grad():
        for image_file in tqdm(os.listdir(test_dir)):
            image = Image.open(os.path.join(test_dir, image_file)).convert("RGB")
            image_tensor = F.to_tensor(image).unsqueeze(0).cuda()
            output = model(image_tensor)[0]

            for i in range(len(output['scores'])):
                if output['scores'][i] < 0.05:
                    continue
                mask = output['masks'][i, 0].cpu().numpy() > 0.5
                if mask.sum() == 0:
                    continue
                rle = encode_mask(mask)

                box = output['boxes'][i].cpu().numpy()
                bbox = [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]

                results.append({
                    'image_id': image_id_map[image_file],
                    'bbox': bbox,
                    'score': float(output['scores'][i]),
                    'category_id': int(output['labels'][i]),
                    'segmentation': {
                        'size': list(mask.shape),
                        'counts': rle['counts']
                    }
                })

    with open(output_json, 'w') as f:
        json.dump(results, f)

# --- Main ---
def main():
    train_root = '/kaggle/input/dataset/train'
    test_root = '/kaggle/input/dataset/test_release'
    image_id_map_path = '/kaggle/input/dataset/test_image_name_to_ids.json'
    model_path = '/kaggle/working/maskrcnn_model_best.pth'

    with open(image_id_map_path, 'r') as f:
        image_id_map = json.load(f)
        if isinstance(image_id_map, list):
            image_id_map = {item['file_name']: item['id'] for item in image_id_map}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 5)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, 5
    )

    model.to(device)

    dataset = CellInstanceDataset(train_root, transforms=F.to_tensor)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.05,
            steps_per_epoch=len(train_loader),
            epochs=25,
            pct_start=0.3,
            anneal_strategy='cos',
            final_div_factor=10
        )

    best_val_loss = float('inf')
    for epoch in range(25):
        print(f"\n--- Epoch {epoch + 1} ---")
        train_loss = train_one_epoch(model, optimizer, train_loader, device, scheduler)
        val_loss = evaluate(model, val_loader, device)

        print(f"[Train] Loss: {train_loss:.4f} | [Validation] Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"\u2714\ufe0f Saved best model at epoch {epoch + 1} with val loss {val_loss:.4f}")

    model.load_state_dict(torch.load(model_path))
    inference(model, test_root, 'test-results.json', image_id_map)

if __name__ == '__main__':
    main()