import random
import shutil
import yaml
from pathlib import Path

def split_dataset(
    dataset_root,
    output_root,
    num_clients=2,
    seed=42,
):
    random.seed(seed)

    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    # Load original data.yaml
    with open(dataset_root / "data.yaml", "r") as f:
        data = yaml.safe_load(f)

    names = data["names"]
    nc = data["nc"]

    train_imgs = list((dataset_root / "train/images").glob("*"))
    val_imgs = list((dataset_root / "val/images").glob("*"))

    random.shuffle(train_imgs)
    splits = [train_imgs[i::num_clients] for i in range(num_clients)]

    for cid in range(num_clients):
        client_root = output_root / f"client_{cid}"

        for split in ["train", "val"]:
            (client_root / split / "images").mkdir(parents=True, exist_ok=True)
            (client_root / split / "labels").mkdir(parents=True, exist_ok=True)

        # Client-specific TRAIN
        for img in splits[cid]:
            lbl = dataset_root / "train/labels" / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy(img, client_root / "train/images" / img.name)
                shutil.copy(lbl, client_root / "train/labels" / lbl.name)

        # Shared VAL
        for img in val_imgs:
            lbl = dataset_root / "val/labels" / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy(img, client_root / "val/images" / img.name)
                shutil.copy(lbl, client_root / "val/labels" / lbl.name)

        # Create client data.yaml
        client_yaml = {
            "path": str(client_root),
            "train": "train/images",
            "val": "val/images",
            "nc": nc,
            "names": names,
        }

        with open(client_root / "data.yaml", "w") as f:
            yaml.dump(client_yaml, f)

        print(f"✅ Client {cid} ready")

if __name__ == "__main__":
    split_dataset(
        dataset_root="C:/Users/PC/Desktop/dentalresearch/models/federated/dataset",
        output_root="C:/Users/PC/Desktop/dentalresearch/models/federated/fl_results",
        num_clients=2,
    )
