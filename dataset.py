from torch.utils.data import Dataset
from torchvision import transforms

class MagicBrushDataset(Dataset):
    def __init__(self, hf_dataset, image_size=256):
        self.ds = hf_dataset

        self.img_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.mask_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]

        # ✅ MagicBrush already returns PIL Images
        source = row["source_img"].convert("RGB")
        target = row["target_img"].convert("RGB")
        mask   = row["mask_img"].convert("L")
        text   = row["instruction"]

        source = self.img_tf(source)
        target = self.img_tf(target)
        mask   = (self.mask_tf(mask) > 0.5).float()

        return {
            "source": source,   # [3, H, W]
            "target": target,   # [3, H, W]
            "mask": mask,       # [1, H, W]
            "text": text
        }
