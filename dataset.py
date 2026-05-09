import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class TongueDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir:
            dataset/
                diabetes/
                nondiabetes/
        """

        self.image_paths = []
        self.labels = []

        classes = {
            "diabetes": 1,
            "nondiabetes": 0
        }

        for class_name, label in classes.items():
            class_folder = os.path.join(root_dir, class_name)

            for file_name in os.listdir(class_folder):
                if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_path = os.path.join(class_folder, file_name)

                    self.image_paths.append(full_path)
                    self.labels.append(label)

        # preprocessing for ResNet50
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, label