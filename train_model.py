import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import pickle
import os
from tqdm import tqdm

# The 12 cat breeds in Oxford-IIIT Pet dataset
CAT_BREEDS_LOWER = {
    'abyssinian', 'bengal', 'birman', 'bombay', 'british_shorthair', 
    'egyptian_mau', 'maine_coon', 'persian', 'ragdoll', 'russian_blue', 
    'siamese', 'sphynx'
}

class CatDataset(Dataset):
    def __init__(self, root_dir, split='trainval', transform=None):
        # Create a temporary dataset without transforms to quickly filter labels
        temp_dataset = datasets.OxfordIIITPet(root=root_dir, split=split, target_types='category', download=True, transform=None)
        
        self.original_classes = temp_dataset.classes
        
        # Find indices of cat breeds in the original dataset's class list
        self.cat_class_indices = []
        self.cat_classes = []
        
        for idx, cls_name in enumerate(self.original_classes):
            if cls_name.lower() in CAT_BREEDS_LOWER:
                self.cat_class_indices.append(idx)
                self.cat_classes.append(cls_name)
                
        self.orig_to_new_idx = {orig: new for new, orig in enumerate(self.cat_class_indices)}
        
        # Filter samples
        self.samples = []
        # In OxfordIIITPet, we can usually access labels directly to avoid loading images.
        # But to be safe across torchvision versions, we iterate the temp dataset (which is fast since no transforms)
        print("Filtering dataset for cat breeds...")
        for i in tqdm(range(len(temp_dataset)), desc="Filtering"):
            # temp_dataset._labels contains the targets in newer torchvision, 
            # but using __getitem__ without transforms is also quite fast.
            _, label = temp_dataset[i]
            if label in self.orig_to_new_idx:
                self.samples.append((i, self.orig_to_new_idx[label]))
                
        # Now create the actual dataset with transforms
        self.dataset = datasets.OxfordIIITPet(root=root_dir, split=split, target_types='category', download=False, transform=transform)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        orig_idx, new_label = self.samples[idx]
        img, _ = self.dataset[orig_idx]
        return img, new_label

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Standard ViT transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    print("Preparing dataset...")
    # Using split='trainval' for training
    cat_train_dataset = CatDataset(root_dir='./data', split='trainval', transform=transform)
    cat_classes = cat_train_dataset.cat_classes
    print(f"Found {len(cat_classes)} cat breeds: {cat_classes}")
    print(f"Total training images: {len(cat_train_dataset)}")

    train_loader = DataLoader(cat_train_dataset, batch_size=32, shuffle=True, num_workers=0)

    # Initialize ViT
    print("Initializing Vision Transformer (ViT)...")
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    
    # Replace the classification head for our cat classes
    num_classes = len(cat_classes)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train for a few epochs
    num_epochs = 3
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'loss': running_loss/total, 'acc': correct/total})

    print("Training complete!")

    # Save model as .pkl as requested
    model_save_path = "cat_breeds_vit.pkl"
    print(f"Saving model to {model_save_path}...")
    
    # Save a dictionary with the model state and the class names
    save_data = {
        'model_state_dict': model.state_dict(),
        'class_names': cat_classes
    }
    
    with open(model_save_path, 'wb') as f:
        pickle.dump(save_data, f)
        
    print("Done!")

if __name__ == "__main__":
    main()
