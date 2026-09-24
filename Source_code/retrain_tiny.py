import pickle
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision import models
from sklearn.metrics import accuracy_score
import sys
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    class _FakeTqdm:
        def __init__(self, iterable, desc=None):
            self.iterable = iterable
            if desc:
                print(desc)

        def __iter__(self):
            return iter(self.iterable)

        def __len__(self):
            return len(self.iterable)

        def set_postfix(self, *args, **kwargs):
            pass

        def update(self, n=1):
            pass

        def close(self):
            pass


    def tqdm(iterable, desc=None):
        return _FakeTqdm(iterable, desc=desc)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_model(model_path, num_classes=200):
    model = models.resnet101(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model


class NpyDataset(Dataset):
    def __init__(self, x_path, y_path, transform=None, mmap=False):
        if mmap:
            self.x = np.load(x_path, mmap_mode='r')
        else:
            self.x = np.load(x_path)
        self.y = np.load(y_path)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        x = x.astype(np.float32) / 255.0
        if self.transform:
            x = self.transform(x)
        x = torch.from_numpy(x).float()
        y = torch.tensor(y, dtype=torch.long)
        return x, y


def evaluate_tensor(model, val_x, val_y, device='cpu', batch_size=32):
    model.eval()
    model = model.to(device)
    all_preds = []
    with torch.no_grad():
        for start in range(0, len(val_x), batch_size):
            end = min(start + batch_size, len(val_x))
            x_batch = val_x[start:end].to(device)
            outputs = model(x_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    acc = accuracy_score(val_y.cpu().numpy(), all_preds)
    return acc


def train_model(model, train_loader, val_x, val_y, epochs=10, lr=1e-3, device='cpu'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

    val_x = val_x.to(device)
    val_y = val_y.to(device)

    best_acc = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        print(f"Epoch {epoch + 1}/{epochs} Start training...", flush=True)

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            if (batch_idx + 1) % 500 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}",
                      flush=True)

        scheduler.step()

        val_acc = evaluate_tensor(model, val_x, val_y, device)
        best_acc = max(best_acc, val_acc)
        print(f"Epoch [{epoch + 1}/{epochs}] complete, Val Acc: {val_acc:.4f}, Best: {best_acc:.4f}",
              flush=True)

    return best_acc


def transform_image(x):
    if x.shape[0] != 224 or x.shape[1] != 224:
        x_img = (x * 255).astype(np.uint8)
        x_pil = Image.fromarray(x_img)
        x_pil = x_pil.resize((224, 224), Image.BILINEAR)
        x = np.array(x_pil, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = np.transpose(x, (2, 0, 1))
    x = (x - mean.reshape(3, 1, 1)) / std.reshape(3, 1, 1)
    return x


if __name__ == '__main__':
    output_path = sys.argv[1]
    data_path = sys.argv[2]
    selection_method = sys.argv[3]

    data_name = "TinyImageNet"
    model_name = "ResNet101"
    size = 500

    print(f"Dataset: {data_name}, Model: {model_name}, Method: {selection_method}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Equipment used: {device}")

    data_dir = os.path.join(data_path, 'Retrain', 'TinyImageNet_ResNet101')
    model_path = os.path.join(data_path, 'Pretrained_model', 'model_tinyimagenet_resnet101.pth')

    test_dataset = NpyDataset(
        os.path.join(data_dir, 'tiny_x_test.npy'),
        os.path.join(data_dir, 'tiny_y_test.npy'),
        transform=transform_image,
        mmap=False
    )


    t_file = f"{data_path}/Retrain/{data_name}_{model_name}.pkl"
    with open(t_file, 'rb') as f:
        t_indices = pickle.load(f)
    v_set = list(set(range(len(test_dataset))) - set(t_indices))

    model = load_model(model_path, num_classes=200)

    print("Preprocess the validation set...", flush=True)
    val_x_list = []
    val_y_list = []
    for idx in tqdm(v_set, desc="Preprocess the validation set"):
        x, y = test_dataset[idx]
        val_x_list.append(x.numpy())
        val_y_list.append(y.item())

    val_x = torch.FloatTensor(np.array(val_x_list))
    val_y = torch.LongTensor(np.array(val_y_list))
    del val_x_list, val_y_list

    acc_ori = evaluate_tensor(model, val_x, val_y, device)
    print(f"Original Model Accuracy: {acc_ori:.4f}")

    print("Start Retraining!")

    train_dataset = NpyDataset(
        os.path.join(data_dir, 'tiny_x_train.npy'),
        os.path.join(data_dir, 'tiny_y_train.npy'),
        transform=transform_image,
        mmap=False
    )

    acc_re_list = []
    acc_imp_list = []

    for i in range(5):
        print(f"\n{'=' * 60}")
        print(f"Run {i + 1}/5")
        print(f"{'=' * 60}", flush=True)

        set_seed(42 + i)


        if selection_method == "SETS":
            file_name = f"{data_path}/Retrain/SETS/{data_name}_{model_name}_{size}.pkl"
        else:
            file_name = f"{data_path}/Retrain/{selection_method}/{data_name}_{model_name}_{size}_seed{i}.pkl"
        with open(file_name, 'rb') as f:
            sub_ets = np.array(pickle.load(f), dtype=np.int32)


        selected_dataset = Subset(test_dataset, sub_ets.tolist())
        combined_dataset = ConcatDataset([train_dataset, selected_dataset])

        train_loader = DataLoader(
            combined_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        model = load_model(model_path, num_classes=200)

        best_acc = train_model(
            model,
            train_loader,
            val_x,
            val_y,
            epochs=10,
            lr=1e-3,
            device=device
        )

        acc_re_list.append(best_acc)
        acc_imp = best_acc - acc_ori
        acc_imp_list.append(acc_imp)

        print(f"Retrained Model Accuracy {selection_method}: {best_acc:.4f}")
        print(f"Accuracy Improvement: {acc_imp:.4f}")


        if device == 'cuda':
            torch.cuda.empty_cache()

    print(f"average accuracy imp {selection_method} {size}: "
          f"{(sum(acc_imp_list) / len(acc_imp_list)):.4f}")


    file_path = str(output_path) + f"/new_retrain_result/{data_name}_{model_name}_{selection_method}_{size}.txt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as f:
        f.write(f"Original Model Accuracy: {acc_ori:.4f}\n")
        f.write(f"{selection_method} Retraining\n")
        f.write("Acc Re List:\n")
        f.write("\n".join([f"{x:.4f}" for x in acc_re_list]) + "\n")
        f.write("Acc Imp List:\n")
        f.write("\n".join([f"{x:.4f}" for x in acc_imp_list]) + "\n")

    print("save successfully!")