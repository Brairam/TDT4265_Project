from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import time
import medmnist
import torch.nn.functional as F
from medmnist import INFO, Evaluator


start_time = time.time()


data_flag = 'organmnist3d' 
download = True
NUM_EPOCHS = 20
BATCH_SIZE = 8
lr = 0.001

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])
class Transform3D:

    def __init__(self, mul=None):
        self.mul = mul

    def __call__(self, voxel):
   
        if self.mul == '0.5':
            voxel = voxel * 0.5
        elif self.mul == 'random':
            voxel = voxel * np.random.uniform()
        
        return voxel.astype(np.float32)

train_transform =  Transform3D()
eval_transform = Transform3D()


train_dataset = DataClass(split='train', transform=train_transform, download=download)
test_dataset = DataClass(split='test', transform=eval_transform, download=download)

train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)


class MLPMixer(nn.Module):
    def __init__(self, in_channels=1, num_classes=7, img_size=28, patch_size=7, embed_dim=64):
        super().__init__()
        num_patches = (img_size // patch_size) ** 3
        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(num_patches, 64),
            nn.GELU(),
            nn.Linear(64, num_patches))
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim))
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x) 
        x = x.flatten(2)  
        x = x.transpose(1, 2)  

        y = self.norm1(x)
        y = self.mlp1(y.transpose(1, 2)).transpose(1, 2)
        x = x + y

        y = self.norm2(x)
        y = self.mlp2(y)
        x = x + y

        x = x.mean(dim=1)
        x = self.head(x)
        return x

model = MLPMixer(in_channels=n_channels, num_classes=n_classes)


if task == "multi-label, binary-class":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # start with a higher learning rate
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # decay after 10 epochs

train_losses = []
train_accuracies = []
test_accuracies = []


for epoch in range(NUM_EPOCHS):
    train_correct = 0
    train_total = 0
    test_correct = 0
    test_total = 0
    epoch_loss = 0
    model.train()
    for inputs, targets in tqdm(train_loader):
        optimizer.zero_grad()
        inputs = inputs.to(torch.float32)
        outputs = model(inputs)

        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32)
            loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += targets.size(0)
        train_correct += (predicted == targets).sum().item()
    scheduler.step()


    train_loss = epoch_loss / len(train_loader)
    train_acc = train_correct / train_total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)


    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(torch.float32)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += targets.size(0)
            test_correct += (predicted == targets.squeeze().long()).sum().item()
    test_acc = test_correct / test_total
    test_accuracies.append(test_acc)

    print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


def test(split):
    model.eval()
    y_true = torch.tensor([]).to(torch.int64)
    y_score = torch.tensor([]).to(torch.float32)

    data_loader = train_loader_at_eval if split == 'train' else test_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(torch.float32)
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                outputs = outputs.softmax(dim=-1)
            else:
                targets = targets.squeeze().long()
                outputs = outputs.softmax(dim=-1)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()
        
        evaluator = Evaluator(data_flag, split)
        metrics = evaluator.evaluate(y_score)

        print(f'{split}  auc: {metrics[0]:.3f}  acc: {metrics[1]:.3f}')

print('==> Evaluating ...')
test('train')
test('test')
end_time = time.time()
print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")



plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()
plt.tight_layout()
plt.show()


y_true = []
y_pred = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(torch.float32)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(targets.numpy())
        y_pred.extend(predicted.numpy())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', 
            xticklabels=[f'{i}' for i in range(n_classes)],
            yticklabels=[f'{i}' for i in range(n_classes)])
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()


y_true_bin = label_binarize(y_true, classes=range(n_classes))
y_score = []

with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(torch.float32)
        outputs = model(inputs)
        y_score.extend(F.softmax(outputs, dim=1).numpy())

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], np.array(y_score)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], 
             label=f'{info["label"][str(i)]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

