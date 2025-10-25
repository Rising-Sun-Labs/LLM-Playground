import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sympy import false


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [B, 3, 32, 32] -> [B, 3072]
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def main():
    device = get_device()
    print(f"Using device: {device}")

    # Hyper-parameters for CIFAR-100
    input_size = 3 * 32 * 32
    hidden_size = 500
    num_classes = 100
    num_epochs = 5
    batch_size = 128
    learning_rate = 0.001

    # CIFAR-100 normalization
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
        ]
    )

    # Datasets (won't re-download if already present)
    train_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=False, transform=transform
    )

    # Data loaders: set num_workers = 0 to avoid multiprocessing issues on macOS
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, # set > 0 later after adding __main__ guard( we did ) and once stable
        pin_memory=False
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )

    model = NeuralNet(input_size, hidden_size, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    total_steps = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}'.format(epoch + 1, num_epochs, i+1, total_steps, loss.item()))

    # Test
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy on CIFAR-100 test images: {:.2f}%'.format(100 * correct / total))

    torch.save(model.state_dict(), "model.ckpt")
    print("Saved weight to model.ckpt")

if __name__ == "__main__":
    main()