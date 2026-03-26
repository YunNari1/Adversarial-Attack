import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
# =========================
# 1. 디바이스 설정
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 디바이스:", device)

# =========================
# 2. MNIST 데이터 로드
# =========================
transform = transforms.Compose([
    transforms.ToTensor()
])

mnist_train = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)

mnist_test = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

mnist_train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)
mnist_test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=False)

print("MNIST 데이터 로드 완료!")

# =========================
# 3. CIFAR-10 데이터 로드
# =========================
transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

cifar_train = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_cifar
)

cifar_test = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_cifar
)

cifar_train_loader = torch.utils.data.DataLoader(
    cifar_train, batch_size=64, shuffle=True
)

cifar_test_loader = torch.utils.data.DataLoader(
    cifar_test, batch_size=64, shuffle=False
)

print("CIFAR-10 데이터 로드 완료!")

# =========================
# 4. MNIST 모델
# =========================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(-1, 64 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# =========================
# 5. CIFAR 모델
# =========================
def get_cifar_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # 마지막 layer 수정 (10 클래스)
    model.fc = nn.Linear(model.fc.in_features, 10)

    return model

# =========================
# 6. 모델 생성
# =========================
mnist_model = SimpleCNN().to(device)
cifar_model = get_cifar_model().to(device)
# =========================
# 7. 학습 함수 (재사용용)
# =========================
def train_model(model, loader, epochs=7):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

# =========================
# 8. 평가 함수
# =========================
def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

# =========================
# 9. MNIST 학습
# =========================
print("\n===== MNIST 학습 시작 =====")
train_model(mnist_model, mnist_train_loader, epochs=3)

mnist_acc = evaluate_model(mnist_model, mnist_test_loader)
print(f"MNIST 정확도: {mnist_acc:.2f}%")

# =========================
# 10. CIFAR 학습
# =========================
print("\n===== CIFAR-10 학습 시작 =====")

train_model(cifar_model, cifar_train_loader, epochs=5)

cifar_acc = evaluate_model(cifar_model, cifar_test_loader)
print(f"CIFAR-10 정확도: {cifar_acc:.2f}%")