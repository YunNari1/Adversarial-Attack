import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

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
