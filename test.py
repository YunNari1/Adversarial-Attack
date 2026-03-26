import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

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

# =========================
# 11. FGSM 구현 
# =========================
def fgsm_untargeted(model, x, y, eps):
    x_adv = x.clone().detach().requires_grad_(True)

    output = model(x_adv)
    loss = F.cross_entropy(output, y)

    model.zero_grad()
    loss.backward()

    x_adv = x_adv + eps * x_adv.grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()

def fgsm_targeted(model, x, target, eps):
    x_adv = x.clone().detach().requires_grad_(True)

    output = model(x_adv)
    loss = F.cross_entropy(output, target)

    model.zero_grad()
    loss.backward()

    x_adv = x_adv - eps * x_adv.grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()

def fgsm_attack_success_rate(model, loader, eps, attack_type="untargeted", max_samples=100):
    model.eval()
    success = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        for i in range(x.size(0)):
            xi = x[i:i+1]
            yi = y[i:i+1]

            if attack_type == "untargeted":
                x_adv = fgsm_untargeted(model, xi, yi, eps)
                pred = model(x_adv).argmax(1)
                if pred.item() != yi.item():
                    success += 1

            elif attack_type == "targeted":
                num_classes = model(xi).shape[1]
                target = torch.tensor([(yi.item()+1)%num_classes]).to(device)

                x_adv = fgsm_targeted(model, xi, target, eps)
                pred = model(x_adv).argmax(1)

                if pred.item() == target.item():
                    success += 1

            total += 1
            if total >= max_samples:
                return success / total

    return success / total    


print("\n===== FGSM 공격 결과 =====")

for eps in [0.05, 0.1, 0.2, 0.3]:

    mnist_unt = fgsm_attack_success_rate(mnist_model, mnist_test_loader, eps, "untargeted")
    mnist_tar = fgsm_attack_success_rate(mnist_model, mnist_test_loader, eps, "targeted")

    cifar_unt = fgsm_attack_success_rate(cifar_model, cifar_test_loader, eps, "untargeted")
    cifar_tar = fgsm_attack_success_rate(cifar_model, cifar_test_loader, eps, "targeted")

    print(f"\n[eps={eps}]")
    print(f"MNIST Untargeted: {mnist_unt:.2f}")
    print(f"MNIST Targeted:   {mnist_tar:.2f}")
    print(f"CIFAR Untargeted: {cifar_unt:.2f}")
    print(f"CIFAR Targeted:   {cifar_tar:.2f}")


# =========================
# 12. PGD 구현 
# =========================

def pgd_untargeted(model, x, y, k, eps, eps_step):
    x_orig = x.clone().detach()
    x_adv = x.clone().detach()

    for _ in range(k):
        x_adv.requires_grad_(True)

        output = model(x_adv)
        loss = F.cross_entropy(output, y)

        model.zero_grad()
        loss.backward()

        grad = x_adv.grad.sign()

        # untargeted: loss를 키우는 방향
        x_adv = x_adv + eps_step * grad

        # 원본 이미지 기준 eps 범위 안으로 projection
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)

        # 이미지 값 범위 제한
        x_adv = torch.clamp(x_adv, 0, 1).detach()

    return x_adv


def pgd_targeted(model, x, target, k, eps, eps_step):
    x_orig = x.clone().detach()
    x_adv = x.clone().detach()

    for _ in range(k):
        x_adv.requires_grad_(True)

        output = model(x_adv)
        loss = F.cross_entropy(output, target)

        model.zero_grad()
        loss.backward()

        grad = x_adv.grad.sign()

        # targeted: target loss를 줄이는 방향
        x_adv = x_adv - eps_step * grad

        # 원본 이미지 기준 eps 범위 안으로 projection
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)

        # 이미지 값 범위 제한
        x_adv = torch.clamp(x_adv, 0, 1).detach()

    return x_adv

def pgd_attack_success_rate(model, loader, eps, eps_step, k,
                            attack_type="untargeted", max_samples=100):
    model.eval()
    success = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        for i in range(x.size(0)):
            xi = x[i:i+1]
            yi = y[i:i+1]

            if attack_type == "untargeted":
                x_adv = pgd_untargeted(model, xi, yi, k, eps, eps_step)
                pred = model(x_adv).argmax(1)

                if pred.item() != yi.item():
                    success += 1

            elif attack_type == "targeted":
                num_classes = model(xi).shape[1]
                target = torch.tensor([(yi.item() + 1) % num_classes]).to(device)

                x_adv = pgd_targeted(model, xi, target, k, eps, eps_step)
                pred = model(x_adv).argmax(1)

                if pred.item() == target.item():
                    success += 1

            total += 1
            if total >= max_samples:
                return success / total

    return success / total

print("\n===== PGD 공격 결과 =====")

k = 40
eps_step = 0.01

for eps in [0.05, 0.1, 0.2, 0.3]:
    mnist_pgd_unt = pgd_attack_success_rate(
        mnist_model, mnist_test_loader, eps, eps_step, k, "untargeted"
    )
    mnist_pgd_tar = pgd_attack_success_rate(
        mnist_model, mnist_test_loader, eps, eps_step, k, "targeted"
    )

    cifar_pgd_unt = pgd_attack_success_rate(
        cifar_model, cifar_test_loader, eps, eps_step, k, "untargeted"
    )
    cifar_pgd_tar = pgd_attack_success_rate(
        cifar_model, cifar_test_loader, eps, eps_step, k, "targeted"
    )

    print(f"\n[eps={eps}]")
    print(f"MNIST PGD Untargeted: {mnist_pgd_unt:.2f}")
    print(f"MNIST PGD Targeted:   {mnist_pgd_tar:.2f}")
    print(f"CIFAR PGD Untargeted: {cifar_pgd_unt:.2f}")
    print(f"CIFAR PGD Targeted:   {cifar_pgd_tar:.2f}")

# =========================
# 11. 이미지 저장
# =========================

def save_images(model, loader, dataset_name):
    os.makedirs("results", exist_ok=True)

    model.eval()

    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    for i in range(5):
        xi = x[i:i+1]
        yi = y[i:i+1]

        # PGD 공격 적용
        x_adv = pgd_untargeted(model, xi, yi, k=40, eps=0.3, eps_step=0.01)

        # CPU로 변환
        orig = xi.squeeze().detach().cpu().numpy()
        adv = x_adv.squeeze().detach().cpu().numpy()
        pert = (x_adv - xi).squeeze().detach().cpu().numpy()

        # CIFAR는 normalization 되어 있어서 보기 좋게 복원 필요
        if orig.ndim == 3:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            for c in range(3):
                orig[c] = orig[c] * std[c] + mean[c]
                adv[c] = adv[c] * std[c] + mean[c]

            orig = orig.transpose(1, 2, 0)
            adv = adv.transpose(1, 2, 0)
            pert = pert.transpose(1, 2, 0)

        plt.figure(figsize=(10,3))

        # 원본
        plt.subplot(1,3,1)
        if orig.ndim == 2:
            plt.imshow(orig, cmap='gray')
        else:
            plt.imshow(orig)
        plt.title("Original")
        plt.axis("off")

        # 공격 이미지
        plt.subplot(1,3,2)
        if adv.ndim == 2:
            plt.imshow(adv, cmap='gray')
        else:
            plt.imshow(adv)
        plt.title("Adversarial")
        plt.axis("off")

        # perturbation
        plt.subplot(1,3,3)
        if pert.ndim == 2:
            plt.imshow(pert * 10, cmap='gray')
        else:
            plt.imshow(pert * 10)
        plt.title("Perturbation x10")
        plt.axis("off")

        # 저장
        plt.savefig(f"results/{dataset_name}_{i}.png")
        plt.close()

    print(f"{dataset_name} 이미지 저장 완료!")


# 실행
print("\n===== 이미지 저장 =====")
save_images(mnist_model, mnist_test_loader, "mnist")
save_images(cifar_model, cifar_test_loader, "cifar")