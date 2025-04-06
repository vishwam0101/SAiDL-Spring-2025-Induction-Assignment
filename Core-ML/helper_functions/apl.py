import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from helper_functions.data_utils import get_CIFAR10_noisy_data
from helper_functions.data_utils import get_custom_loaders


class APLCNN(nn.Module):
    """
    PyTorch version of ThreeLayerConvNet for APL experiments.
    Architecture:
    conv - relu - 2x2 max pool - affine - relu - affine - softmax
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
    ):
        super(APLCNN, self).__init__()

        C, H, W = input_dim
        self.conv = nn.Conv2d(
            C, num_filters, kernel_size=filter_size, padding=(filter_size - 1) // 2
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate size after pooling
        H_out, W_out = H // 2, W // 2
        self.affine1 = nn.Linear(num_filters * H_out * W_out, hidden_dim)
        self.affine2 = nn.Linear(hidden_dim, num_classes)

        # Initialize weights with weight_scale
        nn.init.normal_(self.conv.weight, mean=0.0, std=weight_scale)
        nn.init.constant_(self.conv.bias, 0)
        nn.init.normal_(self.affine1.weight, mean=0.0, std=weight_scale)
        nn.init.constant_(self.affine1.bias, 0)
        nn.init.normal_(self.affine2.weight, mean=0.0, std=weight_scale)
        nn.init.constant_(self.affine2.bias, 0)

    def forward(self, x):
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 channels but got {x.shape[1]} channels!")
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.affine1(x)
        x = self.relu(x)
        x = self.affine2(x)
        return x


class APL(nn.Module):
    """
    Active-Passive Loss (APL) Implementation
    Combines Cross-Entropy (CE) as active loss and Mean Absolute Error (MAE) as passive loss.
    """

    def __init__(self, alpha=0.5):
        super(APL, self).__init__()
        self.alpha = alpha

    def forward(self, outputs, targets):
        ce_loss = F.cross_entropy(outputs, targets, reduction="mean")
        mae_loss = torch.mean(
            torch.abs(
                F.softmax(outputs, dim=1)
                - F.one_hot(targets, num_classes=outputs.shape[1]).float()
            )
        )
        apl_loss = self.alpha * ce_loss + (1 - self.alpha) * mae_loss
        return apl_loss, ce_loss.item(), mae_loss.item()




def run_apl(
    alpha=0.5,
    lr=5e-4,
    num_epochs=10,
    batch_size=128,  # ✅ Default batch size, but modifiable
    print_every=100,
    train_loader=None,  # ✅ Option to pass custom loaders
    test_loader=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Use provided train/test loaders if available, otherwise load CIFAR-10 with noise
    if train_loader is None or test_loader is None:
        noise_rate = 0.6  # ✅ Set your noise rate here if needed
        data = get_CIFAR10_noisy_data(noise_rate=noise_rate)
        train_loader, test_loader = get_custom_loaders(data, batch_size=batch_size)

    # ✅ Check shape of first batch for debugging (only once)
    for images, _ in train_loader:
        print(f"✅ Shape check before model: {images.shape}")
        if images.shape[1] != 3:
            raise ValueError(f"Expected 3 channels but got {images.shape[1]} channels!")
        break  # ✅ Only check the first batch

    # ===============================
    # 🎯 Model, Loss, Optimizer
    # ===============================
    model = APLCNN().to(device)  # ✅ Initialize APL CNN model
    criterion = APL(alpha).to(device)  # ✅ Initialize APL criterion
    optimizer = optim.Adam(model.parameters(), lr=lr)  # ✅ Adam optimizer

    # ===============================
    # 🔥 Training Loop
    # ===============================
    active_loss_history, passive_loss_history = [], []
    train_acc_history, val_acc_history = [], []

    for epoch in range(num_epochs):
        running_loss, running_ce, running_mae = 0.0, 0.0, 0.0
        correct_train, total_train = 0, 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # ✅ Forward pass
            outputs = model(images)
            loss, ce_loss, mae_loss = criterion(outputs, labels)

            # ✅ Backprop and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ✅ Accumulate running losses
            running_loss += loss.item()
            running_ce += ce_loss
            running_mae += mae_loss


            # ✅ Accuracy tracking
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # ✅ Save active/passive losses for plotting
            active_loss_history.append(ce_loss)
            passive_loss_history.append(mae_loss)

            # ✅ Print stats every `print_every` steps
            if (i + 1) % print_every == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
                    f"APL Loss: {running_loss / print_every:.4f}, "
                    f"CE: {running_ce / print_every:.4f}, MAE: {running_mae / print_every:.4f}"
                )
                running_loss, running_ce, running_mae = 0.0, 0.0, 0.0

        # ===============================
        # ✅ Compute training accuracy for this epoch
        # ===============================
        train_acc = 100 * correct_train / total_train
        train_acc_history.append(train_acc)

        # ===============================
        # 🧪 Validation Accuracy
        # ===============================
        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = 100 * correct_val / total_val
        val_acc_history.append(val_acc)
        model.train()

        # ✅ Print accuracy per epoch for debugging
        print(
            f"✅ Epoch [{epoch+1}/{num_epochs}] - Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%"
        )

    # ===============================
    # ✅ Return All 6 Values
    # ===============================
    return (
        model,
        test_loader,
        active_loss_history,
        passive_loss_history,
        train_acc_history,
        val_acc_history,
    )
