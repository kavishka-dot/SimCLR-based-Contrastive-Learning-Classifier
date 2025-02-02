import torch

# Test function
def test_classifier(model, test_loader, device):
    '''Test linear classifier'''
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")