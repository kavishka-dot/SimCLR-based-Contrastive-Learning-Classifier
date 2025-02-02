
def train_classifier(model, train_loader, optimizer, criterion, device, epochs=10):
    '''Train linear classifier'''
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct, total = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}, Train Accuracy: {accuracy:.2f}%")