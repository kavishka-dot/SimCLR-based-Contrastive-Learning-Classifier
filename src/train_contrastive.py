import torch
import torch.functional as F

# NT-Xent Loss
def nt_xent_loss(z_i, z_j, temperature=0.5):
    '''Compute NT-Xent loss'''
    batch_size = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)  # Concatenate positive pairs

    # Compute similarity
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    sim /= temperature

    # Create labels
    labels = torch.arange(batch_size, device=z.device).repeat(2)
    loss = F.cross_entropy(sim, labels)
    return loss

# Training Loop
def train_simclr(model, train_loader, optimizer, device, epochs=10):
    '''Train SimCLR model'''
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for (x_i, x_j), _ in train_loader:
            x_i, x_j = x_i.to(device), x_j.to(device)
            
            z_i = model(x_i)
            z_j = model(x_j)

            loss = nt_xent_loss(z_i, z_j)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
