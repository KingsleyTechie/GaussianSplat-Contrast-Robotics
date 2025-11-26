import torch
import numpy as np
from gaussian_scene import GaussianSplatScene
from contrastive_encoder import ContrastiveEncoder
from train_contrastive import train_step

def sample_triplets(scene, batch_size=32):
    anchors, positives, negatives = [], [], []

    for _ in range(batch_size):
        v1 = np.random.randn(3)
        v2 = v1 + np.random.normal(0, 0.1, 3)
        v3 = np.random.randn(3)

        f1 = scene.sample_view(v1)
        f2 = scene.sample_view(v2)
        f3 = scene.sample_view(v3)

        idx = np.random.randint(0, len(f1))

        anchors.append(f1[idx])
        positives.append(f2[idx])
        negatives.append(f3[idx])

    return (
        torch.tensor(np.array(anchors)),
        torch.tensor(np.array(positives)),
        torch.tensor(np.array(negatives))
    )

def main():
    scene = GaussianSplatScene(num_splats=200)
    encoder = ContrastiveEncoder()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)

    print("Starting contrastive training on synthetic Gaussian splat scene...")

    for step in range(200):
        batch = sample_triplets(scene)
        loss = train_step(encoder, optimizer, batch)

        if (step + 1) % 20 == 0:
            print(f"Step {step+1} Loss {loss:.4f}")

    print("Training complete.")

    test_view = scene.sample_view(np.array([1, 0, 1]))
    z = encoder(torch.tensor(test_view).float())
    print("Embedding vector sample:", z[0].detach().numpy())

if __name__ == "__main__":
    main()
