import torch


def run_inference():
    device = torch.device("cuda")

    # BUG: Batch size is far too large for available GPU memory
    batch_size = 100000
    features = 4096

    data = torch.randn(
        batch_size,
        features,
        device=device
    )

    result = data @ data.T

    return result


if __name__ == "__main__":
    print("Starting GPU inference...")
    result = run_inference()
    print("Inference completed.")
