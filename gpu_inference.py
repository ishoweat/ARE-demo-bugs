import torch


def run_inference(inputs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.nn.Sequential(
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(),
        torch.nn.Linear(4096, 1024)
    ).to(device)

    # Recent optimization: process the complete input in one operation.
    # This becomes a problem when production traffic sends large batches.
    inputs = inputs.to(device)

    with torch.no_grad():
        predictions = model(inputs)

    return predictions


def create_request_batch():
    # Represents a large production inference request.
    return torch.randn(120000, 4096)


if __name__ == "__main__":
    batch = create_request_batch()
    result = run_inference(batch)

    print(f"Inference completed: {result.shape}")
