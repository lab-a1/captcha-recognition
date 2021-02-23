import torch


def accuracy(output, target):
    """Mean between the predictions for the five characters."""
    accuracy_result = 0
    for y, t in zip(output, target):
        _, predicted = torch.max(y.data, 1)
        correct_predictions = (predicted == t).sum().item()
        accuracy_result += correct_predictions / t.size(0)
    return accuracy_result / len(target)
