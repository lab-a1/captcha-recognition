import torch
from tqdm import tqdm
from lib.metric_monitor import MetricMonitor
from lib.metrics import accuracy


def test(model, params, test_dataset_loader, criterion):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(test_dataset_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params["device"], non_blocking=True).float()
            t = [t.to(params["device"], non_blocking=True) for t in target]
            t0, t1, t2, t3, t4 = t
            y = model(images)
            y0, y1, y2, y3, y4 = y
            loss_0 = criterion(y0, t0)
            loss_1 = criterion(y1, t1)
            loss_2 = criterion(y2, t2)
            loss_3 = criterion(y3, t3)
            loss_4 = criterion(y4, t4)
            loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_4
            accuracy_result = accuracy(y, t)
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy_result)
            stream.set_description(f"Test. {metric_monitor}")
