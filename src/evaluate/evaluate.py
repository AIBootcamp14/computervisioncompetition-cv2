import torch
from sklearn.metrics import f1_score
import numpy as np
from typing import Tuple

def evaluate_macro_f1(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[float, np.ndarray]:
    """
    모델의 매크로 F1 스코어를 계산하고 예측 결과를 반환합니다.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return macro_f1, np.array(all_preds)