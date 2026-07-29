from metrics.base_metrics import BaseMetrics
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision, 
    BinaryRecall
)

class SegmentationMetrics(BaseMetrics):
    def __init__(self):
        super().__init__()
        self.zero_division = 0.0
        self._init_metrics()

    def update(self, predicted_masks, true_masks):
        for metric in self.metrics.values():
            metric.update(predicted_masks, true_masks)

    def _init_metrics(self):
        self.metrics['dice'] = BinaryF1Score(zero_division=self.zero_division)
        self.metrics['iou'] = BinaryJaccardIndex(zero_division=self.zero_division)
        self.metrics['precision'] = BinaryPrecision(zero_division=self.zero_division)
        self.metrics['recall'] = BinaryRecall(zero_division=self.zero_division)