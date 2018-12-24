import math
import logging

from rastervision.evaluation import ClassEvaluationItem
from rastervision.evaluation import ClassificationEvaluation

log = logging.getLogger(__name__)


def get_class_eval_item(gt_arr, pred_arr, class_id, class_map):
    # Definitions of precision, recall, and f1 taken from
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html  # noqa
    not_dont_care = (gt_arr != 0)  # By assumption
    gt = (gt_arr == class_id)
    pred = (pred_arr == class_id)
    not_gt = (gt_arr != class_id)
    not_pred = (pred_arr != class_id)

    true_pos = (gt * pred).sum()
    false_pos = (not_gt * pred * not_dont_care).sum()
    false_neg = (gt * not_pred * not_dont_care).sum()

    precision = float(true_pos) / (true_pos + false_pos)
    recall = float(true_pos) / (true_pos + false_neg)
    f1 = 2 * (precision * recall) / (precision + recall)
    count_error = int(false_pos + false_neg)
    gt_count = int(gt.sum())
    class_name = class_map.get_by_id(class_id).name

    if math.isnan(precision):
        precision = None
    else:
        precision = float(precision)
    if math.isnan(recall):
        recall = None
    else:
        recall = float(recall)
    if math.isnan(f1):
        f1 = None
    else:
        f1 = float(f1)

    return ClassEvaluationItem(precision, recall, f1, count_error, gt_count,
                               class_id, class_name)


class SemanticSegmentationEvaluation(ClassificationEvaluation):
    """Evaluation for semantic segmentation."""

    def __init__(self, class_map):
        super().__init__()
        self.class_map = class_map

    def compute(self, gt_labels, pred_labels):
        self.clear()
        for window in pred_labels.get_windows():
            log.debug('Evaluating window: {}'.format(window))
            gt_arr = gt_labels.get_label_arr(window)
            pred_arr = pred_labels.get_label_arr(window)

            eval_items = []
            for class_id in self.class_map.get_keys():
                eval_item = get_class_eval_item(gt_arr, pred_arr, class_id,
                                                self.class_map)
                eval_items.append(eval_item)

            # Treat each window as if it was a small Scene.
            window_eval = SemanticSegmentationEvaluation(self.class_map)
            window_eval.set_class_to_eval_item(
                dict(zip(self.class_map.get_keys(), eval_items)))
            window_eval.compute_avg()
            self.merge(window_eval)
