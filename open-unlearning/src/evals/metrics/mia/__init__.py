"""Min-k membership-inference metric used by the TOFU privleak score."""

import numpy as np
from sklearn.metrics import roc_auc_score

from evals.metrics.base import unlearning_metric
from evals.metrics.mia.min_k import MinKProbAttack


def mia_auc(attack_cls, model, data, collator, batch_size, **kwargs):
    """Run an attack on forget/holdout sets and report its ROC AUC."""
    attack_args = {
        "model": model,
        "collator": collator,
        "batch_size": batch_size,
        **kwargs,
    }
    output = {
        "forget": attack_cls(data=data["forget"], **attack_args).attack(),
        "holdout": attack_cls(data=data["holdout"], **attack_args).attack(),
    }
    forget_scores = [
        item["score"] for item in output["forget"]["value_by_index"].values()
    ]
    holdout_scores = [
        item["score"] for item in output["holdout"]["value_by_index"].values()
    ]
    scores = np.asarray(forget_scores + holdout_scores)
    labels = np.asarray([0] * len(forget_scores) + [1] * len(holdout_scores))
    auc_value = roc_auc_score(labels, scores)
    output["auc"] = auc_value
    output["agg_value"] = auc_value
    return output


@unlearning_metric(name="mia_min_k")
def mia_min_k(model, **kwargs):
    return mia_auc(
        MinKProbAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
        k=kwargs["k"],
    )
