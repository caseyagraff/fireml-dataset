from typing import Any, Dict
from sklearn import metrics
import numpy as np
import time

from .saving import PickleSaveLoadMixin

from fireml.helpers import perimeter_loss


class Results(PickleSaveLoadMixin):
    def __init__(
        self,
        results: Dict[str, Any] = None,
        pred_tr: np.ndarray = None,
        pred_te: np.ndarray = None,
        losses: np.ndarray = None,
        save_pred=False,
        perimeter_loss=False,
        shuffle_pred=True,
    ):
        self.results = results
        self.pred_tr = pred_tr
        self.pred_te = pred_te
        self.losses = losses
        self.save_pred = save_pred
        self.shuffle_pred = shuffle_pred
        self.perimeter_loss = perimeter_loss

    @staticmethod
    def compute_results(
        model,
        data_tr=None,
        data_te=None,
        losses=None,
        save_pred=False,
        shuffle_pred=True,
    ):
        results: Dict[str, Dict[str, float]] = {"train": {}, "test": {}}

        start = time.time()
        if data_tr is not None:
            pred_tr = model.predict(data_tr.x)
        else:
            pred_tr = []

        print(time.time() - start)

        start = time.time()
        if data_te is not None:
            pred_te = model.predict(data_te.x)
        else:
            pred_te = []
        print(time.time() - start)

        print("tot", np.shape(pred_te), np.sum(pred_te))

        # if data_tr is not None:
        #     print(data_tr.y.flatten().shape, pred_tr.flatten().shape)

        start = time.time()
        if data_tr is not None:
            results["train"]["mse"] = metrics.mean_squared_error(
                data_tr.y.flatten(), pred_tr.flatten()
            )
        print("t_mse", time.time() - start)

        if data_tr is not None and len(data_tr.y) > 0:
            inds = np.arange(len(pred_tr))
            np.random.shuffle(inds)
            inds = inds[:500]

            # total, examples, _, _ = perimeter_loss.perimeter_loss(
            #    data_tr.y[inds], pred_tr[inds]
            # )
            # results["test"]["perimeter"] = total / examples

            # total, examples, _, _ = perimeter_loss.new_perimeter_loss(
            #    data_tr.x[:, 0], data_tr.y[inds], pred_tr[inds]
            # )
            # results["test"]["perimeter_new"] = total / examples

        start = time.time()
        if data_te is not None and len(data_te.y) > 0:
            results["test"]["mse"] = metrics.mean_squared_error(
                data_te.y.flatten(), pred_te.flatten()
            )
        print("te_mse", time.time() - start)

        if data_te is not None and len(data_te.y) > 0:
            inds = np.arange(len(pred_te))
            np.random.shuffle(inds)
            inds = inds[:500]

            # total, examples, _, _ = perimeter_loss.perimeter_loss(
            #    data_te.y[inds], pred_te[inds]
            # )
            # results["test"]["perimeter"] = total / examples
            # print("te perim", results["test"]["perimeter"])
            # total, examples, _, _ = perimeter_loss.new_perimeter_loss(
            #    data_te.x[inds, 0], data_te.y[inds], pred_te[inds]
            # )
            # results["test"]["perimeter_new"] = total / examples
            # print("te new_perim", results["test"]["perimeter_new"])

        return Results(
            results, pred_tr, pred_te, losses, save_pred, shuffle_pred=shuffle_pred
        )

    def save_name(self):
        return "results.pkl"

    def save_data(self):
        results = {"results": self.results}
        NUM_INDS = None

        if self.save_pred:
            if self.pred_tr is not None and len(self.pred_tr) > 0:
                num_inds = NUM_INDS if NUM_INDS is not None else len(self.pred_tr)
                inds = np.arange(len(self.pred_tr))
                if (
                    NUM_INDS is not None
                    and NUM_INDS < len(self.pred_tr)
                    and self.shuffle_pred
                ):
                    np.random.shuffle(inds)
                    self.train_inds = inds[:num_inds]
                else:
                    self.train_inds = inds

                results["train_inds"] = self.train_inds
                results["pred_tr"] = self.pred_tr[self.train_inds]

            if self.pred_te is not None and len(self.pred_te) > 0:
                inds = np.arange(len(self.pred_te))
                if (
                    NUM_INDS is not None
                    and NUM_INDS < len(self.pred_te)
                    and self.shuffle_pred
                ):
                    np.random.shuffle(inds)
                    self.test_inds = inds[:num_inds]
                else:
                    self.test_inds = inds

                results["inds"] = self.test_inds
                results["pred_te"] = self.pred_te[self.test_inds]

        if self.losses is not None:
            results["losses"] = {}
            results["losses"]["train"] = self.losses[0]
            results["losses"]["val"] = self.losses[1]

        return results

    def __str__(self):
        return str(self.results)
