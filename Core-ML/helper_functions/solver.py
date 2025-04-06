from __future__ import print_function, division
from future import standard_library

standard_library.install_aliases()
from builtins import range
from builtins import object
import os
import pickle as pickle
import numpy as np

from helper_functions import optim


class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training classification
    models. The Solver performs stochastic gradient descent using different
    update rules defined in optim.py.

    The solver accepts both training and validation data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.
    """

    def __init__(self, model, data, **kwargs):
        """
        Construct a new Solver instance.
        Required arguments:
        - model: A model object conforming to the API described above
        - data: A dictionary of training and validation data containing:
          'X_train': Array, shape (N_train, d_1, ..., d_k) of training images
          'X_val': Array, shape (N_val, d_1, ..., d_k) of validation images
          'y_train': Array, shape (N_train,) of labels for training images
          'y_val': Array, shape (N_val,) of labels for validation images
        """
        self.loss_history = []
        self.ce_loss_history = []
        self.nce_loss_history = []
        self.apl_loss_history = []
        self.active_loss_history = []  # For APL active loss
        self.passive_loss_history = [] 


        self.ce_train_acc_history = []
        self.ce_val_acc_history = []
        self.nce_train_acc_history = []
        self.nce_val_acc_history = []
        self.apl_train_acc_history = []
        self.apl_val_acc_history = []

        self.model = model
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]

        # Unpack keyword arguments
        self.update_rule = kwargs.pop("update_rule", "sgd")
        self.optim_config = kwargs.pop("optim_config", {})
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)
        self.num_train_samples = kwargs.pop("num_train_samples", 1000)
        self.num_val_samples = kwargs.pop("num_val_samples", None)

        self.checkpoint_name = kwargs.pop("checkpoint_name", None)
        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        # Validate and assign update rule
        if not hasattr(optim, self.update_rule):
            raise ValueError(f'Invalid update_rule "{self.update_rule}"')
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()

    def _reset(self):
        """Set up variables for optimization and bookkeeping."""
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self, loss_type):
        """Perform a single gradient update step for the specified loss type."""
        num_train = self.X_train.shape[0]

        # ✅ Randomly sample a mini-batch
        batch_mask = np.random.choice(num_train, self.batch_size, replace=False)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # =========================
        # 🎯 Compute Loss and Gradients
        # =========================
        if loss_type == "apl":
            # ✅ Unpack APL loss correctly (returns 4 values)
            loss, grads, active_loss_val, passive_loss_val = self.model.loss(
                X_batch, y_batch, loss_type=loss_type
            )

            # 📊 Store active and passive loss values
            self.active_loss_history.append(active_loss_val)
            self.passive_loss_history.append(passive_loss_val)
            self.apl_loss_history.append(loss)  # ✅ Store overall APL loss
        else:
            # ✅ Standard CE or NCE loss (returns 2 values)
            loss, grads = self.model.loss(X_batch, y_batch, loss_type=loss_type)

        # =========================
        # 📊 Store Loss in the Correct History
        # =========================
        loss_history_map = {
            "ce": self.ce_loss_history,
            "nce": self.nce_loss_history,
            "apl": self.apl_loss_history,
        }
        if loss_type in loss_history_map:
            loss_history_map[loss_type].append(loss)

        # =========================
        # 🔄 Parameter Update with Gradient Descent
        # =========================
        for p, w in self.model.params.items():
            if p not in grads:  # Skip if no gradient for this parameter
                continue

            dw = grads[p]
            config = self.optim_configs[p]

            # ✅ Use the update rule (SGD, Adam, RMSProp, etc.)
            next_w, next_config = self.update_rule(w, dw, config)

            # ✅ Update model parameters and optimizer state
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config


    def _save_checkpoint(self):
        """Save model checkpoints if a checkpoint name is provided."""
        if self.checkpoint_name is None:
            return
        
        checkpoint = {
            "model": self.model,
            "update_rule": self.update_rule,
            "lr_decay": self.lr_decay,
            "optim_configs": self.optim_configs,
            "batch_size": self.batch_size,
            "num_train_samples": self.num_train_samples,
            "num_val_samples": self.num_val_samples,
            "epoch": self.epoch,
            "ce_loss_history": self.ce_loss_history,
            "nce_loss_history": self.nce_loss_history,
            "apl_loss_history": self.apl_loss_history,
            "ce_train_acc_history": self.ce_train_acc_history,
            "nce_train_acc_history": self.nce_train_acc_history,
            "apl_train_acc_history": self.apl_train_acc_history,
            "ce_val_acc_history": self.ce_val_acc_history,
            "nce_val_acc_history": self.nce_val_acc_history,
            "apl_val_acc_history": self.apl_val_acc_history,
            "best_val_acc": self.best_val_acc,
            "best_params": self.best_params,
        }
        
        filename = f"{self.checkpoint_name}_epoch_{self.epoch}.pkl"
        
        if self.verbose:
            print(f'Saving checkpoint to "{filename}"')
        
        with open(filename, "wb") as f:
            pickle.dump(checkpoint, f)

    def check_accuracy(self, X, y, num_samples=None, batch_size=100, loss_type="ce"):
        """Check accuracy of the model on given data for different loss types."""
        valid_loss_types = ["ce", "nce", "apl"]
        if loss_type not in valid_loss_types:
            raise ValueError(f"Invalid loss_type '{loss_type}'. Must be one of {valid_loss_types}")

        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples, replace=False)
            N = num_samples
            X = X[mask]
            y = y[mask]

        num_batches = max(N // batch_size, 1)
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, N)

            scores = self.model.loss(X[start:end], loss_type=loss_type)

            if isinstance(scores, tuple):
                scores = scores[1]

            y_pred.append(np.argmax(scores, axis=1))

        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)
        return acc * 100.0

    def _train_with_loss_type(self, loss_type):
        """
        Run optimization to train the model with the specified loss type.
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        # ✅ Map correct loss history for the specified loss_type
        loss_history = (
            self.ce_loss_history
            if loss_type == "ce"
            else self.nce_loss_history
            if loss_type == "nce"
            else self.apl_loss_history
        )

        for t in range(num_iterations):
            # ✅ Perform one gradient update step
            self._step(loss_type)

            # ✅ Retrieve latest loss and append to history (Fixed order)
            if loss_type == "apl":
                latest_loss = self.apl_loss_history[-1] if self.apl_loss_history else None
            else:
                latest_loss = loss_history[-1] if loss_history else None

            # ✅ Print loss at intervals if verbose
            if self.verbose and t % self.print_every == 0 and latest_loss is not None:
                if loss_type == "apl":
                    active_loss = self.active_loss_history[-1] if self.active_loss_history else 0.0
                    passive_loss = self.passive_loss_history[-1] if self.passive_loss_history else 0.0
                    print(
                        f"(Iteration {t + 1} / {num_iterations}) APL Loss: {latest_loss:.4f} "
                        f"[NCE: {active_loss:.4f}, MAE: {passive_loss:.4f}]"
                    )
                else:
                    print(
                        f"(Iteration {t + 1} / {num_iterations}) {loss_type.upper()} loss: {latest_loss:.4f}"
                    )

            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]["learning_rate"] *= self.lr_decay

            first_it = t == 0
            last_it = t == num_iterations - 1
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(
                    self.X_train, self.y_train, num_samples=self.num_train_samples, loss_type=loss_type
                )
                val_acc = self.check_accuracy(
                    self.X_val, self.y_val, num_samples=self.num_val_samples, loss_type=loss_type
                )

                # ✅ Update accuracy history
                if loss_type == "ce":
                    self.ce_train_acc_history.append(train_acc)
                    self.ce_val_acc_history.append(val_acc)
                elif loss_type == "nce":
                    self.nce_train_acc_history.append(train_acc)
                    self.nce_val_acc_history.append(val_acc)
                elif loss_type == "apl":
                    self.apl_train_acc_history.append(train_acc)
                    self.apl_val_acc_history.append(val_acc)

                self._save_checkpoint()

                if self.verbose:
                    print(
                        f"(Epoch {self.epoch} / {self.num_epochs}) {loss_type.upper()} train acc: {train_acc:.2f}; {loss_type.upper()} val acc: {val_acc:.2f}"
                    )

                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {k: v.copy() for k, v in self.model.params.items()}

        self.model.params = self.best_params




    # ============================
    # 🎯 Convenience Functions
    # ============================
    def train_ce(self):
        """Train the model using only CE loss."""
        self._train_with_loss_type("ce")

    def train_nce(self):
        """Train the model using only NCE loss."""
        self._train_with_loss_type("nce")

    def train_apl(self):
        """Train the model using only APL loss."""
        self._train_with_loss_type("apl")
