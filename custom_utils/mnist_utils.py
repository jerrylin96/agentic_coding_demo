"""MNIST utilities for training a CNN with Flax NNX.

Provides reusable, well-structured classes for:
- Data loading and preprocessing (MNISTDataLoader)
- Model definition (CNN)
- Training and evaluation (Trainer)
- Inference and visualization (Predictor)
- Model export (ModelExporter)
"""

import tensorflow_datasets as tfds
import tensorflow as tf
import optax
from flax import nnx
from functools import partial
from typing import Optional
from IPython.display import clear_output
import matplotlib.pyplot as plt
from orbax.export import JaxModule, ExportManager, ServingConfig


# ---------------------------------------------------------------------------
# Private JIT-compiled step functions (module-level for clean nnx.jit usage)
# ---------------------------------------------------------------------------

def _loss_fn(model, rngs, batch):
    logits = model(batch['image'], rngs)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']
    ).mean()
    return loss, logits


@nnx.jit
def _train_step(model, optimizer, metrics, rngs, batch):
    grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, rngs, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])
    optimizer.update(model, grads)


@nnx.jit
def _eval_step(model, metrics, rngs, batch):
    loss, logits = _loss_fn(model, rngs, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])


@nnx.jit
def _pred_step(model, batch):
    logits = model(batch['image'], None)
    return logits.argmax(axis=1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class MNISTDataLoader:
    """Handles loading, preprocessing, and batching of MNIST data."""

    def __init__(self, batch_size=32, train_steps=1200, shuffle_buffer=1024, seed=0):
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.shuffle_buffer = shuffle_buffer
        tf.random.set_seed(seed)

    @staticmethod
    def _normalize(sample):
        return {
            'image': tf.cast(sample['image'], tf.float32) / 255,
            'label': sample['label'],
        }

    def load(self):
        """Load, normalize, shuffle, and batch the MNIST train/test splits.

        Returns:
            (train_ds, test_ds) — ready-to-iterate tf.data.Dataset pair.
        """
        train_ds = tfds.load('mnist', split='train')
        test_ds = tfds.load('mnist', split='test')

        train_ds = train_ds.map(self._normalize)
        test_ds = test_ds.map(self._normalize)

        train_ds = (
            train_ds
            .repeat()
            .shuffle(self.shuffle_buffer)
            .batch(self.batch_size, drop_remainder=True)
            .take(self.train_steps)
            .prefetch(1)
        )
        test_ds = (
            test_ds
            .batch(self.batch_size, drop_remainder=True)
            .prefetch(1)
        )

        return train_ds, test_ds


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CNN(nnx.Module):
    """A simple CNN model for MNIST classification."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.batch_norm1 = nnx.BatchNorm(32, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=0.025)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.batch_norm2 = nnx.BatchNorm(64, rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
        self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=0.025)
        self.linear2 = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x, rngs: Optional[nnx.Rngs] = None):
        x = self.avg_pool(nnx.relu(self.batch_norm1(self.dropout1(self.conv1(x), rngs=rngs))))
        x = self.avg_pool(nnx.relu(self.batch_norm2(self.conv2(x))))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nnx.relu(self.dropout2(self.linear1(x), rngs=rngs))
        x = self.linear2(x)
        return x


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class Trainer:
    """Encapsulates the training loop, evaluation, and metric tracking."""

    def __init__(self, model: CNN, learning_rate=0.005, momentum=0.9):
        self.model = model
        self.optimizer = nnx.Optimizer(
            model, optax.adamw(learning_rate, momentum), wrt=nnx.Param
        )
        self.metrics = nnx.MultiMetric(
            accuracy=nnx.metrics.Accuracy(),
            loss=nnx.metrics.Average('loss'),
        )
        self.metrics_history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
        }

    def train(self, train_ds, test_ds, train_steps=1200, eval_every=200):
        """Run the full training loop with periodic evaluation.

        Args:
            train_ds: Training tf.data.Dataset (batched, repeated).
            test_ds: Test tf.data.Dataset (batched).
            train_steps: Total number of training steps.
            eval_every: Evaluate and plot every N steps.
        """
        rngs = nnx.Rngs(0)

        for step, batch in enumerate(train_ds.as_numpy_iterator()):
            self.model.train()
            _train_step(self.model, self.optimizer, self.metrics, rngs, batch)

            if step > 0 and (step % eval_every == 0 or step == train_steps - 1):
                for metric, value in self.metrics.compute().items():
                    self.metrics_history[f'train_{metric}'].append(value)
                self.metrics.reset()

                self.model.eval()
                for test_batch in test_ds.as_numpy_iterator():
                    _eval_step(self.model, self.metrics, rngs, test_batch)

                for metric, value in self.metrics.compute().items():
                    self.metrics_history[f'test_{metric}'].append(value)
                self.metrics.reset()

                self.plot_metrics()

    def plot_metrics(self):
        """Clear the output and plot current training/test loss and accuracy."""
        clear_output(wait=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.set_title('Loss')
        ax2.set_title('Accuracy')
        for dataset in ('train', 'test'):
            ax1.plot(self.metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
            ax2.plot(self.metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
        ax1.legend()
        ax2.legend()
        plt.show()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class Predictor:
    """Handles model inference and result visualization."""

    def __init__(self, model: CNN):
        self.model = model
        self.model.eval()

    def predict(self, batch):
        """Return predicted class labels for a batch."""
        return _pred_step(self.model, batch)

    def visualize_predictions(self, batch, grid_shape=(5, 5)):
        """Display a grid of images with their predicted labels.

        Args:
            batch: A batch dict with 'image' and 'label' keys.
            grid_shape: (rows, cols) for the matplotlib grid.

        Returns:
            Array of predicted labels.
        """
        predictions = self.predict(batch)
        rows, cols = grid_shape
        fig, axs = plt.subplots(rows, cols, figsize=(12, 12))
        for i, ax in enumerate(axs.flatten()):
            ax.imshow(batch['image'][i, ..., 0], cmap='gray')
            ax.set_title(f'label={predictions[i]}')
            ax.axis('off')
        return predictions


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class ModelExporter:
    """Exports a trained CNN as a TF SavedModel."""

    def __init__(self, model: CNN):
        self.model = model

    def export(self, output_dir='/tmp/mnist_export'):
        """Export the model to a TF SavedModel directory.

        Args:
            output_dir: Path to write the SavedModel.

        Returns:
            The output directory path.
        """
        def exported_predict(model, y):
            return model(y, None)

        jax_module = JaxModule(self.model, exported_predict)
        sig = [tf.TensorSpec(shape=(1, 28, 28, 1), dtype=tf.float32)]
        export_mgr = ExportManager(jax_module, [
            ServingConfig('mnist_server', input_signature=sig)
        ])
        export_mgr.save(output_dir)
        return output_dir
