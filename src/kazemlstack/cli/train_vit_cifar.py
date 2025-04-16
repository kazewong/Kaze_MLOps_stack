import fire
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from jaxtyping import Array, Float, Int
import grain.python as grain

from kazemlstack.model.ViT import VisionTransformer
from kazemlstack.dataset.cifar import CIFAR10DataSource


class VITClassification(nnx.Module):

    def __init__(
        self,
        rngs: nnx.Rngs,
        num_classes: int,  # Number of classes to classify
        embed_dim: Int,  # Dimensionality of input and attention feature vectors
        hidden_dim: int,  # Dimensionality of hidden layer in feed-forward network
        num_heads: int,  # Number of heads to use in the Multi-Head Attention block
        num_channels: int,  # Number of channels of the input (3 for RGB)
        num_layers: int,  # Number of layers to use in the Transformer
        patch_size: int,  # Number of pixels that the patches have per dimension
        num_patches: int,  # Maximum number of patches an image can have
        dropout_prob: float = 0.0,  # Amount of dropout to apply in the feed-forward network
    ):
        self.model = VisionTransformer(
            rngs=rngs,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_channels=num_channels,
            num_layers=num_layers,
            patch_size=patch_size,
            num_patches=num_patches,
            dropout_prob=dropout_prob,
        )
        self.classifier = nnx.Linear(
            in_features=embed_dim,
            out_features=num_classes,
            rngs=rngs,
        )

    def __call__(self, x: Float[Array, "n_batch n_channel n_height n_width"]):
        x = self.model(x)
        x = self.classifier(jnp.mean(x, axis=0)) # Note that this is not a very good way to do ViT classification, but here we just want something to run quickly.
        return x


def loss_fn(
    model: VisionTransformer, batch: Float[Array, "n_batch n_channel n_height n_width"]
):
    logits = nnx.vmap(model)(batch["image"])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["label"]
    ).mean()
    return loss, logits


@nnx.jit
def train_step(
    model: VisionTransformer, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch
):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    optimizer.update(grads)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])


@nnx.jit
def eval_step(model: VisionTransformer, metrics: nnx.MultiMetric, batch):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])


def train_model(
    num_epochs: int = 3,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    momentum: float = 0.9,
):
    data_source = CIFAR10DataSource()

    index_sampler = grain.IndexSampler(
        num_records=len(data_source),
        num_epochs=num_epochs,
        shard_options=grain.ShardOptions(
            shard_index=0, shard_count=1, drop_remainder=True
        ),
        shuffle=True,
        seed=0,
    )
    data_loader = grain.DataLoader(
        data_source=data_source,
        operations=[grain.Batch(batch_size=batch_size)],
        sampler=index_sampler,
        worker_count=4,
    )

    model = VITClassification(
        rngs=nnx.Rngs(0),
        num_classes=10,
        embed_dim=16,
        hidden_dim=64,
        num_heads=4,
        num_channels=3,
        num_layers=4,
        patch_size=4,
        num_patches=64,
    )
    optimizer = nnx.Optimizer(
        model, optax.adamw(learning_rate=learning_rate, b1=momentum)
    )
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for idx, batch in enumerate(data_loader):
            print(f"Batch {idx + 1}")
            # train_step(model, optimizer, metrics, batch)


if __name__ == "__main__":
    fire.Fire(train_model)

