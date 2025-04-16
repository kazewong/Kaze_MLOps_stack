import fire
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from jaxtyping import Array, Float, Int

from kazemlstack.model.ViT import VisionTransformer


def test_model():
    model = VisionTransformer(
        rng_seed=0,
        embed_dim=16,
        hidden_dim=64,
        num_heads=4,
        num_channels=3,
        num_layers=4,
        patch_size=4,
        num_patches=64,
    )

    test_input = jnp.zeros((10, 3, 16, 16))
    def single_eval(input):
        return model(input)

    test_output = nnx.jit(nnx.vmap(single_eval))(test_input)
    print(test_output.shape)  # Should be (10, 64, 16)

def loss_fn(model: VisionTransformer, batch: Float[Array, "n_batch n_channel n_height n_width"]):
    logits = model(batch["image"])
    loss = nnx.CrossEntropyLoss()(logits, batch["label"])
    return loss, logits

@nnx.jit
def train_step(model: VisionTransformer, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    optimizer.update(grads)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])

@nnx.jit
def eval_step(model: VisionTransformer, metrics: nnx.MultiMetric, batch):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])

def train_model(
    num_epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    momentum: float = 0.9,
):
    model = VisionTransformer(
        rng_seed=0,
        embed_dim=16,
        hidden_dim=64,
        num_heads=4,
        num_channels=3,
        num_layers=4,
        patch_size=4,
        num_patches=64,
    )
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=learning_rate, b1=momentum))
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average('loss'),
    )



if __name__ == "__main__":
    fire.Fire(test_model)
