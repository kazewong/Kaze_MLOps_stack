import fire
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

from kazemlstack.model.ViT import VisionTransformer


def train_step():
    pass

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



def train(
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


if __name__ == "__main__":
    fire.Fire(test_model)
