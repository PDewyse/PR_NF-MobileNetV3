import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(2) # Arbitrary key
# Produce a large batch of random noise vectors
x = jax.random.normal(key, (1024, 256))
# Variance preserving Hard sigmoid function 
# y = jax.nn.relu6(x+3)* (1.0/6.0)
y = jax.nn.sigmoid(x)
# Variance preserving Hard swish function
# y = jax.nn.relu6(x+3)* (1.0/6.0) * x
# y = jax.nn.hard_swish(x)
# Take the average variance of many random batches
gamma = jnp.mean(jnp.var(y, axis=1)) ** -0.5
print(gamma)