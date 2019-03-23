# Weight Initialization

1 tf.ones(),tf.zeros() as initial weights are of no use.

2 Weight Intialization works with normal distribution of variable between the range 1/root(nodes).

3 Truncated random uniform distribtion worked out better as it has a higher likelihood of picking number cose to mean.


`tf.random.truncated_normal(
    shape,
    mean=0.0,
    stddev=1.0,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
)`
