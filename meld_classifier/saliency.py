# interpretability methods
# general API: saliency(model, X, targets, **kwargs)
# returns: score for each element in X


def integrated_gradients(model, X, targets, kwargs={}):
    """integrated gradients"""
    from alibi.explainers import IntegratedGradients

    # kwargs for ig
    layer = kwargs.pop("layer", None)
    method = kwargs.pop("method", "gausslegendre")
    n_steps = kwargs.pop("n_steps", 25)
    internal_batch_size = kwargs.pop("internal_batch_size", 100)
    ig = IntegratedGradients(
        model, layer=layer, method=method, n_steps=n_steps, internal_batch_size=internal_batch_size
    )
    explanation = ig.explain(X, baselines=None, target=targets)
    return explanation.data["attributions"][0]


def vanilla_backprop(model, X, targets, target="gt"):
    """vanilla gradients"""
    import tensorflow as tf

    bce = tf.keras.losses.BinaryCrossentropy()

    inputs = tf.cast(X, tf.float32)
    gt_outputs = tf.cast(targets, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(inputs)
        outputs = model(inputs)
        loss = bce(gt_outputs, outputs)
    grads = tape.gradient(loss, inputs)
    return grads.numpy()
