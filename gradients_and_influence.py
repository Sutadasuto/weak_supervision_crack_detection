import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy


## Define influence measure functions
# Automatic gradient, hessian respects dimensions but seems incorrect
def automatic_grads(x, y, model):
    layer = model.layers[-2]
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            y_hat = model(x)
            loss = binary_crossentropy(y, y_hat)
        grads = tf.squeeze(tape2.jacobian(loss, layer.kernel))
    hessians = tf.squeeze(tape1.jacobian(grads, layer.kernel))
    return grads, hessians


# Results seem correct, but first order grad needs to be calculated manually
def semi_automatic_grads(x, y, model):
    layer = model.layers[-2]
    with tf.GradientTape() as tape:
        # Calculate grads manually with analytical solution
        y_hat = model(x)
        z = layer(x)
        L_y_hat = -(y / y_hat) + (1 - y) / (1 - y_hat)
        y_hat_z = np.exp(-z) / (1 + np.exp(-z)) ** 2

        grads = tf.concat([L_y_hat * y_hat_z * x[:, feat:feat + 1] for feat in range(x.shape[-1])], 1)
    hessians = tf.squeeze(tape.jacobian(grads, layer.kernel))
    return grads, hessians


# Grad and second order grad analytical solutions
def grads(x, y, model):
    y_hat = model(x)  # Prediction
    layer = model.layers[-2]  # Fully connected layer (the network has one fully conected layer and an activation layer)
    z = layer(x)
    L_y_hat = -(y / y_hat) + (1 - y) / (1 - y_hat)  # dL(y, y_hat)/d(y_hat), for L(y, y_hat) = binary crossentropy
    y_hat_z = tf.math.exp(-z) / (1 + tf.math.exp(-z)) ** 2  # d(y_hat)/d(z) = y_hat*(1-y_hat), for y_hat(z) = sigmoid(z)
    # grad_Theta = [dL/dTheta_1 ... dL/dTheta_m]', for m input features in X
    # dL/dTheta_m = (dL/dy_hat) * (dy_hat/dz) * (dz/dTheta_m), z = Theta' * X
    grads = tf.concat([L_y_hat * y_hat_z * x[:, feat:feat + 1] for feat in range(x.shape[-1])],
                      1)  # d(Theta' * X)/dTheta_m = x_m
    # hessian
    # d''L/(dTheta_m dTheta_n) = (d''L/dy_hat'') * (dy_hat/dz)^2 * (dz/dTheta_m) * (dz/dTheta_n), z = Theta' * X
    L_2_y_hat = (y / y_hat ** 2) + (1 - y) / (1 - y_hat) ** 2
    hessians = tf.concat([
        # Correction factor added as in the paper
        tf.expand_dims(tf.concat(
            [y_hat_z ** 2 * L_2_y_hat * x[:, m:m + 1] * x[:, n:n + 1] + 0.1 * float(m == n) for m in
             range(x.shape[-1])], -1), -1)
        # tf.expand_dims(tf.concat([y_hat_z**2 * L_2_y_hat * x[:, m:m+1] * x[:, n:n+1] for m in range(x.shape[-1])], -1), -1)
        for n in range(x.shape[-1])], -1)
    return grads, hessians


# Influence of z_m on each data point
def influence(m, grads, hess):
    # See Koh, P. W., & Liang, P. (2017). Understanding Black-box Predictions via Influence Functions.
    # ArXiv:1703.04730 [Cs, Stat]. http://arxiv.org/abs/1703.04730
    try:
        z_m_var = tf.linalg.matmul(tf.linalg.inv(hess[m]), tf.transpose(grads[m, None]))
    except:
        # Assuming non-invertable exception. Assuming matrix is not invertible because determinant is very small
        z_m_var = tf.ones((2,1)) * 1e10
    return tf.linalg.matmul(-grads, z_m_var)


def avg_influence(m, grads, hess):
    return tf.reduce_mean(influence(m, grads, hess))


def get_determination_scores(data_summary):
    crossentropies = binary_crossentropy(data_summary[:, 2:3], data_summary[:, 3:4])
    influence_scores = data_summary[:, 4]
    return (crossentropies * influence_scores).numpy()
