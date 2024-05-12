def custom_loss_with_regularization(l2_reg_lambda=0.01):
    def custom_loss(y_true, y_pred):
        # Binary cross-entropy loss
        binary_crossentropy_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        # Regularization term
        regularization_loss = 0.0
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
                regularization_loss += tf.nn.l2_loss(layer.kernel)
        
        # Total loss
        total_loss = binary_crossentropy_loss + l2_reg_lambda * regularization_loss
        return total_loss

    return custom_loss

