import numpy as np


class CrossEntropyLoss():
    """Cross entropy loss function."""

    def forward(self, Y, Y_hat):
        """Computes the loss for predictions `Y_hat` given one-hot encoded labels
        `Y`.

        Parameters
        ----------
        Y      one-hot encoded labels of shape (batch_size, num_classes)
        Y_hat  model predictions in range (0, 1) of shape (batch_size, num_classes)

        Returns
        -------
        a single float representing the loss
        """
        # Clip Y_hat to prevent log(0)
        Y_hat = np.clip(Y_hat, 1e-12, 1.0)
        
        # Compute cross-entropy loss
        loss = -np.sum(Y * np.log(Y_hat)) / Y.shape[0]
        
        return loss

    def backward(self, Y, Y_hat):
        """Backward pass of cross-entropy loss.
        NOTE: This is correct ONLY when the loss function is SoftMax.

        Parameters
        ----------
        Y      one-hot encoded labels of shape (batch_size, num_classes)
        Y_hat  model predictions in range (0, 1) of shape (batch_size, num_classes)

        Returns
        -------
        the gradient of the cross-entropy loss with respect to the vector of
        predictions, `Y_hat`
        """
        # Clip Y_hat to prevent division by zero
        Y_hat = np.clip(Y_hat, 1e-12, 1.0)
        
        # Compute the gradient
        grad = -Y / Y_hat / Y.shape[0]
        

        return grad