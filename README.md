# keras_lr_finder
Plots the change of the loss function of a Keras model when the learning rate is exponentially increasing.
Will also calculate the best learning rate.
## Purpose
See details in ["Estimating an Optimal Learning Rate For a Deep Neural Network"](https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0).

## Usage
Create and compile a Keras model, then execute this code:

```python
# model is a Keras model
lr_finder = LRFinder(model)

# Train a model with batch size 512 for 5 epochs
# with learning rate growing exponentially from 0.0001 to 1
lr_finder.find(x_train, y_train, start_lr=0.0001, end_lr=1, batch_size=512, epochs=5)
```

```python
# Plot the loss, ignore 20 batches in the beginning and 5 in the end
lr_finder.plot_loss(n_skip_beginning=20, n_skip_end=5)
```

![Loss function](https://cdn-images-1.medium.com/max/1600/1*HVj_4LWemjvOWv-cQO9y9g.png)

```python
# Plot rate of change of the loss
# Ignore 20 batches in the beginning and 5 in the end
# Smooth the curve using simple moving average of 20 batches
# Limit the range for y axis to (-0.02, 0.01)
lr_finder.plot_loss_change(sma=20, n_skip_beginning=20, n_skip_end=5, y_lim=(-0.01, 0.01))
```

![Rate of change of the loss function](https://cdn-images-1.medium.com/max/1600/1*87mKq_XomYyJE29l91K0dw.png)

Once the finder has picked your best learning rate, update your model to use it:
```python
# Set the learning rate of your model to the newly found one
import keras.backend as K
new_lr = lr_finder.get_best_lr(sma=4)
K.set_value(model.optimizer.lr, new_lr)
```
You can wrap this up nicely in a `LambdaCallback`, so that you periodically update your learning rate:

```python
from keras.callbacks import LambdaCallback
def find_lr(epoch, logs):
    # You may also make it more effective by only
    # running this if the loss has stopped improving a la ReduceLROnPlateau
    if epoch % 30 == 0:
        lrf = LRFinder(model)
        lrf.find(x_train,y_train, start_lr=0.0001, end_lr=1,batch_size=512,epochs=5)
        new_lr = lrf.get_best_lr(4)
        K.set_value(model.optimizer.lr, new_lr)
    
lcb = LambdaCallback(on_epoch_end=find_lr)
model.train(callbacks=[lcb],...)
```


## Contributions
Contributions are welcome. Please, file issues and submit pull requests on GitHub, or contact me directly.

## References
This code is based on:
- The method described in section 3.3 of the 2015 paper ["Cyclical Learning Rates for Training Neural Networks"](https://arxiv.org/abs/1506.01186) by Leslie N. Smith
- The implementation of the algorithm in [fastai library](https://github.com/fastai/fastai) by Jeremy Howard. See [fast.ai deep learning course](http://course.fast.ai/) for details.

