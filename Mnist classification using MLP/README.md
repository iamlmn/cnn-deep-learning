# Mnist classification using keras
Basic MLP on mnist image dataset for classifying digits.

> Test accuracy -97.64 after training for 10 Epochs with dropout 0.2.

##### Model summary: 


| Layer (type)           |      Output Shape     |      Param #    |
| -----------------------|:---------------------:| ---------------:|
| flatten_3 (Flatten)    |      (None, 784)      |      0          |
| dense_2 (Dense)        |      (None, 512)      |      401920     |
| dropout_1 (Dropout)    |      (None, 512)      |      0          |
| dense_3 (Dense)        |      (None, 512)      |      262656     |
| dropout_2 (Dropout)    |      (None, 512)      |      0          |
| dense_4 (Dense)        |      (None, 10)       |      5130       |

---

Total params: 669,706
Trainable params: 669,706
Non-trainable params: 0
