# FIR_CNN_LSTM
CNN-LSTM model for exteremely low-resolution FIR sequences 

Accuracy (LOOCV): 96.98%

|           | walk | sitdown | standup | falling | no action |
|:---------:|:----:|:-------:|:-------:|:-------:|:---------:|
|    **walk**   |  349 |    6    |    4    |    0    |      1    |
|   **sitdown** |   3  |   327   |    10   |    3    |     17    |
|  **standup**  |   2  |    3    |   343   |    0    |     12    |
|  **falling**  |   2  |    4    |    0    |   353   |     1     |
| **no action** |   4  |    4    |    0    |    0    |    1072   |


To do:

- [ ] more augmentation methods: shifting
- [x] check out cosine loss (negative/neutral impact on accuracy)
- [x] pretraining
- [x] train the streams separately
