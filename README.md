# FIR_CNN_LSTM
CNN-LSTM model for exteremely low-resolution FIR sequences 

Accuracy (LOSOCV): 96.98%

|           | walk | sitdown | standup | falling | no action |
|:---------:|:----:|:-------:|:-------:|:-------:|:---------:|
|    **walk**   |  349 |    6    |    4    |    0    |      1    |
|   **sitdown** |   3  |   327   |    10   |    3    |     17    |
|  **standup**  |   2  |    3    |   343   |    0    |     12    |
|  **falling**  |   2  |    4    |    0    |   353   |     1     |
| **no action** |   4  |    4    |    0    |    0    |    1072   |


Accuracy (LOSOCV): 91.63%

|           | walk | sitdown | standup | falling | sitting | lying | standing |
|:---------:|:----:|:-------:|:-------:|:-------:|:---------:|:---------:|:---------:|
|    **walk**   |  355 |    0    |    1     |    0    |      0    |      1    |      3   |
|   **sitdown** |  1 |    308    |    45    |    1    |      4    |      0    |      1    |
|  **standup**  |  1 |    25    |    32     |    3    |      2    |      0    |      7    |
|  **falling**  |  0 |    8    |    6       |    339    |      0    |      7    |      0   |
|  **sitting**  |  0 |    0    |    0       |    0    |      318    |      2    |      40    |
|  **lying**    |  0 |    1    |    0       |    5    |      4    |      347    |      3    |
|  **standing** |  0 |    0    |    4       |    0    |      35    |      1   |      320    |

To do:

- [ ] log confusion matrix and accuracy for the branches 
- [x] experiments on 5 & 7 classess
- [x] check out cosine loss (negative/neutral impact on accuracy)
- [x] pretraining
- [x] train the streams separately
