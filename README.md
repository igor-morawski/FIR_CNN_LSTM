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


Accuracy (LOSOCV): 90.87%

|           | walk | sitdown | standup | falling | sitting | lying | standing |
|:---------:|:----:|:-------:|:-------:|:-------:|:---------:|:---------:|:---------:|
|    **walk**   |  355 |    1    |    2     |    0    |      0    |      1    |      1   |
|   **sitdown** |  1 |    310    |    39    |    7    |      2    |      0    |      1    |
|  **standup**  |  1 |    37    |    318     |    1    |      0    |      0    |      3    |
|  **falling**  |  0 |    3    |    4       |    349    |      1    |      3    |      0   |
|  **sitting**  |  0 |    2    |    1       |    0    |      299    |      9    |      49    |
|  **lying**    |  0 |    0    |    0       |    6    |      5    |      346    |      3    |
|  **standing** |  0 |    0    |    6       |    0    |      40    |      1   |      313    |

To do:

- [ ] log confusion matrix and accuracy for the branches 
- [x] experiments on 5 & 7 classess
- [x] check out cosine loss (negative/neutral impact on accuracy)
- [x] pretraining
- [x] train the streams separately
