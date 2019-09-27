# NLP Sentiment Analysis 


In this we build a deep LSTM model to achieve NLP sentiment analysis. The input is a set of sentences and the output is the label from 5 emojis. 

## Data
After word to vector map, we represent each sentence into a vector on length 50. Shape of train data is (132,50) and shape of test data is (56,50). Shape of label of train data is (132,5), where emojis preprocessed by one hot encoding. 

## Model

### Neural Networks 
A dense neural networks is built with input dimenstion 50, hidden layers with 50 neurons and output dimension 5. The loss is categorical crossentropy and optimizer is adam. After 500 epochs training, we achieved accuracy 1.00 for training data and 0.8571 for validating data. From the figure, it is overfitting. 
![NN Performance](/pic/performance_nn.png)
For details, check 

### 2-layer Deep LSTM Models


