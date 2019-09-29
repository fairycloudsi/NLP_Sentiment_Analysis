# NLP Sentiment Analysis 


In this we build a deep LSTM model to achieve NLP sentiment analysis. The input is a set of sentences and the output is the label from 5 emojis. 

## Data
After word to vector map, we represent each sentence into a vector on length 50. Shape of train data is (132,50) and shape of test data is (56,50). Shape of label of train data is (132,5), where emojis preprocessed by one hot encoding. 

## Model

### [Pretrained Layer GloVe (global vectors for word representation)](https://nlp.stanford.edu/projects/glove/) 
We use GloVe as a pretrained layer to map each word in the sentence to a vector of length 50. The corresponding vector for one sentence is the average for the vector of each word. This gives the vector representation of a sentence. The emoji label is encoding to vector by one-hot encoding. 

### Neural Networks 
We transfor each sentence to vector by pretrained layer GloVe. A dense neural networks is built with input dimenstion 50, hidden layers with 50 neurons and output dimension 5. The loss is categorical crossentropy and optimizer is adam. After 500 epochs training, we achieved accuracy 1.00 for training data and 0.8571 for validating data. From the figure, it is overfitting. 
![NN Performance](/pic/performance_NN.png)
For details, check [sentiment analysis 2.](sentiment_analysis_p2_Huiwen.ipynb)

One specific problem is this model cannot predict the sentence "I dont't like it" correctly since it ignore the negative word "not". The right label for this sentence is emoji "disappointed" while the predicted label is "smile". 


### 2-layer Deep LSTM Model
The 2 -layer deep LSTM model has inptu layer and pretrained embedding layer same as NN model.  After that, it has 2-lyaer LSTM which combines one LSTM with 128 neurons and 0.5 recurrent dropout rate and another dropout layer with rate 0.8. Another LSTM layer with 128 neurons and 0.5 recurrent dropout rate and the same dropout layer with rate 0.8. The output layer is dense with 5 neurons and softmax activation function. 

The architecture of 2-layer deep LSTM model is as follows. 

| Layer (type)  | Output Shape | Param number| 
| -------------- |:----------------:| -----------------:|
| input 1 (InputLayer) | (None, 10) | 0                 |
|embedding (Embedding) | (None, 10, 50) | 20000050 |
|lstm (LSTM) | (None, 10, 128) | 91648 |
|dropout (Dropout) | (None, 10, 128) | 0 |
|lstm 1 (LSTM) | (None, 128) | 131584 |
| dropout 1 (Dropout) | (None, 128) | 0 |
| dense (Dense) |(None,5) | 645 |

Total params: 20,223,927
Trainable params: 223,877
Non-trainbale params: 20,000,050

The model has a lot of parameters but fortunately 20,000,050 of them is non-trainable which is in the pre-trained embedding layer. 

After 100 epochs training, we reach 0.9773 accuracy for 132 training data and 0.8393 for 56 validation data. By this model, we sucessfully get rid of overfitting issue. 

![LSTM Performance](/pic/performance_LSTM.png)

The benefits of LSTM model is as follows. 
* Successfully get rid of overfitting issue by using 4 dropout. 
* Take the negative word into account by using recurrent model. Get the correct label for the sentence "I am not feeling happy". 
* By using pretrained word embedding GloVe, the model was trained only using 132 training data. 

For details, check [sentiment analysis 3.](sentiment_analysis_p3_Huiwen.ipynb)



