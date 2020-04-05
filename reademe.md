### How many neurons and layers did you select for your neural network model? Why?
Hidden Layers: 2
    1st Layer: 12 Neurons
    2nd Layer: 12 Neurons
    Iterations: 10

I played around with many different numbers of layers and neurons. Of the top performers, this configuration was the most efficient (fewest neurons). 

### Were you able to achieve the target model performance?
The best I could achieve was around 72.5% accuracy

### What steps did you take to try and increase model performance?
Data Preperation
- Delete Columns: EIN, NAME, & SPECIAL_CONSIDERATIONS_N (redundant with SPECIAL_CONSIDERATIONS_Y)
- Scale the data
- Delete outlier data
- Bin data into various group sizes

Training:
- Experiment with activation functions: relu, sigmoid, tanh
- Try differnt amounts of hidden layers
- Increase training iterations
- Try random forests

### If you were to implement a different model to solve this classification problem, which would you choose? Why?
The dataset is a little small. If the dataset is made larger (synthetically or organically) I would stick with Deep Learning and Random Forest.

The statistical models are good for categorizing data, but may not be as good at making predictions.

I'm not certain what is best. Perhaps we could statistically categorize the data, then have a human manually process the groups, or push the groups into a machine learning model.