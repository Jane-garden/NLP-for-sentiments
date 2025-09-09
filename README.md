# NLP-for-sentiments

## Feature Extraction

Implemented a `featureExtractor` function to convert text into feature vectors. Approaches explored:

- Bag-of-Words
- N-gram models
- Word embeddings  

Feature representations were evaluated based on classification performance.

## Classifier Training

Implemented `learnPredictor` to train a linear classifier using hinge loss:

$$
L(x,y; \mathbf{w}) = \max(0, 1 - y \cdot (\mathbf{w} \cdot \phi(x)))
$$

Gradient descent was applied to optimize model weights on the training set.

## Results

- Successfully trained the classifier on the training set.
- Evaluated on the test set and achieved accurate sentiment predictions.
- Plotted training loss curves and analyzed the model’s performance with different feature representations.

## Project Artifacts

- Source code (`model.py`, `featureExtractor.py`, etc.)
- Trained models
- Visualizations of training loss and test results
