from sklearn.tree import DecisionTreeClassifier  # Import the DecisionTreeClassifier class for creating a decision tree model
from sklearn.tree import plot_tree  # Import the plot_tree function for visualizing the decision tree
from sklearn import tree  # Import the tree module which contains utilities for decision trees
from sklearn import datasets  # Import the datasets module to access built-in datasets
from sklearn.metrics import confusion_matrix  # Import the confusion_matrix function to evaluate the classifier
import graphviz  # Import graphviz to visualize the decision tree

# Load the Iris dataset
irisset = datasets.load_iris()
X = irisset.data  # Feature data (sepal length, sepal width, petal length, petal width)
Y = irisset.target  # Target labels (species of Iris flowers: setosa, versicolor, virginica)

# Create an instance of the DecisionTreeClassifier
cf = DecisionTreeClassifier()

# Train the decision tree classifier on the Iris dataset
cf.fit(X, Y)

# Predict the target labels for the same feature data
Ypred = cf.predict(X)

# Compute the confusion matrix to evaluate the classifier
# The confusion matrix compares the true labels (Y) with the predicted labels (Ypred)
cmat = confusion_matrix(Y, Ypred)

# Visualize the decision tree
# decision_tree=cf: the trained decision tree classifier
# feature_names: names of the features
# class_names: names of the target classes
# filled=True: colors the nodes to indicate the class they predict
# precision=4: sets the precision of the numbers in the nodes
# rounded=True: rounds the corners of the nodes
decPlot = plot_tree(decision_tree=cf, 
                    feature_names=["sepal_length", "sepal_width", "petal_length", "petal_width"], 
                    class_names=["setosa", "versicolor", "virginica"], 
                    filled=True, 
                    precision=4, 
                    rounded=True)

# Export a text representation of the decision tree
# cf: the trained decision tree classifier
# feature_names: names of the features
text_representation = tree.export_text(cf, 
                                       feature_names=["sepal_length", "sepal_width", "petal_length", "petal_width"])
print(text_representation)

# Export the decision tree to Graphviz format and create a Graphviz source object for visualization
# cf: the trained decision tree classifier
# out_file=None: no output file is specified, so the output is returned as a string
# feature_names: names of the features
# class_names: names of the target classes
# filled=True: colors the nodes to indicate the class they predict
# rounded=True: rounds the corners of the nodes
# special_characters=True: allows for special characters in the output
dot_data = tree.export_graphviz(cf, 
                                out_file=None, 
                                feature_names=["sepal_length", "sepal_width", "petal_length", "petal_width"],  
                                class_names=["setosa", "versicolor", "virginica"],  
                                filled=True, 
                                rounded=True,  
                                special_characters=True)

# Create a Graphviz source object from the dot_data
graph = graphviz.Source(dot_data)

#The confusion matrix cmat is 100% accurate in this case because the decision
# tree classifier is evaluated on the same data that it was trained on.
# This is often referred to as "training accuracy," which is typically very high
# or even perfect for models like decision trees, which can overfit to the training data.
