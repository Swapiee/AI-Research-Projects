import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree
from sklearn import tree
import graphviz

# Load the data from a CSV file into a DataFrame
purchaseData = pd.read_csv('Purchase_Logistic.csv')

# Select feature columns (independent variables) and target column (dependent variable)
X = purchaseData.iloc[:, [2, 3]]  # Features: columns at index 2 and 3
Y = purchaseData.iloc[:, 4]       # Target: column at index 4

# Split the data into training and testing sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25, random_state=0)

# Initialize the Decision Tree Classifier with a maximum depth of 4
cf = DecisionTreeClassifier(max_depth=4)

# Train the classifier on the training data
cf.fit(Xtrain, Ytrain)

# Use the trained classifier to make predictions on the test data
Ypred = cf.predict(Xtest)

# Compute the confusion matrix to evaluate the predictions
cmat = confusion_matrix(Ytest, Ypred)

# Plot the decision tree with feature names and class names
decPlot = plot_tree(decision_tree=cf, 
                    feature_names=["Age", "Salary"], 
                    class_names=["No", "Yes"], 
                    filled=True, 
                    precision=4, 
                    rounded=True)

# Export the decision tree to a text representation
text_representation = tree.export_text(cf, feature_names=["Age", "Salary"])
print(text_representation)

# Export the decision tree to Graphviz format for visualization
dot_data = tree.export_graphviz(cf, out_file=None, 
                                feature_names=["Age", "Salary"],  
                                class_names=["No", "Yes"],  
                                filled=True, 
                                rounded=True,  
                                special_characters=True)

# Create a Graphviz source object from the exported data
graph = graphviz.Source(dot_data)

# To display the graph in a Jupyter notebook, you can use:
# graph.render("decision_tree")  # Save the tree as a file (optional)
graph.view()  # Display the tree in a viewer
