import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"C:\Users\Kartthik\Desktop\Personal\ML\Lab07\hello.csv")

print(df)

# Assume you have a dataset with features (X) and labels (y)
# Replace these with your actual dataset
X, y = datasets.load_iris(return_X_y=True)

class_label_1 = 'Total' 
class_label_2 = 'Rating'  

# Filter the dataset to include only the selected classes
X_filtered = X[(y == class_label_1) | (y == class_label_2)]
y_filtered = y[(y == class_label_1) | (y == class_label_2)]

# Check if the filtered dataset is empty
if len(X_filtered) == 0:
    print("Filtered dataset is empty. Adjust filtering conditions.")
else:
    # Adjust test_size based on the dataset size
    test_size = 0.2 if len(X_filtered) >= 5 else 0.1  # Adjust as needed

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=test_size, random_state=42)

    # Train a Support Vector Machine classifier
    svm_classifier = SVC(kernel='linear', C=1)  # Linear kernel for simplicity
    svm_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = svm_classifier.predict(X_test)

    # Evaluate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)