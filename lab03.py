import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support


# Load your data from the Excel file
data = pd.read_excel(r'C:\Users\Kartthik\Desktop\Personal\Lab03\hello.xlsx')

# Use the first 10 rows and 17 columns for analysis
df = pd.DataFrame(data.iloc[:10, :17])

# Extract only the numeric columns (exclude non-numeric or non-integer columns)
numeric_columns = df.select_dtypes(include=[np.number])

# Group the DataFrame by the 'Rating' column
grouped = df.groupby('Rating')

# Initialize dictionaries to store the mean and standard deviation for each rating category
rating_means = {}
rating_std = {}

# Calculate the mean (rating centroid) and standard deviation (spread) for each rating category
for name, group in grouped:
    rating_means[name] = group[numeric_columns.columns].mean(axis=0)
    rating_std[name] = group[numeric_columns.columns].std(axis=0)

# Calculate the distance between mean vectors between rating values (Euclidean distance)
ratings = list(rating_means.keys())
num_ratings = len(ratings)

mean_distances = np.zeros((num_ratings, num_ratings))

for i in range(num_ratings):
    for j in range(i + 1, num_ratings):
        rating1 = ratings[i]
        rating2 = ratings[j]
        mean_distances[i, j] = np.linalg.norm(rating_means[rating1] - rating_means[rating2])
        mean_distances[j, i] = mean_distances[i, j]

# Print the mean and standard deviation for each rating value
for rating, mean in rating_means.items():
    print(f"Rating: {rating}, Mean Vector:", mean)
    
for rating, std in rating_std.items():
    print(f"Rating: {rating}, Standard Deviation:", std)

# Print the distance between mean vectors between rating values
for i in range(num_ratings):
    for j in range(i + 1, num_ratings):
        rating1 = ratings[i]
        rating2 = ratings[j]
        print(f"Distance between Rating {rating1} and Rating {rating2} Mean Vectors:", mean_distances[i, j])

# Extract the 'Rating' column for analysis
rating_values = df['Rating']

# Calculate the histogram
plt.hist(rating_values, bins=10)  # Adjust the number of bins as needed
plt.title('Histogram of Rating')
plt.xlabel('Rating Range')
plt.ylabel('Frequency')
plt.show()

# Calculate the mean and variance
mean_rating = rating_values.mean()
variance_rating = rating_values.var()

print(f"Mean Rating: {mean_rating}")
print(f"Variance Rating: {variance_rating}")

# Extract the 'Rating' and 'Quantity' columns for analysis
rating_values = df['Rating']
quantity_values = df['Quantity']

# Initialize a list to store the Minkowski distances for different values of 'r'
minkowski_distances = []

# Calculate Minkowski distances for r values from 1 to 10
for r in range(1, 11):
    distances = np.abs(rating_values - quantity_values) ** r
    minkowski_distance = np.sum(distances) ** (1/r)
    minkowski_distances.append(minkowski_distance)

# Create a plot to observe the nature of the Minkowski distances
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), minkowski_distances, marker='o', linestyle='-')
plt.title('Minkowski Distance vs. r (Rating vs. Quantity)')
plt.xlabel('r')
plt.ylabel('Minkowski Distance')
plt.grid(True)
plt.show()

data['Rating'] = pd.cut(data['Rating'], bins=[0, 2, 4, 6, 8, 10], labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# Select the features and the target variable
X = data[['Quantity', 'gross income']]  # Features (attributes)
y = data['Rating']  # Target variable (class labels)

# Split the data into a 60% training set and a 40% test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# You can print the shapes to verify the sizes of the training and test sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the training data
knn_classifier.fit(X_train, y_train)
# Calculate the accuracy of the classifier using the score method
accuracy = knn_classifier.score(X_test, y_test)

# Print the accuracy
print("Accuracy:", accuracy)

y_pred = knn_classifier.predict(X_test)

# Print the actual and predicted labels for the first few test vectors
print("Actual Labels:", y_test.head().tolist())
print("Predicted Labels:", y_pred[:5].tolist())

k_values = list(range(1, 12))
accuracy_scores = []

# Iterate through different values of k
for k in k_values:
    # Create a k-NN classifier with the current k value
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    
    # Train the classifier on the training data
    knn_classifier.fit(X_train, y_train)
    
    # Use the trained classifier to make predictions on the test set
    y_pred = knn_classifier.predict(X_test)
    
    # Calculate the accuracy and store it in the list
    accuracy = np.mean(y_pred == y_test)
    accuracy_scores.append(accuracy)

    # Plot the accuracy scores for different values of k
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy_scores, marker='o', linestyle='-')
plt.title('Accuracy vs. k (k-NN Classifier)')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

y_train_pred = knn_classifier.predict(X_train)
y_test_pred = knn_classifier.predict(X_test)

# Calculate the confusion matrix for both training and test sets
confusion_matrix_train = confusion_matrix(y_train, y_train_pred)
confusion_matrix_test = confusion_matrix(y_test, y_test_pred)

# Print the confusion matrix
print("Confusion Matrix (Training Data):")
print(confusion_matrix_train)
print("\nConfusion Matrix (Test Data):")
print(confusion_matrix_test)

# Calculate precision, recall, and F1-Score for both training and test sets
precision_train, recall_train, f1_score_train, _ = precision_recall_fscore_support(y_train, y_train_pred, average='weighted')
precision_test, recall_test, f1_score_test, _ = precision_recall_fscore_support(y_test, y_test_pred, average='weighted')

# Print precision, recall, and F1-Score
print("\nPrecision (Training Data):", precision_train)
print("Recall (Training Data):", recall_train)
print("F1-Score (Training Data):", f1_score_train)

print("\nPrecision (Test Data):", precision_test)
print("Recall (Test Data):", recall_test)
print("F1-Score (Test Data):", f1_score_test)
