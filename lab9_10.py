import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import AgglomerativeClustering
# from scipy.cluster.hierarchy import dendrogram, linkage
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import SequentialFeatureSelector
# from sklearn.decomposition import PCA
# from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv(r'C:\Users\Kartthik\Desktop\Personal\ML\Lab9-10\dataset.csv')
print(data)

# # Replace 'class_label' with the actual column name representing class labels in your dataset
# class_labels = data['ph']

# # Drop class labels for clustering
# data_without_labels = data.drop(columns=['ph'])

# # Impute missing values (adjust strategy based on your dataset characteristics)
# imputer = SimpleImputer(strategy='mean')
# data_without_labels_imputed = pd.DataFrame(imputer.fit_transform(data_without_labels), columns=data_without_labels.columns)

# # Standardize the data
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(data_without_labels_imputed)

# # Apply k-means clustering with k=3 (or k=5 based on your preference)
# kmeans = KMeans(n_clusters=3, random_state=42)
# data['cluster'] = kmeans.fit_predict(scaled_data)

# # If k=5 is more appropriate for your dataset, uncomment the following line and comment out the k=3 line
# # kmeans = KMeans(n_clusters=5, random_state=42)
# # data['cluster'] = kmeans.fit_predict(scaled_data)

# # Display the clustered data

# distortions = []
# k_values = range(1, 32)

# for k in k_values:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(scaled_data)
#     distortions.append(kmeans.inertia_)  # Sum of squared distances to the nearest cluster center

# # Plot the elbow curve
# plt.figure(figsize=(8, 6))
# plt.plot(k_values, distortions, marker='o', linestyle='-', color='b')
# plt.title('Elbow Method For Optimal k')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Average Euclidean Distance from Cluster Center')
# plt.show()

# # Choose the k value based on the elbow in the plot
# # In this case, visually inspect the plot and choose an appropriate value for k
# # Update the chosen_k value below
# chosen_k = 5  # Adjust based on the elbow point in the plot

# # Apply k-means clustering with the chosen k value
# kmeans = KMeans(n_clusters=chosen_k, random_state=42)
# data['cluster'] = kmeans.fit_predict(scaled_data)

# # Display the clustered data
# print(data)


# agg_clustering = AgglomerativeClustering(n_clusters=5, linkage='ward')
# data['agg_cluster'] = agg_clustering.fit_predict(scaled_data)

# # Plot the dendrogram
# plt.figure(figsize=(12, 8))
# dendrogram(linkage(scaled_data, method='ward'), labels=data.index, orientation='top', distance_sort='descending', show_leaf_counts=True)
# plt.title('Hierarchical Clustering Dendrogram')
# plt.xlabel('Sample Index')
# plt.ylabel('Distance')
# plt.show()

# class_labels = data['ph']

# # Drop class labels for feature selection
# X = data.drop(columns=['ph'])

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, class_labels, test_size=0.2, random_state=42)

# # Use RandomForestClassifier as an example classification model
# model = RandomForestClassifier(random_state=42)

# # Sequential Forward Selection
# sfs = SequentialFeatureSelector(model, n_features_to_select='best', direction='forward')
# sfs.fit(X_train, y_train)

# # Plot the number of features vs. cross-validation scores
# plt.plot(range(1, len(sfs.get_support()) + 1), sfs.get_metric_dict()[('cv_scores', 1)])
# plt.title('Sequential Forward Selection')
# plt.xlabel('Number of Features Selected')
# plt.ylabel('Cross-validation Score')
# plt.show()

# # Get the selected features
# selected_features = X.columns[sfs.get_support()]
# print(f'Selected Features: {selected_features}')

# #Replace 'class_label' with the actual column name representing class labels in your dataset
# class_labels = data['ph']

# # Drop class labels for PCA
# X = data.drop(columns=['ph'])

# # Impute missing values (adjust strategy based on your dataset characteristics)
# imputer = SimpleImputer(strategy='mean')
# X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# # Standardize the data before applying PCA
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_imputed)

# # Apply PCA
# pca = PCA()
# X_pca = pca.fit_transform(X_scaled)

# # Calculate the cumulative explained variance ratio
# cumulative_variance_ratio = pca.explained_variance_ratio_.cumsum()

# # Find the number of features needed to capture 95% of the variance
# k = next(i for i, ratio in enumerate(cumulative_variance_ratio, 1) if ratio >= 0.95)

# # Print the result
# print(f'Number of features needed to capture 95% of variance: {k}')

# # Plot the explained variance ratio
# plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='-', color='b')
# plt.title('Explained Variance Ratio')
# plt.xlabel('Number of Principal Components')
# plt.ylabel('Cumulative Explained Variance Ratio')
# plt.show()

# K = 3  # Replace with the desired number of components
# pca = PCA(n_components=K)
# X_transformed = pca.fit_transform(X_scaled)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_transformed, class_labels, test_size=0.2, random_state=42)

# # Train a machine learning model (Random Forest as an example)
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy on the transformed dataset: {accuracy:.2f}')