from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Classification
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_scaled)

print("Cluster Centers:\n", kmeans.cluster_centers_)
