import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Linear regression interface with batch/mini-batch options
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.history = {'loss': []}

    def fit(self, X, y, batch_size=None):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iterations):
            indices = np.arange(n_samples) if batch_size is None else np.random.choice(n_samples, batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]
            
            y_pred = np.dot(X_batch, self.weights) + self.bias
            dw = (1/len(indices)) * np.dot(X_batch.T, (y_pred - y_batch))
            db = (1/len(indices)) * np.sum(y_pred - y_batch)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            self.history['loss'].append(mean_squared_error(y_batch, y_pred))

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

def main():
    housing = fetch_california_housing()
    X = housing.data[:, 0].reshape(-1, 1)  # MedInc only
    y = housing.target

    print("\nData Stats:")
    print(f"MedInc - Mean: {np.mean(X):.2f}, Median: {np.median(X):.2f}, Std: {np.std(X):.2f}")
    print(f"MedHouseVal - Mean: {np.mean(y):.2f}, Median: {np.median(y):.2f}, Std: {np.std(y):.2f}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    bgd_model = LinearRegressionGD(learning_rate=0.01, n_iterations=1000)
    bgd_model.fit(X_train_scaled, y_train)

    sgd_model = LinearRegressionGD(learning_rate=0.01, n_iterations=1000)
    sgd_model.fit(X_train_scaled, y_train, batch_size=32)

    # Evaluate
    y_pred_bgd = bgd_model.predict(X_test_scaled)
    y_pred_sgd = sgd_model.predict(X_test_scaled)
    print(f"\nR² Scores:")
    print(f"BGD: {r2_score(y_test, y_pred_bgd):.4f}")
    print(f"SGD: {r2_score(y_test, y_pred_sgd):.4f}")

    # Predict for $80k income
    sample = np.array([[8.0]])
    sample_scaled = scaler.transform(sample)
    print(f"\nPrediction for $80k income:")
    print(f"BGD: ${bgd_model.predict(sample_scaled)[0]*100000:.2f}")
    print(f"SGD: ${sgd_model.predict(sample_scaled)[0]*100000:.2f}")

    # Plot results
    plt.figure(figsize=(12, 8))
    plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Test Data')
    
    X_test_sorted = np.sort(X_test, axis=0)
    y_pred_bgd_line = bgd_model.predict(scaler.transform(X_test_sorted))
    y_pred_sgd_line = sgd_model.predict(scaler.transform(X_test_sorted))
    
    plt.plot(X_test_sorted, y_pred_bgd_line, color='red', label='BGD')
    plt.plot(X_test_sorted, y_pred_sgd_line, color='green', label='SGD')
    
    plt.xlabel('Median Income ($10,000s)')
    plt.ylabel('Median House Value ($100,000s)')
    plt.title('California Housing Prices vs. Median Income')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(bgd_model.history['loss'], label='BGD')
    plt.plot(sgd_model.history['loss'], label='SGD')
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main() 