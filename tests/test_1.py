import shap
import pytest 

from decima2 import model_feature_importance
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler



import torch
import torch.nn as nn
import torch.optim as optim


def test_RFC_scikit():
    X_adult, y_adult = shap.datasets.adult()
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_adult, y_adult, test_size=0.20, random_state=42)
    model1 = RandomForestClassifier(max_depth=100, random_state=42)
    model1.fit(X_train1, y_train1)
    explanation = model_feature_importance(X_test1,y_test1,model1,output='text')
    assert len(explanation)==len(X_test1.columns)

def test_RFR_scikit():
    def generate_regression_data():
        X_regression, y_regression = make_regression(
            n_samples=500, n_features=10, n_targets=1, noise=0.1, random_state=42
        )
        X_regression_df = pd.DataFrame(X_regression, columns=[f"feature_{i}" for i in range(10)])
        return X_regression_df, y_regression

    # Example usage
    X_regression, y_regression = generate_regression_data()

    X_train3, X_test3, y_train3, y_test3 = train_test_split(X_regression, y_regression, test_size=0.20, random_state=42)
    model3 = RandomForestRegressor(max_depth=100, random_state=42)
    model3.fit(X_train3, y_train3)

    explanation = model_feature_importance(X_test3,y_test3,model3,output='text')
    assert len(explanation)==len(X_test3.columns)

def test_keras_classification():

    # Set random seed for reproducibility
    np.random.seed(42)

    # Parameters
    num_samples = 1000
    num_features = 10

    # Generate random features
    X_classification = np.random.rand(num_samples, num_features)

    # Generate binary labels (0 or 1)
    # For example, using a simple rule: if the sum of features is above a threshold, label as 1
    Y_classification = (X_classification.sum(axis=1) > num_features / 2).astype(int)

    # Convert to DataFrame for better visualization
    X_class_df = pd.DataFrame(X_classification, columns=[f'Feature_{i+1}' for i in range(num_features)])
    Y_class_df = pd.Series(Y_classification, name='Target')


    def train_keras_classification_model(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Build Keras model
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train the model
        model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0)

        num_features = 10
        X_scaled_df = pd.DataFrame(X_test_scaled, columns=[f'Feature_{i+1}' for i in range(num_features)])
        
        return model, X_scaled_df, y_test

    # Train Keras classification model
    keras_classification_model, X_keras_classification_test, y_keras_classification_test = train_keras_classification_model(X_class_df, Y_class_df)


    #explanation_app = explanations.model_explanations(X_test1,y_test1,model1,output='text')
    #print(explanation_app)

    explanation = model_feature_importance(X_keras_classification_test, y_keras_classification_test,keras_classification_model,output='text')
    #print("got app back")
    #if __name__ == '__main__':
    #    explanation_app.run_server()
    assert len(explanation)==len(X_keras_classification_test.columns)
    

def test_keras_classification_samples_big():
    print("in here")
    # Set random seed for reproducibility
    np.random.seed(42)

    # Parameters
    num_samples = 100000
    num_features = 10

    # Generate random features
    X_classification = np.random.rand(num_samples, num_features)

    # Generate binary labels (0 or 1)
    # For example, using a simple rule: if the sum of features is above a threshold, label as 1
    Y_classification = (X_classification.sum(axis=1) > num_features / 2).astype(int)

    # Convert to DataFrame for better visualization
    X_class_df = pd.DataFrame(X_classification, columns=[f'Feature_{i+1}' for i in range(num_features)])
    Y_class_df = pd.Series(Y_classification, name='Target')


    def train_keras_classification_model(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Build Keras model
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train the model
        model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0)

        num_features = 10
        X_scaled_df = pd.DataFrame(X_test_scaled, columns=[f'Feature_{i+1}' for i in range(num_features)])
        
        return model, X_scaled_df, y_test

    # Train Keras classification model
    keras_classification_model, X_keras_classification_test, y_keras_classification_test = train_keras_classification_model(X_class_df, Y_class_df)


    #explanation_app = explanations.model_explanations(X_test1,y_test1,model1,output='text')
    #print(explanation_app)

    explanation = model_feature_importance(X_keras_classification_test, y_keras_classification_test,keras_classification_model,output='text')
    #print("got app back")
    #if __name__ == '__main__':
    #    explanation_app.run_server()
    assert len(explanation)==len(X_keras_classification_test.columns)
    


##########################################################################################################################################



##########################################################################################################################################

"""print("tensorflow_regression_model")

# Generate synthetic data
np.random.seed(42)  # For reproducibility
num_samples = 1000
num_features = 10

# Create a DataFrame with random numbers as features
X = pd.DataFrame(np.random.rand(num_samples, num_features), columns=[f'feature_{i}' for i in range(num_features)])

# Create a target variable with some random noise
y = X.sum(axis=1) + np.random.normal(0, 0.1, num_samples)  # Target is the sum of features with noise

# Display the first few rows of the DataFrame
print(X.head())
print(y[:5])  # Display the first few target values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the regression model
model = Sequential([
    Dense(64, activation='relu', input_shape=(num_features,)),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression (single output)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')  # Mean Squared Error for regression

# Train the model
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=1)


# Convert to DataFrame for better visualization
X_df = pd.DataFrame(X_test_scaled, columns=[f'Feature_{i+1}' for i in range(num_features)])
Y_df = pd.Series(y_test, name='Target')

explanation_app = explanations.model_explanations(X_df,Y_df,model)
print(explanation_app)


if __name__ == '__main__':
    explanation_app.run_server()


##########################################################################################################################################


print("Torch model Regression")

# Step 1: Generate synthetic data
np.random.seed(42)  # For reproducibility
num_samples = 1000
num_features = 10

# Create a DataFrame with random numbers as features
X = pd.DataFrame(np.random.rand(num_samples, num_features), columns=[f'feature_{i}' for i in range(num_features)])

# Create a target variable with some random noise
y = X.sum(axis=1) + np.random.normal(0, 0.1, num_samples)  # Target is the sum of features with noise

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  # Reshape to match output shape

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)  # Reshape to match output shape

# Step 3: Define the PyTorch regression model
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.hidden1 = nn.Linear(num_features, 64)  # Input layer to first hidden layer
        self.hidden2 = nn.Linear(64, 32)             # First hidden layer to second hidden layer
        self.output = nn.Linear(32, 1)               # Second hidden layer to output layer

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

# Instantiate the model
model = RegressionModel()

# Step 4: Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Zero the gradients
    outputs = model(X_train_tensor)  # Forward pass
    loss = criterion(outputs, y_train_tensor)  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights

    if (epoch + 1) % 10 == 0:  # Print every 10 epochs
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 6: Evaluate the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation
    test_outputs = model(X_test_tensor)  # Forward pass
    test_loss = criterion(test_outputs, y_test_tensor)  # Compute loss

print(f'Test Loss: {test_loss.item():.4f}')

X_df = pd.DataFrame(X_test_scaled, columns=[f'Feature_{i+1}' for i in range(num_features)])
Y_df = pd.Series(y_test, name='Target')


explanation_app = explanations.model_explanations(X_df,Y_df,model)
print("got app back")
if __name__ == '__main__':
    explanation_app.run_server()"""




##########################################################################################################################################

"""print("Multi Class Classification Torch")

# Step 1: Generate a synthetic multi-class classification dataset
def generate_multiclass_data(num_samples=1000, num_features=20, num_classes=3):
    X, y = make_classification(n_samples=num_samples, 
                               n_features=num_features, 
                               n_informative=15, 
                               n_redundant=5, 
                               n_classes=num_classes, 
                               n_clusters_per_class=1, 
                               random_state=42)
    # Create a DataFrame for features
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(num_features)])
    return X_df, y

# Step 2: Train a PyTorch model for multi-class classification
class MultiClassClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultiClassClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here; use Softmax in loss function
        return x

def train_model(X, y, num_classes):
    # Step 2.1: Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 2.2: Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 2.3: Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Long for classification

    # Step 2.4: Instantiate the model, define loss and optimizer
    model = MultiClassClassifier(input_size=X_train.shape[1], num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Step 2.5: Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(X_train_tensor)  # Forward pass
        loss = criterion(outputs, y_train_tensor)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        if (epoch + 1) % 10 == 0:  # Print every 10 epochs
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return model, scaler, X_test_scaled, y_test

# Generate data
X, y = generate_multiclass_data(num_samples=1000, num_features=20, num_classes=3)

# Train the model
model, scaler, X_test_scaled, y_test = train_model(X, y, num_classes=3)

# Step 3: Evaluate the model on test set
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    test_outputs = model(X_test_tensor)  # Forward pass
    _, predicted = torch.max(test_outputs.data, 1)  # Get predicted class


X_df = pd.DataFrame(X_test_scaled, columns=[f'Feature_{i+1}' for i in range(20)])
Y_df = pd.Series(y_test, name='Target')


explanation_app = explanations.model_explanations(X_df,Y_df,model)
print("got app back")
if __name__ == '__main__':
    explanation_app.run_server()"""







