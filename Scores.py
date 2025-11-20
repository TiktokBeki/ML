# ======================================================================
# FULL MACHINE LEARNING PROJECT WITH PYTORCH + SCIKIT-LEARN
# Predict Daily Electricity Usage
# Dataset Size: 50,000 Rows (Auto-Generated)
# EVERY SECTION AND LIBRARY EXPLAINED IN COMMENTS
# ======================================================================

# -------------------------- IMPORT LIBRARIES ---------------------------

import pandas as pd               # for handling data tables (rows & columns)
import numpy as np                # for math operations & arrays
import matplotlib.pyplot as plt   # for plotting graphs
import seaborn as sns             # for statistical plots (heatmaps etc.)

from sklearn.model_selection import train_test_split  # splits dataset
from sklearn.preprocessing import StandardScaler       # normalizes numbers
from sklearn.compose import ColumnTransformer          # transforms selected columns
from sklearn.pipeline import Pipeline                  # chains steps together
from sklearn.ensemble import RandomForestRegressor     # ML model
from sklearn.ensemble import GradientBoostingRegressor # ML model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # accuracy metrics

import joblib                   # for saving/loading trained models
import warnings
warnings.filterwarnings("ignore")

# PyTorch imports
import torch                    # main pytorch module
import torch.nn as nn           # neural network layers
import torch.optim as optim     # neural network optimizers
from torch.utils.data import TensorDataset, DataLoader # batching & dataset tools

# Set seed for stability (same random numbers each run)
np.random.seed(42)
torch.manual_seed(42)

# ======================================================================
# 1. GENERATE A LARGE SYNTHETIC DATASET (50,000 rows)
# ======================================================================

num_samples = 50000

# Seasonal temperature waves (sinusoidal = realistic yearly pattern)
days = np.arange(num_samples)
season_temp = 10 + 15 * np.sin(2 * np.pi * days / 365)

# Create realistic household data
data = pd.DataFrame({
    "day_of_week": np.random.choice(range(7), num_samples),
    "month":       np.random.choice(range(1, 13), num_samples),
    "num_people_home": np.random.randint(1, 6, num_samples),

    "temperature": season_temp + np.random.normal(0, 4, num_samples),   # realistic noise
    "humidity":    np.random.uniform(20, 90, num_samples),

    "hours_ac_used":     np.random.uniform(0, 10, num_samples),
    "hours_heater_used": np.random.uniform(0, 8, num_samples),
    "tv_hours":          np.random.uniform(0, 6, num_samples),
    "computer_hours":    np.random.uniform(0, 10, num_samples),
    "laundry_cycles":    np.random.choice([0,1,2], num_samples, p=[0.7,0.25,0.05])
})

# Adjust AC/heater usage using temperature
data["hours_ac_used"] += np.clip((data["temperature"] - 25) / 5, 0, 8)
data["hours_heater_used"] += np.clip((15 - data["temperature"]) / 4, 0, 8)

# Formula to compute electricity usage (target)
data["electricity_kWh"] = (
    0.7 * data["hours_ac_used"] +
    0.8 * data["hours_heater_used"] +
    0.6 * data["num_people_home"] +
    0.3 * data["tv_hours"] +
    0.4 * data["computer_hours"] +
    1.5 * data["laundry_cycles"] +
    0.1 * data["temperature"] -
    0.05 * data["humidity"] +
    np.random.normal(0, 1.5, num_samples)       # noise = realism
)

print("Dataset shape:", data.shape)
print(data.head())

# ======================================================================
# 2. FEATURE ENGINEERING
# ======================================================================

# Cyclical encoding for days (better than raw numbers)
data["day_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
data["day_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

# Cyclical encoding for months (Jan is close to Dec)
data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)

# Remove raw categorical numbers
data = data.drop(columns=["day_of_week", "month"])

# ======================================================================
# 3. TRAIN-TEST SPLIT
# ======================================================================

X = data.drop(columns=["electricity_kWh"])  # input features
y = data["electricity_kWh"]                 # output/label

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================================================================
# 4. NORMALIZATION FOR SCIKIT-LEARN MODELS
# ======================================================================

numeric_features = X_train.columns.tolist()

preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric_features)
])

# ======================================================================
# 5. TRADITIONAL MODELS (Random Forest + Gradient Boosting)
# ======================================================================

rf_model = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestRegressor(n_estimators=200, n_jobs=-1))
])

gb_model = Pipeline([
    ("preprocess", preprocess),
    ("model", GradientBoostingRegressor())
])

# Train models
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# ======================================================================
# 6. DEFINE EVALUATION FUNCTION
# ======================================================================

def evaluate(model, name):
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)

    print(f"\n----- {name} -----")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R²  :", r2)

# Evaluate classical ML models
evaluate(rf_model, "Random Forest")
evaluate(gb_model, "Gradient Boosting")

# ======================================================================
# 7. PREPARE DATA FOR PYTORCH (Normalized with StandardScaler)
# ======================================================================

# Fit scaler once
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Convert numpy arrays to PyTorch tensors
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Create dataset + batch loader
train_ds = TensorDataset(X_train_t, y_train_t)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

# ======================================================================
# 8. BUILD A PYTORCH NEURAL NETWORK
# ======================================================================

class ElectricityNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # Simple feed-forward neural network
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),  # Fully connected layer
            nn.ReLU(),                 # Activation
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)           # Output layer (regression)
        )

    def forward(self, x):
        return self.net(x)

# Create model
input_dim = X_train.shape[1]
model = ElectricityNN(input_dim)

# Loss function and optimizer
criterion = nn.MSELoss()          # Mean Squared Error
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ======================================================================
# 9. TRAIN PYTORCH MODEL
# ======================================================================

epochs = 8
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    # iterate batches
    for xb, yb in train_dl:
        optimizer.zero_grad()      # reset gradients
        pred = model(xb)           # forward pass
        loss = criterion(pred, yb) # compute loss
        loss.backward()            # compute gradients
        optimizer.step()           # update weights
        
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

# ======================================================================
# 10. EVALUATE PYTORCH MODEL
# ======================================================================

model.eval()
with torch.no_grad():
    preds = model(X_test_t).numpy().flatten()

mae  = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2   = r2_score(y_test, preds)

print("\n----- PyTorch Neural Network -----")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)

# ======================================================================
# 11. SAVE PYTORCH MODEL
# ======================================================================

torch.save(model.state_dict(), "pytorch_electricity_model.pth")
print("\nSaved PyTorch model.")

# ======================================================================
# 12. SAMPLE PREDICTION
# ======================================================================

sample = X_test.iloc[0:1]                 # take one example
sample_scaled = scaler.transform(sample)  # scale it
sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)
prediction = model(sample_tensor).item()

print("\nSample prediction (kWh):", prediction)

# =============================================================
# CUSTOM USER INPUT PREDICTION
# =============================================================
print("\n----- CUSTOM INPUT PREDICTION -----")

# Ask user for input
temperature = float(input("Temperature (°C): "))
humidity = float(input("Humidity (%): "))
people = int(input("Number of people home: "))
ac = float(input("Hours AC used: "))
heater = float(input("Hours heater used: "))
tv = float(input("TV hours: "))
computer = float(input("Computer hours: "))
laundry = int(input("Laundry cycles (0–2): "))
day = int(input("Day of week (0=Mon .. 6=Sun): "))
month = int(input("Month (1–12): "))

# Convert to model-ready features
custom = pd.DataFrame([{
    "temperature": temperature,
    "humidity": humidity,
    "num_people_home": people,
    "hours_ac_used": ac,
    "hours_heater_used": heater,
    "tv_hours": tv,
    "computer_hours": computer,
    "laundry_cycles": laundry,
    "day_sin": np.sin(2 * np.pi * day / 7),
    "day_cos": np.cos(2 * np.pi * day / 7),
    "month_sin": np.sin(2 * np.pi * month / 12),
    "month_cos": np.cos(2 * np.pi * month / 12),
}])

# Scale + convert to tensor
custom_scaled = scaler.transform(custom)
custom_tensor = torch.tensor(custom_scaled, dtype=torch.float32)

# Predict
model.eval()
with torch.no_grad():
    result = model(custom_tensor).item()

print("\nEstimated Electricity Usage (kWh):", result)

