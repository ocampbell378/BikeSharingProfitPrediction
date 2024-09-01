import matplotlib.pyplot as plt 
plt.style.use('ggplot')
import numpy as np
import pandas as pd  
import seaborn as sns 
plt.rcParams['figure.figsize'] = (12, 8)

# Load dataset
data = pd.read_csv("bike_sharing_data.txt")

# Visualize the relationship between Population and Profit
ax1 = sns.scatterplot(x="Population", y="Profit", data=data)
ax1.set_title("Profit in $10,000s vs City Population in 10,000s")

# Define the cost function for linear regression
def cost_function(x, y, theta):
    m = len(y)
    y_hat = x.dot(theta)
    error = (y_hat - y) ** 2
    return 1 / (2 * m) * np.sum(error)

# Prepare data with bias term for cost calculation
m = data.Population.values.size
x = np.append(np.ones((m, 1)), data.Population.values.reshape(m, 1), axis=1)
y = data.Profit.values.reshape(m, 1)
theta = np.zeros((2,1))

# Calculate initial cost
initial_cost = cost_function(x, y, theta)
print(f"Initial cost: {initial_cost}")

# Implement gradient descent to minimize cost function
def gradient_descent(x, y, theta, alpha, iterations):
    m = len(y)
    costs = []
    for i in range(iterations):
        y_hat = x.dot(theta)
        error  = np.dot(x.transpose(), (y_hat - y))
        theta -= alpha * 1/m * error
        costs.append(cost_function(x, y, theta))
    return theta, costs

# Run gradient descent with specified parameters
theta, costs = gradient_descent(x, y, theta, alpha=0.01, iterations=2000)

# Display the final linear model
print(f"h(x) = {round(theta[0, 0], 2)} + {round(theta[1, 0], 2)}x1")

# Visualize the cost function in 3D space
from mpl_toolkits.mplot3d import Axes3D

theta_0 = np.linspace(-10, 10, 100)
theta_1 = np.linspace(-1, 4, 100)

cost_values = np.zeros((len(theta_0), len(theta_1)))

# Calculate cost for combinations of theta_0 and theta_1
for i in range(len(theta_0)):
    for j in range(len(theta_1)):
        t = np.array([theta_0[i], theta_1[j]])
        cost_values[i, j] = cost_function(x, y, t)

# Plot the cost function surface
fig = plt.figure(figsize= (12,8))
ax2 = fig.add_subplot(111, projection = '3d')

theta_0_mesh, theta_1_mesh = np.meshgrid(theta_0, theta_1)
surf = ax2.plot_surface(theta_0_mesh, theta_1_mesh, cost_values.T, cmap='viridis')
fig.colorbar(surf, shrink=0.5, aspect=5)

ax2.set_xlabel("$\\Theta_0$")
ax2.set_ylabel("$\\Theta_1$")
ax2.set_zlabel("$J(\\Theta)$")
ax2.view_init(30, 330)

plt.show()

# Plot the cost function values over iterations
plt.figure()
plt.plot(costs)
plt.xlabel("Iterations")
plt.ylabel("$J(\\Theta)$")
plt.title("Values of the Cost Function")
plt.show()

# Plot the linear regression fit on the data
theta = np.squeeze(theta)
plt.figure()
sns.scatterplot(x="Population", y="Profit", data=data)

x_value = np.array([x for x in range(5, 25)])
y_value = (x_value * theta[1]) + theta[0]
sns.lineplot(x=x_value, y=y_value)

plt.xlabel("Population in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.title("Linear Regression Fit")
plt.show()

# Function to make predictions with learned theta values
def predict(x, theta):
    y_hat = np.dot(x, theta)
    return y_hat

# Make predictions for specific population sizes
y_hat1 = predict(np.array([1, 4]), theta) * 10000
print(f"For a population of 40,000 people, the model predicts a profit of ${round(y_hat1, 2)}")

y_hat2 = predict(np.array([1, 8.3]), theta) * 10000
print(f"For a population of 83,000 people, the model predicts a profit of ${round(y_hat2, 2)}")
