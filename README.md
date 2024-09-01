# PROJECT NAME: Impact of Population Size on Bike Sales Profit using Linear Regression

## OVERVIEW
This project focuses on learning to use the cost function and gradient descent to find the line of best fit for a linear regression model. It involves using Python libraries like NumPy, Pandas, and Matplotlib to implement linear regression from scratch and visualize the results.

## TABLE OF CONTENTS
1. Installation
2. Usage
3. Features
4. Documentation
5. Credits

## INSTALLATION 

### Prerequisites
- Python 3.10.9 (this is the version used for development and testing)
- Third-party libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`

### Steps
1. Clone the repository:
    git clone https://github.com/ocampbell378/LinearRegressionWithNumPy.git

2. Install the required libraries:
    pip install -r requirements.txt


## USAGE
To run the project, use the following command:
    python main.py

## FEATURES
**Feature 1**: Load and visualize the relationship between population and profit from a dataset using scatter plots.
**Feature 2**: Define and implement the cost function for linear regression.
**Feature 3**: Prepare data for linear regression by adding a bias term and calculating the initial cost.
**Feature 4**: Implement gradient descent to minimize the cost function and find the optimal parameters.
**Feature 5**: Visualize the cost function as a 3D surface plot to understand how the cost changes with different parameters.
**Feature 6**: Plot the linear regression fit on the data and make predictions for specific population sizes.

## DOCUMENTATION
### Modules and Functions
- **main.py**: Contains the primary logic for processing, visualizing data, and implementing linear regression.
- `cost_function(x, y, theta)`: Calculates the cost of using theta as the parameters for linear regression.
- `gradient_descent(x, y, theta, alpha, iterations)`: Performs gradient descent to learn theta.
- `predict(x, theta)`: Makes predictions using the learned theta values.
- Additional visualization functions are used to plot the data, cost function, and linear regression fit.

## CREDITS
- Developed by Owen Campbell
- This project was guided by the "Linear Regression with NumPy and Python" course by Snehan Kekre on Coursera.