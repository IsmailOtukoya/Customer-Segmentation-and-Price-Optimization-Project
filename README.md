# Customer-Segmentation-and-Price-Optimization-Project
## Project Overview
This project aims to combine customer segmentation and dynamic pricing optimization to help businesses identify distinct customer groups and adjust pricing strategies to maximize profits. The project consists of two main parts:

Customer Segmentation: Using clustering techniques to segment customers based on various attributes such as Customer Lifetime Value (CLV), Price Sensitivity, and Churn Risk.
Price Optimization: Developing a pricing model to determine optimal price points and discounts that will maximize sales and profit.

# Table of Contents
- Project Objectives
- Dataset
- Project Workflow
- Customer Segmentation
- Feature Selection
- Clustering Algorithm
- Segmentation Visualization
 - Price Optimization
- Feature Selection
- Model Training
- Price and Discount Optimization
- Results
- Optimal Pricing Strategy
Usage
Future Work
Dependencies

# Project Objectives
The main objectives of the project are:

- To segment customers based on key behavioral and demographic attributes.
- To predict sales volumes based on pricing strategies (base price, discount) and maximize profit using machine learning.
- To identify optimal price points for different customer segments to enhance business performance.
Dataset

The dataset used in this project includes various features related to customer transactions, demographics, and purchasing behavior. Key features include:

- CLV (Customer Lifetime Value)
- Average Order Value
- Purchase Frequency
- Transaction Frequency
- Churn Risk
- Price Sensitivity
- Age
- Income

The dataset is preprocessed, and certain columns have been encoded or transformed to make them suitable for modeling.

Project Workflow
The project is divided into two main phases:

Phase 1: Customer Segmentation

Goal: Group customers into meaningful segments based on their purchasing behavior and demographic data.
Approach: We used clustering techniques (KMeans) for segmentation based on selected customer attributes like CLV, Price Sensitivity, and Churn Risk.

Phase 2: Price Optimization

Goal: Predict how price changes (base price and discount) impact sales volumes and find the optimal price and discount combination that maximizes profit.
Approach: A machine learning model (Random Forest Regressor) was used to predict sales volumes based on pricing strategies and customer data.

# Customer Segmentation
1. Feature Selection
We selected the following features for the segmentation process:

python
Copy code
features = df_encoded[['CLV', 'Avg_Order_Value', 'Purchase_Frequency',
                       'Transaction_Frequency', 'Churn_Risk', 'Price_Sensitivity', 'Age', 'Income']]
2. Clustering Algorithm
We used KMeans Clustering to segment customers. The number of clusters was determined using the elbow method.

python
Copy code
from sklearn.cluster import KMeans 

# Train KMeans model with the optimal number of clusters
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(features)
df_encoded['Cluster_Labels'] = kmeans.labels_
3. Segmentation Visualization
Using Plotly, we visualized the customer clusters to better understand the different groups.

python
Copy code
import plotly.express as px

fig = px.scatter(df_encoded, x='CLV', y='Price_Sensitivity', color='Cluster_Labels', 
                 hover_data=['Avg_Order_Value', 'Churn_Risk', 'Age'])
fig.show()

# Price Optimization
1. Feature Selection
The following features were used to predict the sales volume:

python
Copy code
selected_features = ['Base_Price', 'Discount', 'Final_Price', 'Competitor_Pricing', 'Profit_Margin',
                     'Transaction_Frequency', 'Units_Sold', 'Elasticity_of_Demand']
2. Model Training
We used a Random Forest Regressor to model the relationship between pricing and sales volumes.

python
Copy code
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=500, random_state=42)
rf_model.fit(X_train, y_train)
3. Price and Discount Optimization
A price and discount grid was created, and predictions were made to find the optimal price and discount combination that maximized profit.

python
Copy code
# Define ranges for price and discount
price_range = np.linspace(df_encoded['Base_Price'].min(), df_encoded['Base_Price'].max(), 100)
discount_range = np.linspace(df_encoded['Discount'].min(), df_encoded['Discount'].max(), 100)

# Find the optimal combination
optimal_combination = price_discount_grid.loc[price_discount_grid['Profit'].idxmax()]
Results
Optimal Pricing Strategy:
Optimal Base Price: $100.00
Optimal Discount: 10%
Maximum Profit: $246.48
Predicted Sales Volume: 12 units
These results indicate that the business can achieve maximum profitability at a base price of $100.00 with a 10% discount, selling approximately 12 units.
