import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = '/content/Dataset .csv'
df = pd.read_csv('/content/Dataset .csv')


rows, columns = df.shape
print(f'The dataset has {rows} rows and {columns} columns.')


missing_values = df.isnull().sum()
print('Missing values in each column:\n', missing_values)


numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())


df = df.dropna()


table_booking_percentage = (df['Has Table booking'] == 'Yes').mean() * 100
online_delivery_percentage = (df['Has Online delivery'] == 'Yes').mean() * 100

print(f'Percentage of restaurants offering table booking: {table_booking_percentage:.2f}%')
print(f'Percentage of restaurants offering online delivery: {online_delivery_percentage:.2f}%')


avg_rating_with_table_booking = df[df['Has Table booking'] == 'Yes']['Aggregate rating'].mean()
avg_rating_without_table_booking = df[df['Has Table booking'] == 'No']['Aggregate rating'].mean()

print(f'Average rating of restaurants with table booking: {avg_rating_with_table_booking:.2f}')
print(f'Average rating of restaurants without table booking: {avg_rating_without_table_booking:.2f}')


ratings_comparison = df.groupby('Has Table booking')['Aggregate rating'].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(x='Has Table booking', y='Aggregate rating', data=ratings_comparison)
plt.title('Average Ratings of Restaurants with and without Table Booking')
plt.show()


online_delivery_by_price = df.groupby('Price range')['Has Online delivery'].value_counts(normalize=True).unstack() * 100

print('Availability of online delivery by price range:\n', online_delivery_by_price)


plt.figure(figsize=(10, 6))
online_delivery_by_price.plot(kind='bar', stacked=True)
plt.title('Availability of Online Delivery by Price Range')
plt.ylabel('Percentage')
plt.show()
2)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = {
    'Restaurant Name': ['Restaurant A', 'Restaurant B', 'Restaurant C', 'Restaurant D', 'Restaurant E'],
    'Price Range': [2, 3, 2, 1, 3],
    'Aggregate Rating': [4.2, 3.8, 4.5, 3.6, 4.0],
    'Rating Color': ['Green', 'Yellow', 'Green', 'Orange', 'Yellow']
}

df = pd.DataFrame(data)


most_common_price_range = df['Price Range'].mode()[0]


price_range_ratings = df.groupby('Price Range').agg(
    average_rating=('Aggregate Rating', 'mean'),
    count=('Restaurant Name', 'count')
).reset_index()


highest_avg_rating = price_range_ratings['average_rating'].max()
price_range_with_highest_rating = price_range_ratings.loc[price_range_ratings['average_rating'] == highest_avg_rating, 'Price Range'].values[0]

color_for_highest_rating = df.loc[df['Price Range'] == price_range_with_highest_rating, 'Rating Color'].mode()[0]


print(f'The most common price range among all the restaurants is: {most_common_price_range}')
print('Average rating for each price range:')
print(price_range_ratings)
print(f'The color representing the highest average rating among different price ranges is: {color_for_highest_rating}')


plt.figure(figsize=(10, 6))


plt.subplot(1, 2, 1)
sns.countplot(x='Price Range', data=df, palette='viridis')
plt.title('Most Common Price Range')
plt.xlabel('Price Range')
plt.ylabel('Count')


plt.subplot(1, 2, 2)
sns.barplot(x='Price Range', y='average_rating', data=price_range_ratings, palette='viridis')
plt.title('Average Rating by Price Range')
plt.xlabel('Price Range')
plt.ylabel('Average Rating')

plt.tight_layout()
plt.show()

3)
import pandas as pd


data = {
    'Restaurant Name': ['Restaurant A', 'Restaurant B', 'Restaurant C'],
    'Address': ['123 Street Name, City', '456 Another St, City', '789 Last Rd, City'],
    'Has Table booking': ['Yes', 'No', 'Yes'],
    'Has Online delivery': ['No', 'Yes', 'No']
}

df = pd.DataFrame(data)


df['Restaurant Name Length'] = df['Restaurant Name'].apply(len)
df['Address Length'] = df['Address'].apply(len)


df['Has Table Booking'] = df['Has Table booking'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Has Online Delivery'] = df['Has Online delivery'].apply(lambda x: 1 if x == 'Yes' else 0)


df.drop(['Has Table booking', 'Has Online delivery'], axis=1, inplace=True)


print(df)
4)
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


data = {
    'Restaurant Name': ['Restaurant A', 'Restaurant B', 'Restaurant C', 'Restaurant D', 'Restaurant E'],
    'Address': ['123 Street Name, City', '456 Another St, City', '789 Last Rd, City', '101 First Ave, City', '202 Second Blvd, City'],
    'Has Table booking': ['Yes', 'No', 'Yes', 'No', 'Yes'],
    'Has Online delivery': ['No', 'Yes', 'No', 'Yes', 'No'],
    'Aggregate Rating': [4.2, 3.8, 4.5, 3.6, 4.0]
}

df = pd.DataFrame(data)


df['Restaurant Name Length'] = df['Restaurant Name'].apply(len)
df['Address Length'] = df['Address'].apply(len)


df['Has Table Booking'] = df['Has Table booking'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Has Online Delivery'] = df['Has Online delivery'].apply(lambda x: 1 if x == 'Yes' else 0)


df.drop(['Restaurant Name', 'Address', 'Has Table booking', 'Has Online delivery'], axis=1, inplace=True)


X = df.drop('Aggregate Rating', axis=1)
y = df['Aggregate Rating']


models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}


loo = LeaveOneOut()


for model_name, model in models.items():
    y_true, y_pred = [], []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred.append(model.predict(X_test)[0])
        y_true.append(y_test.values[0])

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f'{model_name} - Mean Squared Error: {mse:.2f}, R^2 Score: {r2:.2f}')
5)
import pandas as pd
import numpy as np


data = {
    'Restaurant Name': ['Restaurant A', 'Restaurant B', 'Restaurant C', 'Restaurant D', 'Restaurant E'],
    'Cuisines': ['Italian, Chinese', 'Mexican', 'Italian', 'Chinese, Indian', 'Indian, Mexican'],
    'Aggregate Rating': [4.2, 3.8, 4.5, 3.6, 4.0],
    'Votes': [200, 150, 250, 100, 300]
}

df = pd.DataFrame(data)



df_cuisines = df.assign(Cuisines=df['Cuisines'].str.split(', ')).explode('Cuisines')


cuisine_stats = df_cuisines.groupby('Cuisines').agg(
    average_rating=('Aggregate Rating', 'mean'),
    total_votes=('Votes', 'sum'),
    count=('Cuisines', 'count')
).reset_index()


most_popular_cuisines = cuisine_stats.sort_values(by='total_votes', ascending=False)


highest_rated_cuisines = cuisine_stats.sort_values(by='average_rating', ascending=False)


print("Most Popular Cuisines Based on Total Votes:")
print(most_popular_cuisines)

print("\nCuisines with the Highest Average Ratings:")
print(highest_rated_cuisines)


import matplotlib.pyplot as plt


plt.figure(figsize=(12, 6))
plt.bar(most_popular_cuisines['Cuisines'], most_popular_cuisines['total_votes'], color='skyblue')
plt.xlabel('Cuisines')
plt.ylabel('Total Votes')
plt.title('Most Popular Cuisines Based on Total Votes')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(12, 6))
plt.bar(highest_rated_cuisines['Cuisines'], highest_rated_cuisines['average_rating'], color='lightgreen')
plt.xlabel('Cuisines')
plt.ylabel('Average Rating')
plt.title('Cuisines with the Highest Average Ratings')
plt.xticks(rotation=45)
plt.show()
6)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = {
    'Restaurant Name': ['Restaurant A', 'Restaurant B', 'Restaurant C', 'Restaurant D', 'Restaurant E'],
    'Cuisines': ['Italian, Chinese', 'Mexican', 'Italian', 'Chinese, Indian', 'Indian, Mexican'],
    'City': ['City1', 'City2', 'City1', 'City3', 'City2'],
    'Aggregate Rating': [4.2, 3.8, 4.5, 3.6, 4.0],
    'Votes': [200, 150, 250, 100, 300]
}

df = pd.DataFrame(data)


df_cuisines = df.assign(Cuisines=df['Cuisines'].str.split(', ')).explode('Cuisines')


plt.figure(figsize=(10, 6))
sns.histplot(df['Aggregate Rating'], bins=10, kde=True)
plt.title('Distribution of Aggregate Ratings')
plt.xlabel('Aggregate Rating')
plt.ylabel('Frequency')
plt.show()


cuisine_ratings = df_cuisines.groupby('Cuisines').agg(
    average_rating=('Aggregate Rating', 'mean'),
    count=('Cuisines', 'count')
).reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(x='Cuisines', y='average_rating', data=cuisine_ratings, palette='viridis')
plt.title('Average Ratings of Different Cuisines')
plt.xlabel('Cuisines')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)
plt.show()


city_ratings = df.groupby('City').agg(
    average_rating=('Aggregate Rating', 'mean'),
    count=('City', 'count')
).reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(x='City', y='average_rating', data=city_ratings, palette='coolwarm')
plt.title('Average Ratings of Different Cities')
plt.xlabel('City')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Votes', y='Aggregate Rating', data=df)
plt.title('Votes vs. Aggregate Rating')
plt.xlabel('Votes')
plt.ylabel('Aggregate Rating')
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(x='Cuisines', y='Aggregate Rating', data=df_cuisines, palette='Set2')
plt.title('Box Plot of Ratings by Cuisine')
plt.xlabel('Cuisines')
plt.ylabel('Aggregate Rating')
plt.xticks(rotation=45)
plt.show()


sns.pairplot(df)
plt.suptitle('Pair Plot of Features', y=1.02)
plt.show()
