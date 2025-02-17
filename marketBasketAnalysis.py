import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

originalData = pd.read_csv('Megastore_Dataset_Task_3 3.csv')
df = pd.DataFrame(originalData)
# remove whitespace from the column names of a DataFrame
df.columns = df.columns.str.strip()
print('print first 10 records:')
print(df.head(10))
print('Print data info:')
df.info()

# drop unused columns
df = df.drop(['InvoiceDate', 'Segment'], axis=1)

# Remove dollar signs and convert to float
df['UnitPrice'] = df['UnitPrice'].replace('[\$,]', '', regex=True).astype(float)
df['TotalCost'] = df['TotalCost'].replace('[\$,]', '', regex=True).astype(float)

# Encode CustomerOrderSatisfaction as ordinal values
satisfaction_mapping = {
    "Very Satisfied": 4,
    "Satisfied": 3,
    "Very Dissatisfied": 2,
    "Dissatisfied": 1,
    "Prefer to not respond": 0
}
df['CustomerOrderSatisfaction'] = df['CustomerOrderSatisfaction'].map(satisfaction_mapping)

# Convert categorical binary variables to numerical (Yes=1, No=0)
binary_columns = ['DiscountApplied', 'ExpeditedShipping']
df[binary_columns] = df[binary_columns].apply(lambda x: x.map({'Yes': 1, 'No': 0}))

# export cleaned data
df.to_csv("CleanData.csv", index=False)


# Select necessary columns (OrderID and ProductName for Market Basket Analysis)
basket = df.groupby(['OrderID', 'ProductName'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('OrderID')

# Convert quantities to boolean (True for 1, False for 0) - FIXED LINE
basket = basket.apply(lambda col: col.map(lambda x: x > 0))

# Apply Apriori algorithm
frequentItems = apriori(basket, min_support=0.02, use_colnames=True)

# Generate association rules
rules = association_rules(frequentItems, metric="lift", min_threshold=1)

# Display results
print("Frequent Items:")
print(frequentItems.head())

print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

# Save results to CSV
frequentItems.to_csv("frequent_items.csv", index=False)
rules.to_csv("association_rules.csv", index=False)