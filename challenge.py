#%%
# Import our dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import matplotlib.pyplot as plt

#%%
# Import our input dataset
charity_df = pd.read_csv('charity_data.csv')
charity_df.head()

# %%
# Check the unique value counts to see if binning is required
charity_df.APPLICATION_TYPE.value_counts()

# %%
charity_df.CLASSIFICATION.value_counts()

# %%
# Check the unique value counts to see if binning is required
classification_counts = charity_df.CLASSIFICATION.value_counts()
# Visualize the value counts
# classification_counts.plot.density()

#%%
# Determine which values to replace
# 150 20 72.4
# 500 20 72.3
# 150 50 72.0
# 150 30 72.3
# 75 20 71.7
# 50 20 72.1
# 50 10 72.0
# 25 20 72.4
# 10 20 72.3

replace_classification = list(classification_counts[classification_counts < 50].index)

# Replace in DataFrame
for CLASSIFICATION in replace_classification:
    charity_df.CLASSIFICATION = charity_df.CLASSIFICATION.replace(CLASSIFICATION,"OTHER")


# Check to make sure binning was successful
charity_df.CLASSIFICATION.value_counts()

# %%
# Check the unique value counts to see if binning is required
application_counts = charity_df.APPLICATION_TYPE.value_counts()
# Visualize the value counts
# classification_counts.plot.density()

#%%
# # Determine which values to replace
# replace_application = list(application_counts[application_counts < 20].index)

# # Replace in DataFrame
# for APPLICATION_TYPE in replace_application:
#     charity_df.APPLICATION_TYPE = charity_df.APPLICATION_TYPE.replace(APPLICATION_TYPE,"OTHER")


# # Check to make sure binning was successful
# charity_df.APPLICATION_TYPE.value_counts()

#%%
# Determine which values to replace
# replace_ask = list(ask_counts[ask_counts > 500000000].index)

# index_Names = charity_df[charity_df['ASK_AMT'] > 5000000000].index
# charity_df.drop(index_Names, inplace=True)

# Replace in DataFrame
# for ASK_AMT in replace_ask:
# charity_df.ASK_AMT = charity_df.ASK_AMT.drop(replace_ask)


# Check to make sure binning was successful
# charity_df.ASK_AMT.value_counts()

# %%
# # Get names of indexes for which rare Application types
# dropID = ['T2', 'T12', 'T13', 'T14', 'T15', 'T17', 'T24', 'T25', 'T29']
# indexNames = []

# def drop_row_func(drop_row):
#     for i in range(len(drop_row)):
#         indexNames = charity_df[charity_df['APPLICATION_TYPE'] == drop_row[i]].index
#         charity_df.drop(indexNames , inplace=True)
#         # indexNames.append(rows)
#         # print(rows)
#         # print(indexNames)
#     return charity_df

# drop_row_func(dropID)

# %%
# Check to make sure binning was successful
charity_df.APPLICATION_TYPE.value_counts()

# %%
# Plot ASK_AMT to determine bins
# charity_df.plot(kind='scatter',x='EIN',y='ASK_AMT',color='red')
# plt.show()
# %%
# Bin ASK_AMT
# bins = [0, 6000, 10000, 15000, 25000, 35000, 50000, 65000, 75000, 85000, 100000, 250000, 500000, 1000000, 50000000, 9000000000]
# labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# bins = [0, 6000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000, 50000000, 9000000000]
# labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# bins = [0, 6000, 10000, 25000, 50000, 9000000000]
# labels = [1, 2, 3, 4, 5]
# charity_df['ASK_AMT_BINS'] = pd.cut(charity_df['ASK_AMT'], 5, precision=10000, labels=labels, duplicates='drop')
# charity_df['ASK_AMT_BINS'] = pd.cut(charity_df['ASK_AMT'], bins=bins, labels=labels)
# charity_df.head()

# %%
charity_df.dtypes

# %%
# Drop classification column
# charity_df_cleaned = charity_df.drop(columns=['NAME', 'CLASSIFICATION'])
# charity_df_cleaned = charity_df.drop(columns=['EIN','NAME', 'ASK_AMT'])
charity_df_cleaned = charity_df.drop(columns=['EIN','NAME'])
# charity_df_cleaned = charity_df.drop(columns=['NAME'])


# %%
# Generate our categorical variable list
charity_cat = charity_df_cleaned.dtypes[charity_df_cleaned.dtypes == "object"].index.tolist()

# Check the number of unique values in each column
charity_df_cleaned[charity_cat].nunique()

# %%
# Create a OneHotEncoder instance
enc = OneHotEncoder(sparse=False)

# Fit and transform the OneHotEncoder using the categorical variable list
encode_df = pd.DataFrame(enc.fit_transform(charity_df_cleaned[charity_cat]))

# Add the encoded variable names to the DataFrame
encode_df.columns = enc.get_feature_names(charity_cat)
encode_df.head()

# %%
# Merge one-hot encoded features and drop the originals
charity_df_merge = charity_df_cleaned.merge(encode_df,left_index=True, right_index=True)
charity_df_merge = charity_df_merge.drop(charity_cat,1)
charity_df_merge.head()

# %%
# Drop SPECIAL_CONSIDERATIONS_N column becuase it's redundant
charity_df_merge = charity_df_merge.drop(columns='SPECIAL_CONSIDERATIONS_N')
charity_df_merge.head()
# %%
# Remove loan status target from features data
y = charity_df_merge.IS_SUCCESSFUL
X = charity_df_merge.drop(columns=["IS_SUCCESSFUL"])

# %%
X.head()

# Split training/test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# %%
# Create a random forest classifier.
rf_model = RandomForestClassifier(n_estimators=128, random_state=78)

# Fitting the model
rf_model = rf_model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test_scaled)
print(f" Random forest predictive accuracy: {accuracy_score(y_test,y_pred):.3f}")

# %%
# Define the model - deep neural net
# 12 12 6 72.4
# 12 12 72.6
# 16 12 72.1
# 10 10 72.1
# 10 12 72.0
# x*3 72.4

number_input_features = len(X_train_scaled[0])
# hidden_nodes_layer1 = number_input_features*4
hidden_nodes_layer1 = 12
hidden_nodes_layer2 = 12
# hidden_nodes_layer3 = 6

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(
    tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu")
)

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))
# nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="sigmoid"))


# Third hidden layer
# nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer3, activation="relu"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Compile the Sequential model together and customize metrics
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
fit_model = nn.fit(X_train_scaled, y_train, epochs=10)

# Evaluate the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

# %%
# Create a DataFrame containing training history
history_df = pd.DataFrame(fit_model.history, index=range(1,len(fit_model.history["loss"])+1))

# Plot the loss
history_df.plot(y="loss")

# %%
# Plot the accuracy
history_df.plot(y="accuracy")

# %%
