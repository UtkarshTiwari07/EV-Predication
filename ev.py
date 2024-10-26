#!/usr/bin/env python
# coding: utf-8

# In[4]:


#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

# Function to calculate the number of charging stations needed
def calculate_charging_stations(data, charger_capacity):
    years_array = data['Years'].values.reshape(-1, 1)
    evs_array = data['EVs'].values.reshape(-1, 1)

    # Create and fit the model
    model = LinearRegression()
    model.fit(years_array, evs_array)

    # Predict the number of EVs in the future
    future_years = np.array(range(max(years_array)[0] + 1, max(years_array)[0] + 11)).reshape(-1, 1)
    future_evs = model.predict(future_years)

    # Calculate the number of charging stations needed
    stations_needed = np.ceil(future_evs / charger_capacity)

    return future_years.flatten(), future_evs.flatten(), stations_needed.flatten()

# Load data from CSV
data = pd.read_csv()  # Replace 'your_data.csv' with the actual file path

# Parameters
charger_capacity = 135

# Calculate charging stations
years_array, future_evs, stations_needed = calculate_charging_stations(data, charger_capacity)

# Print the results
print("Years:", years_array)
print("Predicted EVs:", future_evs)
print("Charging Stations Needed:", stations_needed)

# Plot the results
plt.plot(years_array, future_evs, label="Predicted EVs")
plt.xlabel("Years")
plt.ylabel("Number of EVs")
plt.title("Predicted Number of EVs in Gujrat")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




