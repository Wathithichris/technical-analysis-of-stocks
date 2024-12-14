import pandas as pd
pd.set_option('display.width', None)

data = pd.read_csv("kenyan_91days.csv")
data['Issue Date'] = pd.to_datetime(data['Issue Date'], format="%d/%m/%Y")
data['year'] = data['Issue Date'].dt.year
data['month'] = data['Issue Date'].dt.month
unique_years = data["year"].unique()
t_bills_df = data[data['Tenor']==91]
data_grouped = t_bills_df.groupby(by=['year'])['Weighted Average Rate'].mean()
average_tbill = data_grouped.sum()/ len(unique_years)
median_tbills = data_grouped.median()
print(data_grouped)
print(average_tbill)
print(median_tbills)