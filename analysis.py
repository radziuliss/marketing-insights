import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import pearsonr

customer_data_df = pd.read_excel('./row_data/CustomersData.xlsx')
discount_coupon_df = pd.read_csv('./row_data/Discount_Coupon.csv')
marketing_spend_df = pd.read_csv('./row_data/Marketing_Spend.csv')
online_sales_df = pd.read_csv('./row_data/Online_Sales.csv')
tax_amount_df = pd.read_excel('./row_data/Tax_amount.xlsx')

customer_data_df.head()
discount_coupon_df.head()
online_sales_df.head()
tax_amount_df.head()
marketing_spend_df.head()

print(customer_data_df.info())
print(online_sales_df.info())
print(marketing_spend_df.info())
print(tax_amount_df.info())
print(discount_coupon_df.info())



online_sales_df.isnull().values.any()
tax_amount_df.isnull().values.any()
marketing_spend_df.isnull().values.any()
customer_data_df.isnull().values.any()
discount_coupon_df.isnull().values.any()

online_sales_df.isnull().sum()

print(online_sales_df.shape)
print(tax_amount_df.shape)
print(marketing_spend_df.shape)
print(customer_data_df.shape)
print(discount_coupon_df.shape)


# online_sales_df.dtypes



## Changing format
online_sales_df['Transaction_Date'] = pd.to_datetime(online_sales_df['Transaction_Date'], format='%m/%d/%Y').dt.strftime('%d/%m/%Y')
online_sales_df['Transaction_Date'] = pd.to_datetime(online_sales_df['Transaction_Date'], format='%d/%m/%Y')

marketing_spend_df['Date'] = pd.to_datetime(marketing_spend_df['Date'], format='%m/%d/%Y').dt.strftime('%d/%m/%Y')
marketing_spend_df['Date'] = pd.to_datetime(marketing_spend_df['Date'], format='%d/%m/%Y')



customer_data_df.head()
discount_coupon_df.head()
online_sales_df.head()
tax_amount_df.head()
marketing_spend_df.head()

online_sales_df[['CustomerID', 'Avg_Price', 'Quantity', 'Delivery_Charges']].corr()

online_sales_df.describe()



# 1. Calculating Invoice amount

## Invoice Value = ((Quantity*Avg_price)*(1-Discount_pct) * (1 + GST)) + Delivery_Charges

## For this We can split this task on small steps.
## The first one is merging online_sales and Tax_amount tables it's not difficult:

temp_merg_data = pd.merge(
    online_sales_df,
    tax_amount_df,how = 'left',
    on = 'Product_Category')


## Add column 'Month' where will be months 'January', 'February' etc.
discount_coupon_df.sort_values(by = ['Month', 'Product_Category'])
temp_merg_data['Month'] = temp_merg_data['Transaction_Date'].dt.strftime('%b')


# while merging tables online_sales and discount_coupon only Product_Category is not enough because we changed % of discount coupons each month so here we need:
temp_merg_data = pd.merge(
    temp_merg_data,
    discount_coupon_df,
    how = 'left',
    on = ['Month', 'Product_Category'])  ## And now merge !using both column because the discount % is not static!


# Last step before accounting is to mark that the coupon in the transaction was used and discount will be active in that case
temp_merg_data['Discount_pct_active'] = temp_merg_data.apply(
    lambda x: x['Discount_pct'] if x['Coupon_Status'] == 'Used' else 0, axis = 1)

temp_merg_data.info()

print(temp_merg_data.isna().sum())
temp_merg_data = temp_merg_data.fillna(0)
print(temp_merg_data.isna().sum())

# And then let's count Invoice_Value. There are 2 methods: numpy array or through pandas . Let's check both methods
temp_merg_data.info()
array_temp_merg_data = temp_merg_data.to_numpy() # one array is enough

## Invoice Value = ((Quantity*Avg_price)*(1-Discount_pct/100) * (1 + GST)) + Delivery_Charges
Invoice_Value_np = ((array_temp_merg_data[:,6] * array_temp_merg_data[:,7]) * (1 - array_temp_merg_data[:,14]/100)
                    * (1 + array_temp_merg_data[:,10])) + array_temp_merg_data[:,8]

print(f'Our calculated Invoice Value is {round(np.sum(Invoice_Value_np), 2)} $')

# We also could do it with pandas:
temp_merg_data['Invoice'] = (temp_merg_data['Quantity'] * temp_merg_data['Avg_Price'] * (1 - temp_merg_data['Discount_pct_active']/100)
                             * (1+temp_merg_data['GST']) + temp_merg_data['Delivery_Charges'])
temp_merg_data['Invoice'].describe()
print(f'Our calculated Invoice Value is {temp_merg_data["Invoice"].sum():,.2f} $')


                                    # 2.  Perform Detailed exploratory analysis

### 2.1. Understanding how many total customers acquired every month

temp_merg_data['Transaction_Date_Month'] = temp_merg_data['Transaction_Date'].dt.to_period('M')

cust_amount = temp_merg_data ##.sort_values(by=['CustomerID', 'Transaction_Date', 'Min_Transaction_Date'])
cust_amount.groupby(['Transaction_Date_Month'])['CustomerID'].nunique()
cust_amount.info()

### 2.1.1 Understanding how many new customers acquired every month
new_amount_cust = temp_merg_data
new_amount_cust['Min_Transaction_Date'] = new_amount_cust.groupby('CustomerID')['Transaction_Date'].transform('min')
uniq_cust_trans = new_amount_cust[new_amount_cust['Transaction_Date'] ==
                                  new_amount_cust['Min_Transaction_Date']].drop_duplicates(subset='CustomerID') ## New user made a few transaction in one day, let's eliminate it
uniq_cust_trans.groupby(['Transaction_Date_Month'])['CustomerID'].count()

# more shorter solution:
new_amount_cust['Is_New_Customer'] = (new_amount_cust['Transaction_Date'] == new_amount_cust['Min_Transaction_Date'])
new_amount_cust[new_amount_cust['Is_New_Customer']].groupby(new_amount_cust['Transaction_Date'].dt.to_period('M'))['CustomerID'].nunique()


### 2.2. Understand the retention of customers on month on month basis
# To explore Customer Retention Rate (CRR):
# 1. We need Total at the start of month (let's it will be total cust for prev month), total at the end and new_active customer(who make more than 1 purchase)
# 2. Above We counted Customers who were new(total) in each month
# 3. Now let's find new customers who made only 1 purchase and subtract them from active customers

purchase_counts = new_amount_cust.groupby('CustomerID')['Transaction_ID'].count()
one_more_time_customers = purchase_counts[purchase_counts == 1].index
one_time_customers_ids = new_amount_cust[(new_amount_cust['CustomerID'].isin(one_more_time_customers))
                                         & (new_amount_cust['Is_New_Customer'] == True)]
new_one_time_customers = one_time_customers_ids.groupby('Transaction_Date_Month')['CustomerID'].nunique().reset_index()

current_month = cust_amount.groupby(['Transaction_Date_Month'])['CustomerID'].nunique().reset_index()
prev_month = cust_amount.groupby(['Transaction_Date_Month'])['CustomerID'].nunique().reset_index()


pre_CRR_table_accounting = pd.merge(current_month, prev_month, on='Transaction_Date_Month')
CRR_table_accounting = pd.merge(pre_CRR_table_accounting,
                                new_one_time_customers,
                                how='left',
                                on= 'Transaction_Date_Month')
CRR_table_accounting.columns = ['Transaction_Date_Month', 'Customers_curr_month', 'Customers_prev_month', 'New_One_Time_Cust']
empty_row = pd.DataFrame({
    'Transaction_Date_Month': [pd.Period('9999-01', freq='M')]
})
CRR_table_accounting = pd.concat([CRR_table_accounting, empty_row])

CRR_table_accounting['Customers_prev_month'] = CRR_table_accounting['Customers_curr_month'].shift(+1).fillna(0)
CRR_table_accounting['New_One_Time_Cust'] = CRR_table_accounting['New_One_Time_Cust'].shift(+1).fillna(0)
CRR_table_accounting['CRR'] = np.where(
    CRR_table_accounting['Customers_prev_month'] != 0,
    ((CRR_table_accounting['Customers_prev_month'] - CRR_table_accounting['New_One_Time_Cust']) / CRR_table_accounting['Customers_curr_month'] * 100),
    0
)

# np.where is good when we divide by 0 because it prevent to get inf/NaN in some places
CRR_table_accounting['% DIFF_CUST_AMOUNT_MONTH'] = np.where(
    CRR_table_accounting['Customers_prev_month'] != 0, ## condition
    (CRR_table_accounting['Customers_curr_month'] / CRR_table_accounting['Customers_prev_month'] - 1) * 100, # if yes
    0 # if no
)

CRR_table_accounting = CRR_table_accounting.fillna(0)
CRR_table_accounting = CRR_table_accounting.round({
    'Customers_curr_month': 2,
    'Customers_prev_month': 2,
    'New_One_Time_Cust': 2,
    'CRR': 2,
    '% DIFF_CUST_AMOUNT_MONTH': 2
})
print(CRR_table_accounting[['CRR', '% DIFF_CUST_AMOUNT_MONTH']])
# Using np.where we prevent having inf
# in CRR formula we subtract new customers who made only one purchase, it help us because our retention rate will not being artificially inflated by those customer



### 2.3. Build small report about company's fundamental health split on total/new customers on month on month basis
cust_amount.info()
new_amount_cust.info()
print(cust_amount)
print(new_amount_cust)

monthly_summary = cust_amount.groupby('Transaction_Date_Month').agg(
    total_cust = ('CustomerID', 'nunique'),
    total_trans = ('Transaction_ID', 'count'),
    total_invoice = ('Invoice', 'sum')
).reset_index()



monthly_summary['avg_trans_per_client'] = monthly_summary['total_trans']/monthly_summary['total_cust']
monthly_summary['avg_invoice_per_client'] = monthly_summary['total_invoice']/monthly_summary['total_cust']

print(monthly_summary)


monthly_summary_new_c = new_amount_cust[new_amount_cust['Is_New_Customer']].groupby('Transaction_Date_Month').agg(
    total_new_cust = ('CustomerID', 'nunique'),
    total_trans = ('Transaction_ID', 'count'),
    total_invoice = ('Invoice', 'sum')
).reset_index()
print(monthly_summary_new_c)

monthly_summary_new_c['avg_trans_per_client'] = monthly_summary_new_c['total_trans']/monthly_summary_new_c['total_new_cust']
monthly_summary_new_c['avg_invoice_per_client'] = monthly_summary_new_c['total_invoice']/monthly_summary_new_c['total_new_cust']

cust_amount.info()


# 2.4 How the discounts playing role in the 'Invoice'?

discount_affect = pd.pivot_table(cust_amount, index= 'Transaction_Date_Month', columns= 'Coupon_Status', values= 'Is_New_Customer', fill_value= 0)

invoice_by_coupon_status = cust_amount.groupby('Coupon_Status')['Invoice'].sum().reset_index()


order_comparison = cust_amount.groupby(['Transaction_Date_Month', 'Coupon_Status'])['Transaction_ID'].count().unstack(fill_value=0).reset_index()
order_comparison = order_comparison[['Transaction_Date_Month', 'Not Used', 'Used']]
order_comparison.columns = ['T_D_M', 'Order_Without_Coupon', 'Order_With_Coupon']
print(order_comparison.columns)

# For this task also great suit an Anova Analysis

mean_invoice = temp_merg_data.groupby('Coupon_Status')['Invoice'].mean().reset_index()
print("Mean Invoice Amounts:")
print(mean_invoice)

temp_merg_data['Transaction_Date_Month'] = temp_merg_data['Transaction_Date'].dt.to_period('M')

anova_result = stats.f_oneway(
    temp_merg_data[temp_merg_data['Coupon_Status'] == "Used"]['Invoice'],
    temp_merg_data[temp_merg_data['Coupon_Status'] == "Not Used"]['Invoice'],
    temp_merg_data[temp_merg_data['Coupon_Status'] == "Clicked"]['Invoice']
)

print(f"F-statistic: {anova_result.statistic:.2f}, P-value: {anova_result.pvalue:.4f}")

if anova_result.pvalue < 0.05:
    print("There is a statistically significant difference in invoice amounts among the different discount statuses.")
else:
    print("There is no statistically significant difference in invoice amounts among the different discount statuses.")
# as a result, statistically significant difference is existing
# But if we would like to expand our approach to this question and let's check for each month ANOVA analysis
invoice_discounts = temp_merg_data[['Invoice', 'Discount_pct_active', 'Transaction_Date_Month']].copy()
invoice_discounts.groupby(['Transaction_Date_Month', 'Discount_pct_active'])['Invoice'].describe()

invoice_discounts['Discount'] = invoice_discounts['Discount_pct_active'].apply(lambda x: 'With Discount' if x > 0 else 'Without Discount')

def perform_test(month, df):
    discount_applied = df[(df['Transaction_Date_Month'] == month) & (df['Discount_pct_active'] > 0)]['Invoice']
    no_discount = df[(df['Transaction_Date_Month'] == month) & (df['Discount_pct_active'] == 0)]['Invoice']

    if len(discount_applied) > 0 and len(no_discount) > 0:
        # Normality test
        _, p_value_discount = stats.shapiro(discount_applied)
        _, p_value_no_discount = stats.shapiro(no_discount)

        if p_value_discount > 0.05 and p_value_no_discount > 0.05:
            # ANOVA
            f_stat, p_value = stats.f_oneway(discount_applied, no_discount)
            test_used = 'ANOVA'
        else:
            # Mann-Whitney U Test
            u_stat, p_value = stats.mannwhitneyu(discount_applied, no_discount)
            f_stat = u_stat # Assign u_stat to f_stat to ensure a value is returned
            test_used = 'Matt-Whitney'

        return test_used, f_stat, p_value

    else:
        return None, None, None

anova_results = []
for month in invoice_discounts['Transaction_Date_Month'].unique():
    test_used, f_stat, p_value = perform_test(month, invoice_discounts)
    if test_used is not None and f_stat is not None and p_value is not None:
        anova_results.append({'Month': month, 'Test Used': test_used, 'F-statistic': f_stat, 'P-value': p_value})

anova_results_df = pd.DataFrame(anova_results)
print(anova_results_df)


# 2.5. Analysis of seasonality/trends by category and location
plt.figure(figsize=(10,6))
plt.plot(temp_merg_data['Transaction_Date'],temp_merg_data['Invoice'], linestyle = '-', color = 'r', label = 'Seasonality')
plt.xlabel('Month')
plt.ylabel('Invoice')

plt.legend()
plt.show()

temp_merg_data = pd.merge(temp_merg_data, customer_data_df, how='left', on= 'CustomerID')

pivot_by_category = pd.pivot_table(temp_merg_data, index= 'Transaction_Date_Month', columns= 'Product_Category', values= 'Invoice', fill_value= 0, aggfunc = 'sum')
pivot_by_location = pd.pivot_table(temp_merg_data, index= 'Transaction_Date_Month', columns= 'Location', values= 'Invoice', fill_value= 0, aggfunc = 'sum')

def format_currency(value):
    return "${:,.2f}".format(value)


formatted_pivot_by_category = pivot_by_category.applymap(format_currency)
formatted_pivot_by_location = pivot_by_location.applymap(format_currency)

print("Pivot Table by Product Category:")
print(formatted_pivot_by_category)

print("\nPivot Table by Location:")
print(formatted_pivot_by_location)


# 2.6. How number order varies and sales with different days  of week?
temp_merg_data['Day_of_Week'] = temp_merg_data['Transaction_Date'].dt.day_name()


day_by_week_df = temp_merg_data.groupby('Day_of_Week').agg(
    Invoice = ('Invoice', 'sum'),
    Transactions = ('Transaction_ID', 'nunique')
).reset_index().sort_values('Day_of_Week').copy()

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

day_by_week_df['Day_of_Week'] = pd.Categorical(day_by_week_df['Day_of_Week'], categories = day_order, ordered = True)
day_by_week_df = day_by_week_df.sort_values('Day_of_Week')
print(day_by_week_df)


# 2.7. Which products were appeared in the transactions the more often by seasons (summer, autumn, winter, spring)?
# Which products were appeared in the transactions the more often by seasons(summer, autumn, winter, spring)?

def assign_season(month):
    if month in ['Dec', 'Jan', 'Feb']:
        return 'Winter'
    elif month in ['Mar', 'Apr', 'May']:
        return 'Spring'
    elif month in ['Jun', 'Jul', 'Aug']:
        return 'Summer'
    else:
        return 'Autumn'

temp_merg_data['Season'] = temp_merg_data['Month'].apply(assign_season)


product_appearance = temp_merg_data.groupby(['Season', 'Product_Description']).size().reset_index(name='Appearances')
product_appearance['Rank'] = product_appearance.groupby('Season')['Appearances'].rank(method='first', ascending=False)
top_products_by_appearance = product_appearance[product_appearance['Rank'] <= 5].sort_values(by=['Season', 'Rank'])
pivot_appearance = top_products_by_appearance.pivot(index='Rank', columns='Season', values='Product_Description')
print("Top 5 products ranked by the number of transactions for each season:")
print(pivot_appearance)

product_quantity = temp_merg_data.groupby(['Season', 'Product_Description'])['Quantity'].sum().reset_index(name='Total_Quantity')
product_quantity['Rank'] = product_quantity.groupby('Season')['Total_Quantity'].rank(method='first', ascending=False)
top_products_by_quantity = product_quantity[product_quantity['Rank'] <= 5].sort_values(by=['Season', 'Rank'])
pivot_quantity = top_products_by_quantity.pivot(index='Rank', columns='Season', values='Product_Description')
print("\nTop 5 products ranked by quantity ordered for each season:")
print(pivot_quantity)



# 2.8. How marketing spend is impacting on Invoice?
# The first method it could be simple linear regression
total_invoice_by_days = temp_merg_data.groupby('Transaction_Date')['Invoice'].sum().reset_index(name='Total_Invoice')

marketing_spend_df['Marketing_Spend_Total'] = marketing_spend_df['Offline_Spend'] + marketing_spend_df['Online_Spend']
merged_marketing_invoice = pd.merge(
    total_invoice_by_days,
    marketing_spend_df,
    left_on='Transaction_Date',   # Column name from total_invoice_by_days
    right_on='Date', # Column name from marketing_spend_df
)

X = merged_marketing_invoice['Marketing_Spend_Total']
Y = merged_marketing_invoice['Total_Invoice']

X = sm.add_constant(X)
model = sm.OLS(Y,X).fit()
print(model.summary())
#  For every one unit increase in Marketing_Spend_Total, the dependent variable is expected to increase by approximately 2.8946 units. In other words: Increased marketing spend is strongly associated with increased Invoice.
#  Standard Error (std err) for Marketing_Spend_Total: 0.086. It means that the estimate is precise.
#  t-Statistic (t) for Marketing_Spend_Total: 33.561 (>2). It means that the variable is likely an important predictor of the Total_Invoice variable
#  Effect is statistical significant (p_value = 0.000...)


# The second method for measure impacting on revenue(Our Invoice) is ROI
# (Invoice - Marketing Costs)/ Marketing Costs * 100
merged_marketing_invoice['Month'] = merged_marketing_invoice['Transaction_Date'].dt.to_period('M')
merged_marketing_invoice['%ROI'] = (merged_marketing_invoice['Total_Invoice'] -
                                    merged_marketing_invoice['Marketing_Spend_Total'])/merged_marketing_invoice['Marketing_Spend_Total']*100

merged_marketing_invoice.groupby('Month')['%ROI'].mean()
merged_marketing_invoice.info()


# Third method is Calculate Pearson correlation coefficient
corr_coef, p_value = pearsonr(merged_marketing_invoice['Marketing_Spend_Total'], merged_marketing_invoice['Total_Invoice'])
print(f"Pearson Correlation Coefficient: {corr_coef:.2f}, P-value: {p_value:.4f}")
# t-Statistic (t) for Marketing_Spend_Total: 33.561

"""
# Fourth method lagged_correlation
def calculate_lagged_correlation(df, target_column, columns, max_lag):
    '''
    Calculate the correlation considering a time shift in the data.
    '''
    results = []
    for col in columns:
        for lag in range(1, max_lag + 1):
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
            correlation = df[[target_column, f'{col}_lag{lag}']].corr().iloc[0, 1]
            results.append({'Lag': lag, 'Variable': col, 'Correlation': correlation})
    return pd.DataFrame(results)

max_lag = 20  ##  a lag of 20 means comparing today's 'Total_Invoice' with 20 back days value pf marketing spend
correlations = calculate_lagged_correlation(merged_marketing_invoice, 'Total_Invoice',['Online_Spend', 'Offline_Spend', 'Marketing_Spend_Total'], max_lag)

# Pivot
pivoted_correlation_df = correlations.pivot(index='Lag', columns='Variable', values='Correlation')
print(pivoted_correlation_df)

pivoted_correlation_df.plot(marker='o', figsize=(12, 6))
plt.title('Lagged Correlation between Marketing Spend and Invoice Totals')
plt.xlabel('Lag (Days)')
plt.ylabel('Correlation')
plt.show()
"""














"""
 3. Performing Customer Segmentation
RFM analysis (Recency, Frequency, and Monetary analysis).
Divide the customers into Premium, Gold, Silver,
Standard customers and define strategy on the same.





 Scientific (Using K-Means) & Understand the profiles. Define strategy for each
segment.
"""




# COHORT ANALYSIS
temp_merg_data.info()

cust_amount['CohortMonth'] = cust_amount.groupby('CustomerID')['Transaction_Date'].transform('min').dt.to_period('M')
cohort_counts = cust_amount.groupby(['CohortMonth', 'Transaction_Date_Month'])['CustomerID'].nunique().reset_index()
cohort_counts = cohort_counts.pivot_table(index='CohortMonth', columns='Transaction_Date_Month', values='CustomerID', fill_value=0)

cust_amount['CohortMonth'] = cust_amount.groupby('CustomerID')['Transaction_Date'].transform('min').dt.to_period('M')

cohort_sizes = cohort_counts.iloc[:, 0]
retention = cohort_counts.divide(cohort_sizes, axis=0) * 100

print("Cohort Counts:")
print(cohort_counts)
print("\nRetention Rates (%):")
print(retention)









# Problems I met
# Accounting in array after creating common table appear Nan
# That coupon also need to activate
# Merging by 2 column

During the course I learned