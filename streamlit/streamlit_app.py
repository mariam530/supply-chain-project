import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Supply Chain Analysis", layout="wide")

st.title("DataCo Supply Chain Data Analysis and Cleaning")

# Load the dataset directly from the same folder
try:
    df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding='latin1')
    st.success("Dataset loaded successfully!")
    st.write("Initial structure placeholder")
    #  1. Data Understanding 
    st.header("1. Data Understanding")
    st.write("Preview of the dataset:")
    st.dataframe(df.head())

    st.write(f"Shape of the dataset: Rows = {df.shape[0]}, Columns = {df.shape[1]}")

    with st.expander(" Data Info"):
        st.write("Data Types:")
        st.write(df.dtypes)
        st.write("Unique Values per Column:")
        st.write(df.nunique())
        st.write("Missing Values:")
        st.write(df.isnull().sum())
        st.write("Missing Value Percentage:")
        st.write(df.isnull().mean() * 100)
        st.write(f"Duplicated Rows: {df.duplicated().sum()}")

    with st.expander("Descriptive Statistics"):
        st.subheader("Numerical Columns")
        st.dataframe(df.describe(include='number'))
        st.subheader("Categorical Columns")
        st.dataframe(df.describe(include='object'))

    #  2. Data Cleaning 
    st.header("Data Cleaning")
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    df.drop_duplicates(inplace=True)
    df.fillna(0, inplace=True)
    st.write(" Cleaned data preview:")
    # 2.1 Remove Irrelevant Columns
    columns_to_drop = [
        'customer_id',
        'customer_email',
        'customer_password',
        'customer_street',
        'customer_zipcode',
        'order_id',
        'order_customer_id',
        'order_item_id',
        'order_item_cardprod_id',
        'product_card_id',
        'product_category_id',
        'product_image',
        'product_description'
    ]
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df.drop(columns=columns_to_drop, inplace=True)
    st.success("Irrelevant columns dropped.")

    st.dataframe(df.head())
    
     
# 2.2 DOMAIN KNOWLEDGE FEATURES

# Profitability_Flag
    df['Is_Profitable_Order'] = (df['order_profit_per_order'] > 0).astype(int)

# Zero_Profit_Flag
    df['Is_Zero_Profit'] = (df['order_profit_per_order'] == 0).astype(int)

# Profit_Ratio
    df['Order_Item_Profit_Ratio'] = df['order_profit_per_order'] / df['order_item_total']

# Profit_Margin_Copy
    df['Profit_Margin'] = df['Order_Item_Profit_Ratio']

# Profit_Category_Binning
    df['Profitability_Category'] = pd.cut(df['Profit_Margin'], bins=[-1, 0, 0.2, 0.5, 1], labels=['Loss', 'Low', 'Medium', 'High'])

# Low_Profit_High_Sales_Flag
    df['Low_Profit_High_Sales'] = ((df['Profit_Margin'] < 0.1) & (df['order_item_total'] > 500)).astype(int)

# Order_Value_Binning
    df['Order_Value_Category'] = pd.cut(df['order_item_total'], bins=[0, 100, 500, 1000, float('inf')], labels=['Low', 'Medium', 'High', 'Very High'])

# Customer_Segment_Binning
    df['Customer_Segment'] = pd.cut(df['sales_per_customer'], bins=[0, 100, 500, 1000, np.inf], labels=['Low', 'Medium', 'High', 'Very High'])

# Order_Quarter_Extraction
    df['Order_Quarter'] = pd.to_datetime(df['order_date']).dt.to_period('Q')

# Custom_Profit_Level
    def classify_profit(ratio):
    if ratio < 0.2:
        return 'low'
    elif ratio <= 0.5:
        return 'medium'
    else:
        return 'high'
    df['Profit_Category'] = df['Order_Item_Profit_Ratio'].apply(classify_profit)

# Order_Type_Classification
    df['Order_Type'] = np.where((df['order_item_product_price'] > 1000) & (df['order_item_discount'] < 100), 'premium', 'regular')

    #  3. Profit Classification 
    st.header(" Profit Category Classification")

    def classify_profit(ratio):
        if ratio <= 0.2:
            return 'low'
        elif ratio <= 0.5:
            return 'medium'
        else:
            return 'high'

    df['profit_category'] = df['order_item_profit_ratio'].apply(classify_profit)
    st.write(df[['order_item_profit_ratio', 'profit_category']].head())

    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='profit_category', ax=ax1)
    st.pyplot(fig1)

    #  4. Order Type Classification 
    st.header("Order Type Classification")

    df['order_type'] = np.where(
        (df['order_item_product_price'] > 1000) & (df['order_item_discount'] < 100),
        'premium',
        'regular'
    )
    st.write(df[['order_item_product_price', 'order_item_discount', 'order_type']].head())

    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x='order_type', ax=ax2)
    st.pyplot(fig2)

    #  5. Visual Exploration 
    st.header(" Visual Exploration")

    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    if num_cols:
        selected_num = st.selectbox("Choose a numerical column", num_cols)
        fig3, ax3 = plt.subplots()
        sns.histplot(df[selected_num], kde=True, ax=ax3)
        st.pyplot(fig3)

    if cat_cols:
        selected_cat = st.selectbox("Choose a categorical column", cat_cols)
        st.bar_chart(df[selected_cat].value_counts())

    #  6. Profit by Region 
    st.header(" Profit Distribution by Region")
    if 'order_region' in df.columns:
        region_profit = df.groupby('order_region')['order_item_profit_ratio'].mean().sort_values()
        st.bar_chart(region_profit)

    #  7. Customer Segmentation 
    st.header("Customer Segmentation by Spending")
    if 'sales_per_customer' in df.columns:
        df['customer_segment'] = pd.cut(
            df['sales_per_customer'],
            bins=[0, 100, 500, 1000, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        st.write(df[['sales_per_customer', 'customer_segment']].head())
        fig4, ax4 = plt.subplots()
        sns.countplot(data=df, x='customer_segment', ax=ax4)
        st.pyplot(fig4)

    #  8. Late Delivery Risk Analysis 
    st.header("🚚 Late Delivery Risk Analysis")
    if 'late_delivery_risk' in df.columns:
        risk_by_type = df.groupby('order_type')['late_delivery_risk'].mean()
        st.bar_chart(risk_by_type)


    # 9. Profitability Category Pie Chart 
   
    #  Order Profit Per Order - Pie Chart
    st.header("Order Profit Per Order - Pie Chart")

    df['order_profit_category'] = pd.cut(df['order_profit_per_order'],
                                         bins=[-float('inf'), 0, 200, 500, float('inf')],
                                         labels=['Loss', 'Low', 'Medium', 'High'])

    profit_counts = df['order_profit_category'].value_counts().reset_index()
    profit_counts.columns = ['order_profit_category', 'count']

    fig_pie_profit = px.pie(profit_counts,
                            names='order_profit_category',
                            values='count',
                            title='Order Profit Per Order Distribution by Category')
    st.plotly_chart(fig_pie_profit)


#  10. Correlation Heatmap 
   
    # 🔍 Correlation Heatmap (After Column Removal)
    st.header("Correlation Heatmap")
    corr = df.select_dtypes(include='number').corr()
    fig_corr, ax_corr = plt.subplots(figsize=(16, 10))  
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)


    # 11. Univariate Analysis
    st.header("Univariate Analysis")
    selected_uni = st.selectbox("Choose a column for univariate analysis", df.columns)

    st.write("Summary Statistics:")
    st.write(df[selected_uni].describe())

    if df[selected_uni].dtype in ['int64', 'float64']:
        fig_uni = px.histogram(df, x=selected_uni, nbins=30, marginal='box', title=f'Distribution of {selected_uni}')
    else:
        value_counts_df = df[selected_uni].value_counts().reset_index()
        value_counts_df.columns = [selected_uni, 'count']
        fig_uni = px.bar(
            value_counts_df,
            x=selected_uni,
            y='count',
            labels={selected_uni: selected_uni, 'count': 'Count'},
            title=f'Value Counts of {selected_uni}'
        )
    st.plotly_chart(fig_uni)

    # 12. Bivariate Analysis
    st.header("Bivariate Analysis")

    col1 = st.selectbox("Select first column", df.columns, key="bi_col1")
    col2 = st.selectbox("Select second column", df.columns, key="bi_col2")

    if df[col1].dtype in ['int64', 'float64'] and df[col2].dtype in ['int64', 'float64']:
        fig_bi = px.scatter(df, x=col1, y=col2, trendline="ols", title=f'{col1} vs {col2}')
    elif df[col1].dtype == 'object' and df[col2].dtype in ['int64', 'float64']:
        fig_bi = px.box(df, x=col1, y=col2, title=f'{col2} by {col1}')
    elif df[col2].dtype == 'object' and df[col1].dtype in ['int64', 'float64']:
        fig_bi = px.box(df, x=col2, y=col1, title=f'{col1} by {col2}')
    else:
        fig_bi = None
        st.warning("Cannot plot selected combination.")

    if fig_bi:
        st.plotly_chart(fig_bi)

    # 13. Multivariate Analysis
    st.header("Multivariate Analysis (Scatter Matrix with Plotly)")

    selected_multi = st.multiselect("Choose up to 5 numeric columns", num_cols, max_selections=5)

    if len(selected_multi) >= 2:
        fig_mv = px.scatter_matrix(df, dimensions=selected_multi, title='Scatter Matrix of Selected Variables')
        st.plotly_chart(fig_mv)
    else:
        st.info("Please select at least 2 numeric columns.")


    
    # Sorted Statistics
    st.header("Sorted Statistics for Numeric Columns")

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    selected_sort_col = st.selectbox("Select column to sort", numeric_cols, key="sort_col")
    sort_order = st.radio("Select sort order", ["Ascending", "Descending"], key="sort_order")

    sorted_df = df.sort_values(by=selected_sort_col, ascending=(sort_order == "Ascending"))
    st.write(f"Top 10 rows sorted by {selected_sort_col} ({sort_order}):")
    st.dataframe(sorted_df[[selected_sort_col]].head(10))



except FileNotFoundError:
    st.error("Dataset file not found. Please make sure 'DataCoSupplyChainDataset.csv' is in the same folder.")


