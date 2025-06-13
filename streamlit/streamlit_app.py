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
    st.dataframe(df.head())

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

    #  9. Correlation Heatmap 
    st.header(" Correlation Heatmap")
    corr = df.select_dtypes(include='number').corr()
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax5)
    st.pyplot(fig5)

    # 10. Univariate Analysis
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

    # 11. Bivariate Analysis
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

    # 12. Multivariate Analysis
    st.header("Multivariate Analysis (Scatter Matrix with Plotly)")

    selected_multi = st.multiselect("Choose up to 5 numeric columns", num_cols, max_selections=5)

    if len(selected_multi) >= 2:
        fig_mv = px.scatter_matrix(df, dimensions=selected_multi, title='Scatter Matrix of Selected Variables')
        st.plotly_chart(fig_mv)
    else:
        st.info("Please select at least 2 numeric columns.")

except FileNotFoundError:
    st.error("Dataset file not found. Please make sure 'DataCoSupplyChainDataset.csv' is in the same folder.")


