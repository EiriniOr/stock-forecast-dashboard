"""
Stock Forecasting Dashboard
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")

# --- DATA LOADING ---
def find_excel_file():
    """Auto-detect xlsx file in same folder as script"""
    script_dir = Path(__file__).parent
    xlsx_files = list(script_dir.glob("*.xlsx"))
    # Filter out temp files (starting with ~$)
    xlsx_files = [f for f in xlsx_files if not f.name.startswith("~$")]
    if xlsx_files:
        # Return most recently modified
        return str(max(xlsx_files, key=lambda x: x.stat().st_mtime))
    return None

@st.cache_data
def load_data(file_path):
    """Parse the Excel file and extract structured data"""

    # Read cases/sales data
    df_raw = pd.read_excel(file_path, sheet_name='Cases2021 2022 FC', header=None)

    # Extract years from row 8 and months from row 9
    years_row = df_raw.iloc[8, 3:27].values
    months_row = df_raw.iloc[9, 3:27].values

    # Build date columns
    date_cols = []
    for y, m in zip(years_row, months_row):
        if pd.notna(y) and pd.notna(m):
            try:
                date_cols.append(f"{int(y)}-{int(m):02d}")
            except:
                date_cols.append(None)
        else:
            date_cols.append(None)

    # Get data starting from row 10
    data_df = df_raw.iloc[10:].copy()

    # Create column names
    col_names = ['Category', 'SKU_Code', 'Product_Name']
    for i, dc in enumerate(date_cols):
        if dc:
            col_names.append(dc)
        else:
            col_names.append(f'extra_{i}')

    # Add remaining columns
    remaining = len(data_df.columns) - len(col_names)
    for i in range(remaining):
        col_names.append(f'meta_{i}')

    data_df.columns = col_names[:len(data_df.columns)]

    # Filter valid rows
    data_df = data_df[data_df['SKU_Code'].notna()].reset_index(drop=True)

    # Clean SKU codes
    data_df['SKU_Code'] = data_df['SKU_Code'].astype(str).str.strip()
    data_df['SKU_Short'] = data_df['SKU_Code'].str[-8:]

    # Get date columns only
    date_columns = [c for c in data_df.columns if '-' in str(c) and str(c)[:4].isdigit()]

    # Read stock data
    df_stock = pd.read_excel(file_path, sheet_name='Stock')
    df_stock['Material Code'] = df_stock['Material Code'].astype(str).str.strip()

    return data_df, df_stock, date_columns

# --- FORECASTING ---
def forecast_demand(historical_data, periods_ahead=12):
    """Simple forecasting using trend + seasonality"""

    data = pd.Series(historical_data).dropna()
    if len(data) < 6:
        return None, None, None

    # Prepare features
    X = np.arange(len(data)).reshape(-1, 1)
    y = data.values

    # Fit linear trend
    model = LinearRegression()
    model.fit(X, y)

    # Predict future
    future_X = np.arange(len(data), len(data) + periods_ahead).reshape(-1, 1)
    forecast = model.predict(future_X)

    # Calculate metrics
    fitted = model.predict(X)
    residuals = y - fitted
    rmse = np.sqrt(np.mean(residuals**2))

    # Seasonal adjustment (simple monthly pattern if > 12 months)
    if len(data) >= 12:
        monthly_avg = []
        for m in range(12):
            month_vals = [data.iloc[i] for i in range(m, len(data), 12) if i < len(data)]
            if month_vals:
                monthly_avg.append(np.mean(month_vals))
            else:
                monthly_avg.append(np.mean(data))

        overall_avg = np.mean(data)
        seasonal_factors = [m / overall_avg if overall_avg > 0 else 1 for m in monthly_avg]

        # Apply seasonality to forecast
        start_month = len(data) % 12
        for i in range(len(forecast)):
            month_idx = (start_month + i) % 12
            forecast[i] *= seasonal_factors[month_idx]

    # Ensure non-negative
    forecast = np.maximum(forecast, 0)

    return forecast, rmse, model

def calculate_safety_stock(historical_data, service_level=0.95):
    """Calculate safety stock based on demand variability"""
    data = pd.Series(historical_data).dropna()
    if len(data) < 3:
        return 0

    # Z-score for service level
    z_scores = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
    z = z_scores.get(service_level, 1.65)

    std_dev = data.std()
    lead_time = 1  # months

    safety_stock = z * std_dev * np.sqrt(lead_time)
    return max(0, safety_stock)

# --- MAIN APP ---
def main():
    st.title("Stock Forecasting Dashboard")

    # Auto-detect xlsx file in same folder
    auto_file = find_excel_file()

    # Sidebar
    st.sidebar.header("Settings")

    if auto_file:
        st.sidebar.success(f"Using: {Path(auto_file).name}")
        file_path = auto_file
    else:
        st.error("No .xlsx file found in the app folder!")
        st.info("Place your Excel file in the same folder as this app")
        return

    # Button to clear cache and reload (for updated data)
    if st.sidebar.button("Reload Data"):
        st.cache_data.clear()
        st.rerun()

    # Load data
    try:
        data_df, stock_df, date_columns = load_data(file_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    st.sidebar.success(f"Loaded {len(data_df)} products")

    # Product selection
    st.sidebar.header("Product Selection")

    # Category filter
    categories = ['All'] + sorted(data_df['Category'].dropna().unique().tolist())
    selected_category = st.sidebar.selectbox("Category", categories)

    if selected_category != 'All':
        filtered_df = data_df[data_df['Category'] == selected_category]
    else:
        filtered_df = data_df

    # Product selector
    product_options = filtered_df.apply(
        lambda x: f"{x['SKU_Short']} - {str(x['Product_Name'])[:40]}", axis=1
    ).tolist()

    if not product_options:
        st.warning("No products in selected category")
        return

    selected_product = st.sidebar.selectbox("Select Product", product_options)

    # Get selected product data
    selected_idx = product_options.index(selected_product)
    product_row = filtered_df.iloc[selected_idx]

    # Forecast settings
    st.sidebar.header("Forecast Settings")
    forecast_months = st.sidebar.slider("Forecast Horizon (months)", 1, 24, 12)
    service_level = st.sidebar.selectbox("Service Level", [0.90, 0.95, 0.99], index=1)

    # --- MAIN CONTENT ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("SKU Code", product_row['SKU_Short'])
    with col2:
        st.metric("Category", product_row['Category'])
    with col3:
        # Check current stock
        sku_short = product_row['SKU_Short']
        stock_match = stock_df[stock_df['Material Code'].str.contains(sku_short, na=False)]
        if not stock_match.empty:
            current_stock = stock_match.iloc[0]['Unrestricted Stock']
            st.metric("Current Stock", f"{int(current_stock):,}")
        else:
            st.metric("Current Stock", "N/A")

    st.subheader(product_row['Product_Name'])

    # Extract historical data
    historical = product_row[date_columns].values.astype(float)
    dates = pd.to_datetime([f"{d}-01" for d in date_columns])

    # Create historical dataframe
    hist_df = pd.DataFrame({
        'Date': dates,
        'Demand': historical
    })
    hist_df = hist_df.dropna()

    # --- HISTORICAL ANALYSIS TAB ---
    tab1, tab2, tab3, tab4 = st.tabs(["Historical Analysis", "Forecast", "Stock Analysis", "Comparison"])

    with tab1:
        st.subheader("Historical Demand")

        if len(hist_df) > 0:
            fig = px.line(hist_df, x='Date', y='Demand',
                         title='Monthly Demand Over Time',
                         markers=True)
            fig.update_layout(xaxis_title="Month", yaxis_title="Cases")
            st.plotly_chart(fig, width="stretch")

            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Monthly Demand", f"{hist_df['Demand'].mean():,.0f}")
            with col2:
                st.metric("Max Demand", f"{hist_df['Demand'].max():,.0f}")
            with col3:
                st.metric("Min Demand", f"{hist_df['Demand'].min():,.0f}")
            with col4:
                st.metric("Std Deviation", f"{hist_df['Demand'].std():,.0f}")

            # Year-over-year comparison
            st.subheader("Year-over-Year Comparison")
            hist_df['Year'] = hist_df['Date'].dt.year
            hist_df['Month'] = hist_df['Date'].dt.month

            pivot = hist_df.pivot_table(values='Demand', index='Month', columns='Year', aggfunc='sum')

            fig_yoy = go.Figure()
            for year in pivot.columns:
                fig_yoy.add_trace(go.Scatter(
                    x=pivot.index,
                    y=pivot[year],
                    name=str(year),
                    mode='lines+markers'
                ))
            fig_yoy.update_layout(
                title='Demand by Month (Year Comparison)',
                xaxis_title='Month',
                yaxis_title='Cases',
                xaxis=dict(tickmode='array', tickvals=list(range(1,13)),
                          ticktext=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
            )
            st.plotly_chart(fig_yoy, width="stretch")

            # Monthly summary table
            st.subheader("Monthly Summary")
            st.dataframe(pivot.round(0), width="stretch")
        else:
            st.warning("No historical data available")

    with tab2:
        st.subheader("Demand Forecast")

        if len(hist_df) >= 6:
            forecast, rmse, model = forecast_demand(hist_df['Demand'].values, forecast_months)

            if forecast is not None:
                # Create forecast dates
                last_date = hist_df['Date'].max()
                forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                              periods=forecast_months, freq='MS')

                forecast_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Forecast': forecast
                })

                # Combined chart
                fig = go.Figure()

                # Historical
                fig.add_trace(go.Scatter(
                    x=hist_df['Date'], y=hist_df['Demand'],
                    name='Historical', mode='lines+markers',
                    line=dict(color='blue')
                ))

                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'], y=forecast_df['Forecast'],
                    name='Forecast', mode='lines+markers',
                    line=dict(color='red', dash='dash')
                ))

                # Confidence interval (simple)
                upper = forecast_df['Forecast'] + 1.96 * rmse
                lower = np.maximum(forecast_df['Forecast'] - 1.96 * rmse, 0)

                fig.add_trace(go.Scatter(
                    x=list(forecast_df['Date']) + list(forecast_df['Date'][::-1]),
                    y=list(upper) + list(lower[::-1]),
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.1)',
                    line=dict(color='rgba(255,0,0,0)'),
                    name='95% CI'
                ))

                fig.update_layout(
                    title=f'{forecast_months}-Month Demand Forecast',
                    xaxis_title='Date',
                    yaxis_title='Cases'
                )
                st.plotly_chart(fig, width="stretch")

                # Forecast metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Next Month Forecast", f"{forecast[0]:,.0f}")
                with col2:
                    st.metric(f"Total {forecast_months}m Forecast", f"{forecast.sum():,.0f}")
                with col3:
                    st.metric("Forecast RMSE", f"{rmse:,.0f}")
                with col4:
                    safety = calculate_safety_stock(hist_df['Demand'].values, service_level)
                    st.metric("Safety Stock", f"{safety:,.0f}")

                # Forecast table
                st.subheader("Forecast Details")
                forecast_df['Month'] = forecast_df['Date'].dt.strftime('%Y-%m')
                forecast_df['Forecast'] = forecast_df['Forecast'].round(0).astype(int)
                st.dataframe(forecast_df[['Month', 'Forecast']], width="stretch")
            else:
                st.warning("Could not generate forecast")
        else:
            st.warning("Need at least 6 months of data for forecasting")

    with tab3:
        st.subheader("Stock Position Analysis")

        if not stock_match.empty:
            stock_row = stock_match.iloc[0]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Current Stock Levels")
                st.metric("Unrestricted Stock", f"{int(stock_row['Unrestricted Stock']):,}")
                st.metric("Available Quantity", f"{int(stock_row['Available Quantity']):,}")
                st.metric("Quality Inspection", f"{int(stock_row['Quality Inspection Stock']):,}")

            with col2:
                st.markdown("### Orders & Deliveries")
                st.metric("Customer Orders", f"{int(stock_row['Cust.Ord.Total Qty']):,}")
                st.metric("Confirmed Orders", f"{int(stock_row['Cust.Conf.Total Qty']):,}")
                st.metric("On Deliveries", f"{int(stock_row['Quantities on Deliveries']):,}")
                st.metric("Goods Receipts", f"{int(stock_row['Goods Receipts Total Qty']):,}")

            # Stock vs Demand analysis
            if len(hist_df) > 0:
                avg_monthly = hist_df['Demand'].mean()
                current = stock_row['Unrestricted Stock']

                if avg_monthly > 0:
                    months_of_stock = current / avg_monthly

                    st.markdown("### Stock Coverage")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Monthly Demand", f"{avg_monthly:,.0f}")
                    with col2:
                        st.metric("Months of Stock", f"{months_of_stock:.1f}")
                    with col3:
                        safety = calculate_safety_stock(hist_df['Demand'].values, service_level)
                        reorder_point = avg_monthly + safety
                        st.metric("Reorder Point", f"{reorder_point:,.0f}")

                    # Visual gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=months_of_stock,
                        title={'text': "Months of Stock Coverage"},
                        gauge={
                            'axis': {'range': [0, 6]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 1], 'color': "red"},
                                {'range': [1, 2], 'color': "orange"},
                                {'range': [2, 4], 'color': "lightgreen"},
                                {'range': [4, 6], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': 2
                            }
                        }
                    ))
                    st.plotly_chart(fig, width="stretch")
        else:
            st.info("No current stock data found for this product")

    with tab4:
        st.subheader("Product Comparison")

        # Multi-select for comparison
        compare_products = st.multiselect(
            "Select products to compare",
            product_options,
            default=[selected_product]
        )

        if compare_products:
            fig = go.Figure()

            for prod in compare_products:
                idx = product_options.index(prod)
                row = filtered_df.iloc[idx]
                hist = row[date_columns].values.astype(float)

                fig.add_trace(go.Scatter(
                    x=dates,
                    y=hist,
                    name=prod[:30],
                    mode='lines+markers'
                ))

            fig.update_layout(
                title='Product Demand Comparison',
                xaxis_title='Date',
                yaxis_title='Cases'
            )
            st.plotly_chart(fig, width="stretch")

    # --- SUMMARY TABLE ---
    st.header("All Products Summary")

    # Build summary
    summary_data = []
    for _, row in data_df.iterrows():
        hist = pd.Series(row[date_columns].values.astype(float)).dropna()

        # Match stock
        sku = row['SKU_Short']
        stock_match = stock_df[stock_df['Material Code'].str.contains(sku, na=False)]
        current_stock = stock_match.iloc[0]['Unrestricted Stock'] if not stock_match.empty else None

        # Calculate forecast
        if len(hist) >= 6:
            forecast, _, _ = forecast_demand(hist.values, 3)
            next_3m = forecast.sum() if forecast is not None else None
        else:
            next_3m = None

        avg_demand = hist.mean() if len(hist) > 0 else None
        latest = hist.iloc[-1] if len(hist) > 0 else None

        summary_data.append({
            'SKU': sku,
            'Product': str(row['Product_Name'])[:35],
            'Category': row['Category'],
            'Current Stock': current_stock,
            'Avg Monthly': avg_demand,
            'Latest Month': latest,
            'Next 3m Forecast': next_3m,
            'Months Coverage': current_stock / avg_demand if current_stock and avg_demand and avg_demand > 0 else None
        })

    summary_df = pd.DataFrame(summary_data)

    # Format and display
    st.dataframe(
        summary_df.style.format({
            'Current Stock': lambda x: f"{int(x):,}" if pd.notna(x) else "-",
            'Avg Monthly': lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
            'Latest Month': lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
            'Next 3m Forecast': lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
            'Months Coverage': lambda x: f"{x:.1f}" if pd.notna(x) else "-"
        }),
        width="stretch",
        height=400
    )

if __name__ == "__main__":
    main()
