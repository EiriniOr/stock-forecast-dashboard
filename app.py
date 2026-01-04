"""
Stock Forecasting Dashboard - Cloud Version
Πίνακας Πρόβλεψης Αποθέματος
Groups products by description (not code) for better forecasting
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Πρόβλεψη Αποθέματος", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_data(uploaded_file):
    """Load all relevant sheets from Excel and aggregate by product description"""

    # 1. Cases sheet - historical sales
    df_raw = pd.read_excel(uploaded_file, sheet_name='Cases2021 2022 FC', header=None)

    years_row = df_raw.iloc[8, 3:26].values
    months_row = df_raw.iloc[9, 3:26].values

    date_cols = []
    for y, m in zip(years_row, months_row):
        if pd.notna(y) and pd.notna(m):
            try:
                date_cols.append(f"{int(y)}-{int(m):02d}")
            except:
                date_cols.append(None)
        else:
            date_cols.append(None)

    data_df = df_raw.iloc[10:].copy()
    col_names = ['Category', 'SKU_Code', 'Product_Name']
    for i, dc in enumerate(date_cols):
        col_names.append(dc if dc else f'extra_{i}')

    remaining = len(data_df.columns) - len(col_names)
    for i in range(remaining):
        col_names.append(f'meta_{i}')

    data_df.columns = col_names[:len(data_df.columns)]
    data_df = data_df[data_df['SKU_Code'].notna()].reset_index(drop=True)
    data_df['SKU_Code'] = data_df['SKU_Code'].astype(str).str.strip()
    data_df['Material'] = data_df['SKU_Code'].str[-8:]

    date_columns = [c for c in data_df.columns if c and '-' in str(c) and str(c)[:4].isdigit()]

    # 2. S1P sheet - current stock (sum by Material)
    uploaded_file.seek(0)
    df_s1p = pd.read_excel(uploaded_file, sheet_name='S1P')
    stock_by_material = df_s1p.groupby('Material').agg({
        'Unrestricted stock': 'sum',
        'Description': 'first',
        'Category': 'first'
    }).reset_index()
    stock_by_material['Material'] = stock_by_material['Material'].astype(str)

    # 3. SOP sheet - existing system's OOS predictions
    uploaded_file.seek(0)
    df_sop = pd.read_excel(uploaded_file, sheet_name='SOP', header=2)
    df_sop.columns = ['Category_SOP', 'Trade', 'MRDR', 'Product_Desc', 'Link', 'Status',
                      'Avg_Daily_Sales', 'Stock_Days', 'Days_Till_Avail', 'Available_Stock'] + \
                     [f'extra_{i}' for i in range(len(df_sop.columns) - 10)]
    df_sop['MRDR'] = df_sop['MRDR'].astype(str).str.strip()

    # ================================================================
    # AGGREGATE BY PRODUCT DESCRIPTION
    # ================================================================
    # Group sales data by Product_Name (sum across all material codes)
    agg_dict = {col: 'sum' for col in date_columns}
    agg_dict['Category'] = 'first'
    agg_dict['Material'] = lambda x: list(x)  # Keep list of all material codes

    grouped_df = data_df.groupby('Product_Name').agg(agg_dict).reset_index()
    grouped_df['Material_Codes'] = grouped_df['Material']
    grouped_df['Material'] = grouped_df['Material_Codes'].apply(lambda x: x[0] if x else '')
    grouped_df['Num_Codes'] = grouped_df['Material_Codes'].apply(len)

    return grouped_df, stock_by_material, df_sop, date_columns, data_df


# --- TIME SERIES FORECASTING ---
def holt_winters_forecast(data, alpha=0.3, beta=0.1, gamma=0.2, season_length=12, forecast_periods=1):
    """
    Holt-Winters Triple Exponential Smoothing
    Captures: Level + Trend + Seasonality
    """
    n = len(data)
    if n < season_length + 2:
        return None, None

    # Initialize
    level = np.mean(data[:season_length])
    trend = (np.mean(data[season_length:2*season_length]) - np.mean(data[:season_length])) / season_length if n >= 2*season_length else 0

    # Seasonal factors
    seasonal = np.zeros(season_length)
    for i in range(season_length):
        seasonal[i] = data[i] / level if level > 0 else 1.0

    # Smooth through the data
    fitted = np.zeros(n)
    for t in range(n):
        if t == 0:
            fitted[t] = level + trend
        else:
            prev_level = level
            level = alpha * (data[t] / seasonal[t % season_length]) + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            seasonal[t % season_length] = gamma * (data[t] / level) + (1 - gamma) * seasonal[t % season_length]
            fitted[t] = (level + trend) * seasonal[t % season_length]

    # Forecast
    forecasts = []
    for h in range(1, forecast_periods + 1):
        forecast = (level + h * trend) * seasonal[(n - 1 + h) % season_length]
        forecasts.append(max(0, forecast))

    # Calculate error
    errors = data - fitted
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(np.abs(errors / np.where(data != 0, data, 1))) * 100

    return np.array(forecasts), {'rmse': rmse, 'mape': mape, 'fitted': fitted}


def forecast_demand(historical_monthly, current_month):
    """
    Forecast daily demand using time series + seasonality
    """
    data = pd.Series(historical_monthly).dropna()
    positive_data = data[data > 0]

    if len(positive_data) < 6:
        daily_avg = positive_data.mean() / 30 if len(positive_data) > 0 else 0
        return {
            'daily_avg': daily_avg,
            'demand_1d': daily_avg,
            'demand_7d': daily_avg * 7,
            'demand_14d': daily_avg * 14,
            'method': 'simple_average',
            'confidence': 'low'
        }

    # Try Holt-Winters
    hw_forecast, hw_metrics = holt_winters_forecast(
        positive_data.values,
        alpha=0.4, beta=0.1, gamma=0.3,
        season_length=min(12, len(positive_data) // 2),
        forecast_periods=2
    )

    if hw_forecast is not None:
        next_month_forecast = hw_forecast[0]
        daily_forecast = next_month_forecast / 30
        monthly_avg = positive_data.mean()
        seasonal_factor = next_month_forecast / monthly_avg if monthly_avg > 0 else 1.0

        return {
            'daily_avg': daily_forecast,
            'demand_1d': daily_forecast,
            'demand_7d': daily_forecast * 7,
            'demand_14d': daily_forecast * 14,
            'monthly_forecast': next_month_forecast,
            'seasonal_factor': seasonal_factor,
            'method': 'holt_winters',
            'confidence': 'high' if hw_metrics['mape'] < 30 else 'medium',
            'mape': hw_metrics['mape']
        }

    daily_avg = positive_data.mean() / 30
    return {
        'daily_avg': daily_avg,
        'demand_1d': daily_avg,
        'demand_7d': daily_avg * 7,
        'demand_14d': daily_avg * 14,
        'method': 'average',
        'confidence': 'medium'
    }


def calculate_safety_stock(historical_data, days=1, service_level=0.95):
    """Calculate safety stock for given number of days"""
    data = pd.Series(historical_data).dropna()
    positive_data = data[data > 0]
    if len(positive_data) < 3:
        return 0

    daily_std = positive_data.std() / 30
    z_scores = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
    z = z_scores.get(service_level, 1.65)
    return max(0, z * daily_std * np.sqrt(days))


def get_total_stock_for_product(product_row, stock_df):
    """Get total stock across all material codes for a product"""
    material_codes = product_row.get('Material_Codes', [product_row.get('Material', '')])
    if isinstance(material_codes, str):
        material_codes = [material_codes]

    total_stock = 0
    for mat in material_codes:
        mat_str = str(mat)
        stock_match = stock_df[stock_df['Material'] == mat_str]
        if not stock_match.empty:
            total_stock += int(stock_match['Unrestricted stock'].values[0])
    return total_stock


def get_system_prediction_for_product(product_row, sop_df):
    """Get system prediction across all material codes for a product"""
    material_codes = product_row.get('Material_Codes', [product_row.get('Material', '')])
    if isinstance(material_codes, str):
        material_codes = [material_codes]

    total_daily_sales = 0
    has_oos = False
    found_any = False

    for mat in material_codes:
        mat_str = str(mat)
        sop_match = sop_df[sop_df['MRDR'].str.contains(mat_str, na=False)]
        if not sop_match.empty:
            found_any = True
            daily = sop_match['Avg_Daily_Sales'].values[0]
            status = sop_match['Status'].values[0]
            if pd.notna(daily):
                total_daily_sales += daily
            if status == "OOS":
                has_oos = True

    if not found_any:
        return None, None, None

    return total_daily_sales, has_oos, found_any


# --- MAIN APP ---
def main():
    st.title("📦 Πρόβλεψη Αποθέματος & Παραγγελίες")
    st.caption("Ομαδοποίηση κατά περιγραφή προϊόντος (όχι κωδικό)")

    uploaded_file = st.file_uploader(
        "Ανέβασε το αρχείο Excel",
        type=['xlsx'],
        help="Αρχείο με sheets 'Cases2021 2022 FC', 'S1P', και 'SOP'"
    )

    if uploaded_file is None:
        st.info("👆 Ανέβασε το αρχείο Excel για να ξεκινήσεις")
        return

    try:
        grouped_df, stock_df, sop_df, date_columns, raw_df = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Σφάλμα: {e}")
        return

    # Initialize session state
    if 'selected_product' not in st.session_state:
        st.session_state.selected_product = None

    # --- PRODUCT FILTERING ---
    def is_discontinued(row):
        values = pd.Series(row[date_columns].values.astype(float)).dropna()
        if len(values) == 0:
            return True
        return (values <= 0).all()

    def is_old_product_12m(row):
        recent_cols = date_columns[-12:] if len(date_columns) >= 12 else date_columns
        values = pd.Series(row[recent_cols].values.astype(float)).dropna()
        if len(values) == 0:
            return True
        return (values <= 0).all()

    def is_old_product_6m(row):
        recent_cols = date_columns[-6:] if len(date_columns) >= 6 else date_columns
        values = pd.Series(row[recent_cols].values.astype(float)).dropna()
        if len(values) == 0:
            return True
        return (values <= 0).all()

    grouped_df['_discontinued'] = grouped_df.apply(is_discontinued, axis=1)
    grouped_df['_old_12m'] = grouped_df.apply(is_old_product_12m, axis=1)
    grouped_df['_old_6m'] = grouped_df.apply(is_old_product_6m, axis=1)

    # --- PRODUCT SELECTION ---
    st.markdown("---")

    # Filter checkboxes
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    with col_filter1:
        show_discontinued = st.checkbox("Διακοπτόμενα", value=False,
                                        help="Προϊόντα με μόνο αρνητικές τιμές")
    with col_filter2:
        show_old_6m = st.checkbox("Χωρίς πωλήσεις 6μ", value=False,
                                  help="Προϊόντα χωρίς πωλήσεις 6 μήνες")
    with col_filter3:
        show_old_12m = st.checkbox("Χωρίς πωλήσεις 12μ", value=False,
                                   help="Προϊόντα χωρίς πωλήσεις 12 μήνες")

    working_df = grouped_df.copy()
    if not show_discontinued:
        working_df = working_df[~working_df['_discontinued']]
    if not show_old_6m:
        working_df = working_df[~working_df['_old_6m']]
    if not show_old_12m:
        working_df = working_df[~working_df['_old_12m']]

    n_total = len(grouped_df)
    n_shown = len(working_df)
    st.caption(f"Εμφανίζονται {n_shown}/{n_total} προϊόντα (ομαδοποιημένα κατά περιγραφή)")

    col1, col2 = st.columns([2, 3])

    with col1:
        categories = ['Όλες οι κατηγορίες'] + sorted(working_df['Category'].dropna().unique().tolist())
        selected_category = st.selectbox("Κατηγορία", categories)

    if selected_category != 'Όλες οι κατηγορίες':
        filtered_df = working_df[working_df['Category'] == selected_category].reset_index(drop=True)
    else:
        filtered_df = working_df.reset_index(drop=True)

    # Product options now show description + number of codes
    product_options = filtered_df.apply(
        lambda x: f"{str(x['Product_Name'])[:50]} ({x['Num_Codes']} κωδ.)", axis=1
    ).tolist()

    default_idx = 0
    if st.session_state.selected_product:
        for i, opt in enumerate(product_options):
            if st.session_state.selected_product in opt:
                default_idx = i
                break

    with col2:
        selected_product = st.selectbox("Προϊόν", product_options, index=default_idx)

    if not product_options:
        st.warning("Δεν βρέθηκαν προϊόντα")
        return

    # Get selected product data
    selected_idx = product_options.index(selected_product)
    product_row = filtered_df.iloc[selected_idx]
    product_name = product_row['Product_Name']

    # Get TOTAL stock across all material codes
    current_stock = get_total_stock_for_product(product_row, stock_df)

    # Get system prediction (aggregated)
    sys_daily_sales, sys_has_oos, sys_found = get_system_prediction_for_product(product_row, sop_df)

    # Historical data (now aggregated)
    historical = product_row[date_columns].values.astype(float)
    dates = pd.to_datetime([f"{d}-01" for d in date_columns])
    hist_df = pd.DataFrame({'Date': dates, 'Demand': historical}).dropna()

    # --- FORECASTING ---
    current_month = datetime.now().month
    forecast_result = forecast_demand(hist_df['Demand'].values, current_month)

    # Safety stock
    safety_1d = calculate_safety_stock(hist_df['Demand'].values, 1)
    safety_7d = calculate_safety_stock(hist_df['Demand'].values, 7)
    safety_14d = calculate_safety_stock(hist_df['Demand'].values, 14)

    # Our predictions
    our_need_1d = forecast_result['demand_1d'] + safety_1d
    our_need_7d = forecast_result['demand_7d'] + safety_7d
    our_need_14d = forecast_result['demand_14d'] + safety_14d

    # System predictions
    if sys_daily_sales and sys_daily_sales > 0:
        sys_need_1d = sys_daily_sales * 1
        sys_need_7d = sys_daily_sales * 7
        sys_need_14d = sys_daily_sales * 14
    else:
        sys_need_1d = sys_need_7d = sys_need_14d = None

    # ORDER CALCULATIONS
    our_order_1d = max(0, our_need_1d - current_stock)
    our_order_7d = max(0, our_need_7d - current_stock)
    our_order_14d = max(0, our_need_14d - current_stock)

    if sys_need_1d is not None:
        sys_order_1d = max(0, sys_need_1d - current_stock)
        sys_order_7d = max(0, sys_need_7d - current_stock)
        sys_order_14d = max(0, sys_need_14d - current_stock)
    else:
        sys_order_1d = sys_order_7d = sys_order_14d = None

    # ================================================================
    # MAIN DISPLAY
    # ================================================================
    st.markdown("---")
    num_codes = product_row.get('Num_Codes', 1)
    st.markdown(f"## 📊 {product_name}")
    st.caption(f"Αθροισμένο από {num_codes} κωδικούς υλικού")

    # BIG ORDER RECOMMENDATION
    st.markdown("### 🛒 ΠΟΣΟΤΗΤΑ ΓΙΑ ΠΑΡΑΓΓΕΛΙΑ")

    if our_order_7d > 0:
        st.markdown(f"""
        <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; border: 3px solid #dc3545; text-align: center;">
        <h1 style="color: #721c24; margin: 0;">⚠️ ΠΑΡΑΓΓΕΙΛΕ: {our_order_7d:,.0f} κιβώτια</h1>
        <p style="color: #721c24; font-size: 16px; margin: 10px 0 0 0;">Για κάλυψη επόμενων 7 ημερών</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; border: 3px solid #28a745; text-align: center;">
        <h1 style="color: #155724; margin: 0;">✅ ΕΠΑΡΚΕΙΑ</h1>
        <p style="color: #155724; font-size: 16px; margin: 10px 0 0 0;">Το απόθεμα καλύπτει τις επόμενες 7 ημέρες</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # CURRENT STOCK
    st.markdown("### 📦 Τρέχον Απόθεμα (σύνολο όλων κωδικών)")
    st.metric("Διαθέσιμο", f"{current_stock:,} κιβώτια")

    # COMPARISON TABLE
    st.markdown("---")
    st.markdown("### ⚖️ Σύγκριση: Σύστημα vs Δική μας Πρόβλεψη")

    comparison_data = {
        'Περίοδος': ['1 Ημέρα', '7 Ημέρες', '14 Ημέρες'],
        '📋 Σύστημα - Ζήτηση': [
            f"{sys_need_1d:.0f}" if sys_need_1d else "—",
            f"{sys_need_7d:.0f}" if sys_need_7d else "—",
            f"{sys_need_14d:.0f}" if sys_need_14d else "—"
        ],
        '📋 Σύστημα - Παραγγελία': [
            f"{sys_order_1d:.0f}" if sys_order_1d is not None else "—",
            f"{sys_order_7d:.0f}" if sys_order_7d is not None else "—",
            f"{sys_order_14d:.0f}" if sys_order_14d is not None else "—"
        ],
        '🤖 Εμείς - Ζήτηση': [
            f"{our_need_1d:.0f}",
            f"{our_need_7d:.0f}",
            f"{our_need_14d:.0f}"
        ],
        '🤖 Εμείς - Παραγγελία': [
            f"{our_order_1d:.0f}" if our_order_1d > 0 else "✅ 0",
            f"{our_order_7d:.0f}" if our_order_7d > 0 else "✅ 0",
            f"{our_order_14d:.0f}" if our_order_14d > 0 else "✅ 0"
        ],
    }

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    col_legend1, col_legend2 = st.columns(2)
    with col_legend1:
        st.caption("📋 **Σύστημα**: Από Excel (SOP)")
    with col_legend2:
        method_name = "Holt-Winters" if forecast_result['method'] == 'holt_winters' else "Μέσος Όρος"
        st.caption(f"🤖 **Εμείς**: {method_name}")

    if sys_has_oos:
        st.error("📋 Υπάρχον σύστημα: **🔴 OOS**")

    # FORECAST DETAILS
    st.markdown("---")
    st.markdown("### 📈 Λεπτομέρειες Πρόβλεψης")

    col_details1, col_details2 = st.columns(2)

    with col_details1:
        st.markdown("**Δική μας Ανάλυση:**")
        st.write(f"- Μέθοδος: **{forecast_result['method'].replace('_', ' ').title()}**")
        st.write(f"- Αξιοπιστία: **{forecast_result['confidence'].upper()}**")
        st.write(f"- Ημερήσια ζήτηση: **{forecast_result['daily_avg']:.1f}** κιβώτια")
        if 'mape' in forecast_result:
            st.write(f"- Σφάλμα (MAPE): **{forecast_result['mape']:.1f}%**")

    with col_details2:
        if sys_daily_sales and sys_daily_sales > 0:
            st.markdown("**Σύστημα:**")
            st.write(f"- Ημερήσια ζήτηση: **{sys_daily_sales:.0f}** κιβώτια")
        else:
            st.warning("Δεν βρέθηκαν δεδομένα συστήματος")

    # CHART
    st.markdown("---")
    st.markdown("### 📊 Ιστορικά (Αθροισμένα)")

    if len(hist_df) > 0:
        fig = go.Figure()

        hist_df['Year'] = pd.to_datetime(hist_df['Date']).dt.year
        hist_df['Month'] = pd.to_datetime(hist_df['Date']).dt.month
        colors = {2024: '#AB63FA', 2025: '#FFA15A', 2026: '#19D3F3'}

        for year in sorted(hist_df['Year'].unique()):
            year_data = hist_df[hist_df['Year'] == year]
            color = colors.get(year, '#1f77b4')
            fig.add_trace(go.Scatter(
                x=year_data['Month'],
                y=year_data['Demand'] / 30,
                name=f'{year}',
                mode='lines+markers',
                line=dict(color=color, width=3),
                marker=dict(size=10)
            ))

        fig.add_vline(x=current_month, line_dash="dash", line_color="red",
                     annotation_text="Τώρα", annotation_position="top")

        fig.add_trace(go.Scatter(
            x=[current_month],
            y=[forecast_result['daily_avg']],
            name='Πρόβλεψη',
            mode='markers',
            marker=dict(size=20, symbol='star', color='red', line=dict(width=2, color='black'))
        ))

        fig.update_layout(
            xaxis_title='Μήνας',
            yaxis_title='Κιβώτια/Ημέρα',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(tickmode='array', tickvals=list(range(1,13)),
                      ticktext=['Ιαν','Φεβ','Μαρ','Απρ','Μαϊ','Ιουν','Ιουλ','Αυγ','Σεπ','Οκτ','Νοε','Δεκ'])
        )
        st.plotly_chart(fig, use_container_width=True)

    # ================================================================
    # ALL PRODUCTS SUMMARY
    # ================================================================
    st.markdown("---")
    st.subheader("📋 Όλα τα Προϊόντα")

    summary_data = []
    for idx, row in filtered_df.iterrows():
        hist = pd.Series(row[date_columns].values.astype(float)).dropna()
        pname = str(row['Product_Name'])[:40]
        num_codes = row.get('Num_Codes', 1)

        # Total stock for this product
        curr = get_total_stock_for_product(row, stock_df)

        # System data
        s_daily, s_oos, _ = get_system_prediction_for_product(row, sop_df)

        # Our forecast
        positive_hist = hist[hist > 0]
        our_daily = positive_hist.mean() / 30 if len(positive_hist) > 0 else 0

        # 7-day needs
        our_need = (our_daily * 7 * 1.2) if our_daily > 0 else 0
        sys_need = (s_daily * 7) if s_daily else None

        # Orders needed
        our_order = max(0, our_need - curr)
        sys_order = max(0, sys_need - curr) if sys_need else None

        summary_data.append({
            'Προϊόν': pname,
            'Κωδ.': num_codes,
            'Απόθεμα': curr,
            '🤖 Παραγγ.': int(our_order) if our_order > 0 else 0,
            '📋 Παραγγ.': int(sys_order) if sys_order and sys_order > 0 else (0 if sys_order is not None else None),
            '📋': "🔴" if s_oos else ("✅" if s_daily else "—"),
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('🤖 Παραγγ.', ascending=False)

    st.dataframe(
        summary_df.style.format({
            'Απόθεμα': lambda x: f"{x:,}" if pd.notna(x) else "—",
            '🤖 Παραγγ.': lambda x: f"{x:,}" if pd.notna(x) and x > 0 else "✅",
            '📋 Παραγγ.': lambda x: f"{x:,}" if pd.notna(x) and x > 0 else ("✅" if pd.notna(x) else "—"),
        }),
        use_container_width=True,
        height=400
    )

    st.caption("🤖 = Δική μας | 📋 = Σύστημα | Κωδ. = Αριθμός κωδικών υλικού")
    st.caption("v3.0 - Ομαδοποίηση κατά περιγραφή")

if __name__ == "__main__":
    main()
