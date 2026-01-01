"""
Stock Forecasting Dashboard - Cloud Version
Πίνακας Πρόβλεψης Αποθέματος
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Πρόβλεψη Αποθέματος", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_data(uploaded_file):
    """Load all relevant sheets from Excel"""

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

    return data_df, stock_by_material, df_sop, date_columns

# --- ADVANCED FORECASTING ---
def prepare_features(historical_data, dates):
    """Create features for ML model"""
    df = pd.DataFrame({'date': dates, 'demand': historical_data})
    df = df.dropna()
    if len(df) < 6:
        return None, None, None

    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Lag features
    df['lag_1'] = df['demand'].shift(1)
    df['lag_2'] = df['demand'].shift(2)
    df['lag_3'] = df['demand'].shift(3)
    df['rolling_mean_3'] = df['demand'].rolling(3).mean()
    df['rolling_std_3'] = df['demand'].rolling(3).std()
    df['rolling_mean_6'] = df['demand'].rolling(6).mean()

    # Trend
    df['trend'] = np.arange(len(df))

    df = df.dropna()

    feature_cols = ['month_sin', 'month_cos', 'quarter', 'lag_1', 'lag_2', 'lag_3',
                    'rolling_mean_3', 'rolling_std_3', 'rolling_mean_6', 'trend']

    return df, feature_cols, df['demand'].values

def train_ensemble_model(historical_data, dates):
    """Train an ensemble model for better predictions"""
    df, feature_cols, y = prepare_features(historical_data, dates)
    if df is None or len(df) < 6:
        return None, None, None, None

    X = df[feature_cols].values

    # Use only positive values for training
    positive_mask = y > 0
    if positive_mask.sum() < 6:
        return None, None, None, None

    X_pos = X[positive_mask]
    y_pos = y[positive_mask]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pos)

    # Train Gradient Boosting model
    model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X_scaled, y_pos)

    # Calculate metrics
    predictions = model.predict(X_scaled)
    rmse = np.sqrt(np.mean((y_pos - predictions) ** 2))

    ss_res = np.sum((y_pos - predictions) ** 2)
    ss_tot = np.sum((y_pos - np.mean(y_pos)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    cv = (np.std(y_pos) / np.mean(y_pos) * 100) if np.mean(y_pos) > 0 else 100

    return model, scaler, df, {'r_squared': r_squared, 'cv': cv, 'rmse': rmse, 'feature_cols': feature_cols}

def forecast_daily(model, scaler, df_train, feature_cols, days_ahead, monthly_avg):
    """Forecast for next N days"""
    if model is None:
        return None

    # Get last known values for lag features
    last_values = df_train['demand'].values[-3:]
    last_rolling_3 = df_train['rolling_mean_3'].values[-1]
    last_rolling_6 = df_train['rolling_mean_6'].values[-1]
    last_std = df_train['rolling_std_3'].values[-1]
    last_trend = df_train['trend'].values[-1]

    # Monthly average to daily
    daily_avg = monthly_avg / 30

    forecasts = []
    current_date = df_train['date'].max()

    for day in range(days_ahead):
        future_date = current_date + timedelta(days=day + 1)
        month = future_date.month
        quarter = (month - 1) // 3 + 1

        features = np.array([[
            np.sin(2 * np.pi * month / 12),  # month_sin
            np.cos(2 * np.pi * month / 12),  # month_cos
            quarter,
            last_values[-1] if len(last_values) > 0 else daily_avg * 30,  # lag_1
            last_values[-2] if len(last_values) > 1 else daily_avg * 30,  # lag_2
            last_values[-3] if len(last_values) > 2 else daily_avg * 30,  # lag_3
            last_rolling_3,
            last_std,
            last_rolling_6,
            last_trend + day + 1
        ]])

        features_scaled = scaler.transform(features)
        monthly_pred = max(0, model.predict(features_scaled)[0])
        daily_pred = monthly_pred / 30
        forecasts.append(daily_pred)

        # Update lag values for next iteration
        if len(forecasts) >= 30:
            last_values = np.array(forecasts[-3:]) * 30

    return np.array(forecasts)

def calculate_safety_stock(historical_data, days=1, service_level=0.95):
    """Calculate safety stock for given number of days"""
    data = pd.Series(historical_data).dropna()
    positive_data = data[data > 0]
    if len(positive_data) < 3:
        return 0

    daily_std = positive_data.std() / 30  # Convert monthly to daily
    z_scores = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
    z = z_scores.get(service_level, 1.65)
    return max(0, z * daily_std * np.sqrt(days))

def get_reliability_level(r2, cv):
    if r2 >= 0.7 and cv < 30:
        return "high", "✅", "Υψηλή αξιοπιστία"
    elif r2 >= 0.4 and cv < 50:
        return "medium", "⚡", "Μέτρια αξιοπιστία"
    else:
        return "low", "🔴", "Χαμηλή αξιοπιστία"

# --- MAIN APP ---
def main():
    st.title("📦 Πρόβλεψη Αποθέματος")
    st.caption("Πρόβλεψη για ημέρα, εβδομάδα, δεκαπενθήμερο")

    uploaded_file = st.file_uploader(
        "Ανέβασε το αρχείο Excel",
        type=['xlsx'],
        help="Αρχείο με sheets 'Cases2021 2022 FC', 'S1P', και 'SOP'"
    )

    if uploaded_file is None:
        st.info("👆 Ανέβασε το αρχείο Excel για να ξεκινήσεις")
        return

    try:
        data_df, stock_df, sop_df, date_columns = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Σφάλμα: {e}")
        return

    # Initialize session state
    if 'selected_material' not in st.session_state:
        st.session_state.selected_material = None

    # --- PRODUCT SELECTION ---
    st.markdown("---")
    col1, col2 = st.columns([2, 3])

    with col1:
        categories = ['Όλες οι κατηγορίες'] + sorted(data_df['Category'].dropna().unique().tolist())
        selected_category = st.selectbox("Κατηγορία", categories)

    if selected_category != 'Όλες οι κατηγορίες':
        filtered_df = data_df[data_df['Category'] == selected_category].reset_index(drop=True)
    else:
        filtered_df = data_df.reset_index(drop=True)

    product_options = filtered_df.apply(
        lambda x: f"{x['Material']} - {str(x['Product_Name'])[:45]}", axis=1
    ).tolist()

    default_idx = 0
    if st.session_state.selected_material:
        for i, opt in enumerate(product_options):
            if opt.startswith(st.session_state.selected_material):
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
    material = product_row['Material']

    # Get current stock from S1P
    stock_match = stock_df[stock_df['Material'] == material]
    current_stock = int(stock_match['Unrestricted stock'].values[0]) if not stock_match.empty else 0

    # Get existing system's prediction from SOP
    sop_match = sop_df[sop_df['MRDR'].str.contains(material, na=False)]
    existing_stock_days = None
    existing_daily_sales = None
    existing_status = None
    if not sop_match.empty:
        existing_stock_days = sop_match['Stock_Days'].values[0]
        existing_daily_sales = sop_match['Avg_Daily_Sales'].values[0]
        existing_status = sop_match['Status'].values[0]

    # Historical data
    historical = product_row[date_columns].values.astype(float)
    dates = pd.to_datetime([f"{d}-01" for d in date_columns])
    hist_df = pd.DataFrame({'Date': dates, 'Demand': historical}).dropna()

    # --- MAIN ANALYSIS ---
    st.markdown("---")
    st.subheader(f"📊 {product_row['Product_Name']}")

    # Train model and make predictions
    positive_data = hist_df[hist_df['Demand'] > 0]['Demand']
    monthly_avg = positive_data.mean() if len(positive_data) > 0 else 0
    daily_avg = monthly_avg / 30

    model, scaler, df_train, reliability = train_ensemble_model(
        hist_df['Demand'].values, hist_df['Date'].values
    )

    # Calculate predictions for different time horizons
    if model is not None:
        forecast_1d = forecast_daily(model, scaler, df_train, reliability['feature_cols'], 1, monthly_avg)
        forecast_7d = forecast_daily(model, scaler, df_train, reliability['feature_cols'], 7, monthly_avg)
        forecast_14d = forecast_daily(model, scaler, df_train, reliability['feature_cols'], 14, monthly_avg)

        demand_1d = forecast_1d.sum() if forecast_1d is not None else daily_avg
        demand_7d = forecast_7d.sum() if forecast_7d is not None else daily_avg * 7
        demand_14d = forecast_14d.sum() if forecast_14d is not None else daily_avg * 14

        safety_1d = calculate_safety_stock(hist_df['Demand'].values, 1)
        safety_7d = calculate_safety_stock(hist_df['Demand'].values, 7)
        safety_14d = calculate_safety_stock(hist_df['Demand'].values, 14)

        needed_1d = demand_1d + safety_1d
        needed_7d = demand_7d + safety_7d
        needed_14d = demand_14d + safety_14d

        level, icon, message = get_reliability_level(reliability['r_squared'], reliability['cv'])
    else:
        # Fallback to simple average
        demand_1d = daily_avg
        demand_7d = daily_avg * 7
        demand_14d = daily_avg * 14
        needed_1d = demand_1d * 1.2
        needed_7d = demand_7d * 1.2
        needed_14d = demand_14d * 1.2
        level, icon, message = "low", "⚠️", "Απλή εκτίμηση (λίγα δεδομένα)"
        reliability = {'r_squared': 0, 'cv': 100}

    # Days of stock remaining
    days_remaining = current_stock / daily_avg if daily_avg > 0 else float('inf')

    # --- CURRENT STOCK INFO ---
    st.markdown("### 📦 Τρέχον Απόθεμα (από S1P)")
    col_stock1, col_stock2 = st.columns(2)
    with col_stock1:
        st.metric("Διαθέσιμο Απόθεμα", f"{current_stock:,} κιβώτια")
    with col_stock2:
        if days_remaining != float('inf'):
            st.metric("Ημέρες Κάλυψης", f"{days_remaining:.0f} ημέρες")
        else:
            st.metric("Ημέρες Κάλυψης", "—")

    # ================================================================
    # SIDE-BY-SIDE COMPARISON: OUR PREDICTION vs SYSTEM PREDICTION
    # ================================================================
    st.markdown("---")
    st.markdown("## ⚖️ Σύγκριση Προβλέψεων")

    left_col, right_col = st.columns(2)

    # --- LEFT: OUR PREDICTION ---
    with left_col:
        st.markdown("""
        <div style="background-color: #d4edda; padding: 15px; border-radius: 10px; border: 2px solid #28a745;">
        <h3 style="color: #155724; margin: 0;">🤖 ΔΙΚΗ ΜΑΣ ΠΡΟΒΛΕΨΗ</h3>
        <p style="color: #155724; font-size: 12px; margin: 5px 0 0 0;">Machine Learning (Gradient Boosting)</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # Our predictions
        our_status_7d = "✅ ΕΠΑΡΚΕΙΑ" if current_stock >= needed_7d else "🔴 ΕΛΛΕΙΨΗ"
        our_status_color = "green" if current_stock >= needed_7d else "red"

        st.markdown(f"**Κατάσταση 7 ημερών:** <span style='color:{our_status_color}; font-weight:bold;'>{our_status_7d}</span>", unsafe_allow_html=True)

        st.metric("📅 Ημέρες αποθέματος", f"{days_remaining:.0f}" if days_remaining != float('inf') else "—")
        st.metric("🔮 Ανάγκη 1 ημέρα", f"{needed_1d:,.0f}")
        st.metric("📅 Ανάγκη 7 ημέρες", f"{needed_7d:,.0f}")
        st.metric("📆 Ανάγκη 14 ημέρες", f"{needed_14d:,.0f}")

        # Reliability indicator
        if level == "high":
            st.success(f"{icon} {message}")
        elif level == "medium":
            st.warning(f"{icon} {message}")
        else:
            st.error(f"{icon} {message}")

    # --- RIGHT: EXISTING SYSTEM PREDICTION ---
    with right_col:
        st.markdown("""
        <div style="background-color: #fff3cd; padding: 15px; border-radius: 10px; border: 2px solid #ffc107;">
        <h3 style="color: #856404; margin: 0;">📋 ΥΠΑΡΧΟΝ ΣΥΣΤΗΜΑ</h3>
        <p style="color: #856404; font-size: 12px; margin: 5px 0 0 0;">Από το Excel αρχείο (SOP sheet)</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        if existing_stock_days is not None and pd.notna(existing_stock_days):
            sys_status_display = "🔴 OOS" if existing_status == "OOS" else "✅ OK"
            sys_status_color = "red" if existing_status == "OOS" else "green"

            st.markdown(f"**Κατάσταση:** <span style='color:{sys_status_color}; font-weight:bold;'>{sys_status_display}</span>", unsafe_allow_html=True)

            st.metric("📊 Ημέρες αποθέματος", f"{existing_stock_days:.0f}")
            if pd.notna(existing_daily_sales):
                st.metric("📈 Μ.Ο. ημερ. πωλήσεις", f"{existing_daily_sales:.0f}")
            else:
                st.metric("📈 Μ.Ο. ημερ. πωλήσεις", "—")

            # Comparison
            if days_remaining != float('inf') and existing_stock_days > 0:
                diff_days = days_remaining - existing_stock_days
                diff_pct = (diff_days / existing_stock_days) * 100
                if abs(diff_pct) > 10:
                    st.info(f"Διαφορά: {diff_days:+.0f} ημέρες ({diff_pct:+.0f}%)")
        else:
            st.warning("Δεν βρέθηκε πρόβλεψη για αυτό το προϊόν στο υπάρχον σύστημα")

    # --- CHART ---
    st.markdown("---")
    st.markdown("### 📈 Ιστορικά & Πρόβλεψη (Δικό μας μοντέλο)")

    fig = go.Figure()

    # Historical data by year
    if len(hist_df) > 0:
        hist_df['Year'] = pd.to_datetime(hist_df['Date']).dt.year
        hist_df['Month'] = pd.to_datetime(hist_df['Date']).dt.month
        colors = {'2024': '#AB63FA', '2025': '#FFA15A', '2026': '#19D3F3'}

        for year in sorted(hist_df['Year'].unique()):
            year_data = hist_df[hist_df['Year'] == year]
            color = colors.get(str(year), '#1f77b4')
            # Convert monthly to daily for display consistency
            fig.add_trace(go.Scatter(
                x=year_data['Date'],
                y=year_data['Demand'] / 30,  # Daily average
                name=f'{year} (ημερ.)',
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=6)
            ))

        # Current month highlight
        current_month = datetime.now().month
        current_month_data = hist_df[hist_df['Month'] == current_month]
        if len(current_month_data) > 0:
            fig.add_trace(go.Scatter(
                x=current_month_data['Date'],
                y=current_month_data['Demand'] / 30,
                name=f'Τρέχων μήνας',
                mode='markers',
                marker=dict(size=14, symbol='circle-open', color='black', line=dict(width=3))
            ))

    # Add forecast line
    if model is not None and forecast_14d is not None:
        forecast_dates = pd.date_range(
            start=hist_df['Date'].max() + pd.DateOffset(days=1),
            periods=14, freq='D'
        )
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_14d,
            name='Πρόβλεψη (14 ημ.)',
            mode='lines+markers',
            line=dict(color='#d62728', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        ))

    # Zero line
    if len(hist_df) > 0 and (hist_df['Demand'] < 0).any():
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)

    fig.update_layout(
        title='Ημερήσια Ζήτηση & Πρόβλεψη',
        xaxis_title='',
        yaxis_title='Κιβώτια/Ημέρα',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- FORECAST BREAKDOWN ---
    if model is not None:
        st.markdown("**Ανάλυση Πρόβλεψης:**")
        breakdown_df = pd.DataFrame({
            'Περίοδος': ['1 ημέρα', '7 ημέρες', '14 ημέρες'],
            'Πρόβλεψη Ζήτησης': [f"{demand_1d:.0f}", f"{demand_7d:.0f}", f"{demand_14d:.0f}"],
            'Safety Stock': [f"{safety_1d:.0f}", f"{safety_7d:.0f}", f"{safety_14d:.0f}"],
            'Σύνολο Απαιτ.': [f"{needed_1d:.0f}", f"{needed_7d:.0f}", f"{needed_14d:.0f}"],
            'Κατάσταση': [
                "✅" if current_stock >= needed_1d else "🔴",
                "✅" if current_stock >= needed_7d else "🔴",
                "✅" if current_stock >= needed_14d else "🔴"
            ]
        })
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    # --- ALL PRODUCTS SUMMARY ---
    st.markdown("---")
    st.subheader("📋 Σύνοψη Όλων των Προϊόντων")
    st.caption("🤖 = Δική μας πρόβλεψη | 📋 = Υπάρχον σύστημα (Excel)")

    summary_data = []
    for idx, row in filtered_df.iterrows():
        hist = pd.Series(row[date_columns].values.astype(float)).dropna()
        mat = row['Material']
        product_name = str(row['Product_Name'])[:35]

        # Stock from S1P
        stock_row = stock_df[stock_df['Material'] == mat]
        curr = int(stock_row['Unrestricted stock'].values[0]) if not stock_row.empty else 0

        # Existing system data
        sop_row = sop_df[sop_df['MRDR'].str.contains(mat, na=False)]
        sys_days = sop_row['Stock_Days'].values[0] if not sop_row.empty and pd.notna(sop_row['Stock_Days'].values[0]) else None
        sys_status = sop_row['Status'].values[0] if not sop_row.empty else None

        positive_hist = hist[hist > 0]
        daily_avg = positive_hist.mean() / 30 if len(positive_hist) > 0 else 0

        # Simple 7-day forecast
        need_7d = daily_avg * 7 * 1.2 if daily_avg > 0 else None
        our_status = "✅" if curr >= (need_7d or 0) else "🔴" if need_7d else "⚠️"
        our_days = curr / daily_avg if daily_avg > 0 else None

        # System status
        sys_status_display = "🔴" if sys_status == "OOS" else "✅" if pd.notna(sys_status) else "—"

        summary_data.append({
            '🤖': our_status,
            '📋': sys_status_display,
            'Material': mat,
            'Προϊόν': product_name,
            'Απόθεμα (S1P)': curr,
            '🤖 Ημέρες': int(our_days) if our_days else None,
            '📋 Ημέρες': int(sys_days) if sys_days else None,
        })

    summary_df = pd.DataFrame(summary_data)

    # Quick selector
    product_from_table = st.selectbox(
        "Γρήγορη επιλογή:",
        options=summary_df['Προϊόν'].tolist(),
        index=None,
        placeholder="Επίλεξε προϊόν...",
        key="table_selector"
    )

    if product_from_table:
        selected_row = summary_df[summary_df['Προϊόν'] == product_from_table].iloc[0]
        if selected_row['Material'] != st.session_state.selected_material:
            st.session_state.selected_material = selected_row['Material']
            st.rerun()

    st.dataframe(
        summary_df.style.format({
            'Απόθεμα (S1P)': lambda x: f"{x:,}" if pd.notna(x) else "—",
            '🤖 Ημέρες': lambda x: f"{x:,}" if pd.notna(x) else "—",
            '📋 Ημέρες': lambda x: f"{x:,}" if pd.notna(x) else "—",
        }),
        use_container_width=True,
        height=400
    )

    st.caption("✅ = Επαρκές | 🔴 = Ανεπαρκές/OOS | ⚠️ = Λίγα δεδομένα | — = Δεν υπάρχει")

if __name__ == "__main__":
    main()
