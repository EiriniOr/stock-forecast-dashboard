"""
Stock Forecasting Dashboard - Cloud Version
Πίνακας Πρόβλεψης Αποθέματος
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Πρόβλεψη Αποθέματος", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_data(uploaded_file):
    df_raw = pd.read_excel(uploaded_file, sheet_name='Cases2021 2022 FC', header=None)

    years_row = df_raw.iloc[8, 3:27].values
    months_row = df_raw.iloc[9, 3:27].values

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
        if dc:
            col_names.append(dc)
        else:
            col_names.append(f'extra_{i}')

    remaining = len(data_df.columns) - len(col_names)
    for i in range(remaining):
        col_names.append(f'meta_{i}')

    data_df.columns = col_names[:len(data_df.columns)]
    data_df = data_df[data_df['SKU_Code'].notna()].reset_index(drop=True)

    data_df['SKU_Code'] = data_df['SKU_Code'].astype(str).str.strip()
    data_df['SKU_Short'] = data_df['SKU_Code'].str[-8:]

    date_columns = [c for c in data_df.columns if '-' in str(c) and str(c)[:4].isdigit()]

    uploaded_file.seek(0)
    df_stock = pd.read_excel(uploaded_file, sheet_name='Stock')
    df_stock['Material Code'] = df_stock['Material Code'].astype(str).str.strip()

    return data_df, df_stock, date_columns

# --- FORECASTING ---
def forecast_demand(historical_data, periods_ahead=12):
    """Forecast demand using only positive values (negative = returns)"""
    data = pd.Series(historical_data).dropna()

    # Use only positive values for forecasting (negatives are returns)
    positive_data = data[data > 0]

    if len(positive_data) < 6:
        return None, None, None

    X = np.arange(len(positive_data)).reshape(-1, 1)
    y = positive_data.values

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(positive_data), len(positive_data) + periods_ahead).reshape(-1, 1)
    forecast = model.predict(future_X)

    fitted = model.predict(X)
    residuals = y - fitted
    rmse = np.sqrt(np.mean(residuals**2))

    # R² and CV for reliability
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    cv = (positive_data.std() / positive_data.mean() * 100) if positive_data.mean() > 0 else 100

    if len(positive_data) >= 12:
        monthly_avg = []
        for m in range(12):
            month_vals = [positive_data.iloc[i] for i in range(m, len(positive_data), 12) if i < len(positive_data)]
            if month_vals:
                monthly_avg.append(np.mean(month_vals))
            else:
                monthly_avg.append(np.mean(positive_data))

        overall_avg = np.mean(positive_data)
        seasonal_factors = [m / overall_avg if overall_avg > 0 else 1 for m in monthly_avg]

        start_month = len(positive_data) % 12
        for i in range(len(forecast)):
            month_idx = (start_month + i) % 12
            forecast[i] *= seasonal_factors[month_idx]

    forecast = np.maximum(forecast, 0)

    reliability = {'r_squared': r_squared, 'cv': cv, 'rmse': rmse}
    return forecast, rmse, reliability

def calculate_safety_stock(historical_data, service_level=0.95):
    """Calculate safety stock using only positive demand values"""
    data = pd.Series(historical_data).dropna()
    positive_data = data[data > 0]

    if len(positive_data) < 3:
        return 0

    z_scores = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
    z = z_scores.get(service_level, 1.65)
    std_dev = positive_data.std()
    return max(0, z * std_dev)

def get_reliability_level(r2, cv):
    if r2 >= 0.7 and cv < 30:
        return "high", "✅", "Υψηλή αξιοπιστία"
    elif r2 >= 0.4 and cv < 50:
        return "medium", "⚡", "Μέτρια αξιοπιστία"
    else:
        return "low", "🔴", "Χαμηλή αξιοπιστία"

def analyze_product_pattern(historical_data):
    """Analyze if product has mostly returns (negative) or sales (positive)"""
    data = pd.Series(historical_data).dropna()
    if len(data) == 0:
        return "no_data", "Χωρίς δεδομένα"

    neg_count = (data < 0).sum()
    pos_count = (data > 0).sum()
    total = len(data)

    if neg_count > pos_count:
        return "returns", f"⚠️ Κυρίως επιστροφές ({neg_count}/{total} αρνητικά)"
    elif pos_count < 6:
        return "low_data", f"Λίγα δεδομένα πωλήσεων ({pos_count} μήνες)"
    else:
        return "normal", None

# --- MAIN APP ---
def main():
    st.title("📦 Πρόβλεψη Αποθέματος")

    uploaded_file = st.file_uploader(
        "Ανέβασε το αρχείο Excel",
        type=['xlsx'],
        help="Αρχείο με sheets 'Cases2021 2022 FC' και 'Stock'"
    )

    if uploaded_file is None:
        st.info("👆 Ανέβασε το αρχείο Excel για να ξεκινήσεις")
        return

    try:
        data_df, stock_df, date_columns = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Σφάλμα: {e}")
        return

    # Initialize session state for selected product
    if 'selected_sku' not in st.session_state:
        st.session_state.selected_sku = None

    # --- PRODUCT SELECTION ---
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        categories = ['Όλες οι κατηγορίες'] + sorted(data_df['Category'].dropna().unique().tolist())
        selected_category = st.selectbox("Κατηγορία", categories)

    if selected_category != 'Όλες οι κατηγορίες':
        filtered_df = data_df[data_df['Category'] == selected_category].reset_index(drop=True)
    else:
        filtered_df = data_df.reset_index(drop=True)

    product_options = filtered_df.apply(
        lambda x: f"{x['SKU_Short']} - {str(x['Product_Name'])[:50]}", axis=1
    ).tolist()

    # Find default index - use session state if set from table click
    default_idx = 0
    if st.session_state.selected_sku:
        for i, opt in enumerate(product_options):
            if opt.startswith(st.session_state.selected_sku):
                default_idx = i
                break

    with col2:
        selected_product = st.selectbox("Προϊόν", product_options, index=default_idx)

    with col3:
        forecast_months = st.selectbox("Μήνες πρόβλεψης", [1, 3, 6, 12], index=1)

    if not product_options:
        st.warning("Δεν βρέθηκαν προϊόντα")
        return

    # Get selected product data
    selected_idx = product_options.index(selected_product)
    product_row = filtered_df.iloc[selected_idx]
    sku_short = product_row['SKU_Short']

    # Get stock and historical data
    stock_match = stock_df[stock_df['Material Code'].str.contains(sku_short, na=False)]
    current_stock = stock_match.iloc[0]['Unrestricted Stock'] if not stock_match.empty else None

    historical = product_row[date_columns].values.astype(float)
    dates = pd.to_datetime([f"{d}-01" for d in date_columns])
    hist_df = pd.DataFrame({'Date': dates, 'Demand': historical}).dropna()

    # --- MAIN ANALYSIS ---
    st.markdown("---")
    st.subheader(f"📊 {product_row['Product_Name']}")

    # Check product pattern (returns vs sales)
    pattern, pattern_msg = analyze_product_pattern(hist_df['Demand'].values)

    if pattern == "returns":
        st.error(pattern_msg)
        st.info("Αυτό το προϊόν έχει κυρίως αρνητικές τιμές (επιστροφές/πιστώσεις). Δεν είναι δυνατή η πρόβλεψη ζήτησης.")

        # Show the raw data
        with st.expander("📋 Δες τα δεδομένα"):
            st.dataframe(hist_df, use_container_width=True)

    elif pattern == "low_data":
        st.warning(f"⚠️ {pattern_msg}")
        if len(hist_df) > 0:
            positive_avg = hist_df[hist_df['Demand'] > 0]['Demand'].mean()
            if pd.notna(positive_avg):
                fallback = positive_avg * forecast_months * 1.3
                st.info(f"Εκτίμηση με buffer 30%: **{fallback:,.0f}** για {forecast_months} μήνες")

    else:
        # Normal product with enough positive data
        positive_hist = hist_df[hist_df['Demand'] > 0].copy()

        forecast, rmse, reliability = forecast_demand(hist_df['Demand'].values, forecast_months)
        safety_stock = calculate_safety_stock(hist_df['Demand'].values, 0.95)

        if forecast is not None:
            total_forecast = forecast.sum()
            recommended = total_forecast + safety_stock
            level, icon, message = get_reliability_level(reliability['r_squared'], reliability['cv'])

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("🎯 Προτεινόμενο Απόθεμα", f"{recommended:,.0f}")

            with col2:
                st.metric("📈 Πρόβλεψη Ζήτησης", f"{total_forecast:,.0f}")

            with col3:
                st.metric("🛡️ Safety Stock", f"{safety_stock:,.0f}")

            with col4:
                if current_stock is not None:
                    diff = current_stock - recommended
                    st.metric("📦 Τρέχον Απόθεμα", f"{int(current_stock):,}", f"{diff:+,.0f}")
                else:
                    st.metric("📦 Τρέχον Απόθεμα", "—")

            # Reliability box
            if level == "high":
                st.success(f"{icon} {message} — R²={reliability['r_squared']:.2f}, CV={reliability['cv']:.0f}%")
            elif level == "medium":
                st.warning(f"{icon} {message} — Προτείνεται buffer +15-20% — R²={reliability['r_squared']:.2f}, CV={reliability['cv']:.0f}%")
            else:
                st.error(f"{icon} {message} — Προτείνεται buffer +30-50% — R²={reliability['r_squared']:.2f}, CV={reliability['cv']:.0f}%")

            st.caption("R² = πόσο καλά ταιριάζει το μοντέλο (>0.7 καλό) | CV = μεταβλητότητα ζήτησης (<30% σταθερή)")

            # --- COMBINED CHART: Historical + Forecast ---
            st.markdown("---")

            # Prepare forecast dates
            forecast_dates = pd.date_range(
                start=hist_df['Date'].max() + pd.DateOffset(months=1),
                periods=forecast_months, freq='MS'
            )

            fig = go.Figure()

            # Historical data - color by year, show ALL values (including negative)
            hist_df['Year'] = hist_df['Date'].dt.year
            years = hist_df['Year'].unique()
            colors = {'2021': '#636EFA', '2022': '#EF553B', '2023': '#00CC96', '2024': '#AB63FA', '2025': '#FFA15A'}

            for year in sorted(years):
                year_data = hist_df[hist_df['Year'] == year]
                color = colors.get(str(year), '#1f77b4')
                fig.add_trace(go.Scatter(
                    x=year_data['Date'],
                    y=year_data['Demand'],
                    name=f'{year}',
                    mode='lines+markers',
                    line=dict(color=color, width=2),
                    marker=dict(size=6)
                ))

            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast,
                name='Πρόβλεψη',
                mode='lines+markers',
                line=dict(color='#d62728', width=3, dash='dash'),
                marker=dict(size=8, symbol='diamond')
            ))

            # Confidence interval (±1.5 RMSE)
            upper = forecast + 1.5 * rmse
            lower = np.maximum(forecast - 1.5 * rmse, 0)

            fig.add_trace(go.Scatter(
                x=list(forecast_dates) + list(forecast_dates[::-1]),
                y=list(upper) + list(lower[::-1]),
                fill='toself',
                fillcolor='rgba(214, 39, 40, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Εύρος αβεβαιότητας',
                showlegend=True
            ))

            # Add zero line if there are negative values
            if (hist_df['Demand'] < 0).any():
                fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)

            fig.update_layout(
                title='Ιστορική Ζήτηση & Πρόβλεψη',
                xaxis_title='',
                yaxis_title='Κιβώτια',
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Note about negative values
            neg_count = (hist_df['Demand'] < 0).sum()
            if neg_count > 0:
                st.caption(f"ℹ️ Αρνητικές τιμές ({neg_count} μήνες) = επιστροφές/πιστώσεις. Η πρόβλεψη βασίζεται μόνο στις θετικές πωλήσεις.")

            # Monthly forecast breakdown
            st.markdown("**Ανάλυση πρόβλεψης ανά μήνα:**")
            monthly_df = pd.DataFrame({
                'Μήνας': forecast_dates.strftime('%b %Y'),
                'Πρόβλεψη': forecast.round(0).astype(int),
                'Με Safety Stock': (forecast + safety_stock/forecast_months).round(0).astype(int)
            })
            st.dataframe(monthly_df, use_container_width=True, hide_index=True)

    # --- ALL PRODUCTS SUMMARY ---
    st.markdown("---")
    st.subheader("📋 Όλα τα Προϊόντα")
    st.caption("Κάνε κλικ σε ένα όνομα για να δεις την ανάλυση")

    # Build summary with clickable product names
    summary_data = []
    for idx, row in filtered_df.iterrows():
        hist = pd.Series(row[date_columns].values.astype(float)).dropna()
        sku = row['SKU_Short']
        product_name = str(row['Product_Name'])[:35]

        stock_row = stock_df[stock_df['Material Code'].str.contains(sku, na=False)]
        curr = stock_row.iloc[0]['Unrestricted Stock'] if not stock_row.empty else None

        # Check pattern
        pattern, _ = analyze_product_pattern(hist.values)

        if pattern == "returns":
            rec, status, icon = None, "↩️", "↩️"
        elif len(hist[hist > 0]) >= 6:
            fc, _, rel = forecast_demand(hist.values, 3)
            if fc is not None:
                rec = fc.sum() + calculate_safety_stock(hist.values, 0.95)
                status = "✅" if curr and curr >= rec else "🔴" if curr else "—"
                _, icon, _ = get_reliability_level(rel['r_squared'], rel['cv'])
            else:
                rec, status, icon = None, "—", "⚠️"
        else:
            rec, status, icon = None, "—", "⚠️"

        summary_data.append({
            'idx': idx,
            'Κατάσταση': status,
            'SKU': sku,
            'Προϊόν': product_name,
            'Τρέχον': int(curr) if curr else None,
            'Σύσταση 3μ': int(rec) if rec else None,
            'Αξιοπ.': icon
        })

    # Display as clickable list
    cols = st.columns([0.5, 1, 3, 1.2, 1.2, 0.5])
    cols[0].markdown("**Στ.**")
    cols[1].markdown("**SKU**")
    cols[2].markdown("**Προϊόν**")
    cols[3].markdown("**Τρέχον**")
    cols[4].markdown("**Σύσταση**")
    cols[5].markdown("**Αξ.**")

    st.markdown("<hr style='margin: 5px 0'>", unsafe_allow_html=True)

    # Create scrollable container
    container = st.container(height=400)

    with container:
        for item in summary_data:
            cols = st.columns([0.5, 1, 3, 1.2, 1.2, 0.5])
            cols[0].write(item['Κατάσταση'])
            cols[1].write(item['SKU'])

            # Clickable product name
            if cols[2].button(item['Προϊόν'], key=f"btn_{item['SKU']}", use_container_width=True):
                st.session_state.selected_sku = item['SKU']
                st.rerun()

            cols[3].write(f"{item['Τρέχον']:,}" if item['Τρέχον'] else "—")
            cols[4].write(f"{item['Σύσταση 3μ']:,}" if item['Σύσταση 3μ'] else "—")
            cols[5].write(item['Αξιοπ.'])

    st.caption("✅ Επαρκές | 🔴 Ανεπαρκές | ↩️ Επιστροφές | ⚠️ Λίγα δεδομένα | ⚡ Μέτρια αξιοπ.")

if __name__ == "__main__":
    main()
