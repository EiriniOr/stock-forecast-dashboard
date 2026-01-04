"""
Stock Dashboard v5.0
Two tabs: OOS Risk, Promote Sales
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Dashboard", layout="wide")

MONTH_NAMES_GR = ['Ιαν', 'Φεβ', 'Μαρ', 'Απρ', 'Μαϊ', 'Ιουν', 'Ιουλ', 'Αυγ', 'Σεπ', 'Οκτ', 'Νοε', 'Δεκ']

@st.cache_data
def load_all_data(uploaded_file):
    """Load S1P and Cases sheets"""
    xl = pd.ExcelFile(uploaded_file)

    # --- S1P Sheet: Stock, descriptions, and shelf life ---
    df_s1p = pd.read_excel(uploaded_file, sheet_name='S1P')

    # Find Unrestricted Stock column
    unrestricted_col = None
    for col in df_s1p.columns:
        col_lower = str(col).lower()
        if 'unrestricted' in col_lower:
            unrestricted_col = col
            break

    if unrestricted_col is None:
        raise ValueError(f"S1P sheet: δεν βρέθηκε στήλη unrestricted. Διαθέσιμες στήλες: {list(df_s1p.columns)}")

    # Find description column in S1P
    s1p_desc_col = None
    for col in df_s1p.columns:
        col_lower = str(col).lower()
        if 'description' in col_lower or 'περιγραφή' in col_lower:
            s1p_desc_col = col
            break

    if s1p_desc_col is None:
        raise ValueError(f"S1P sheet: δεν βρέθηκε στήλη description. Διαθέσιμες στήλες: {list(df_s1p.columns)}")

    # Aggregate stock by description
    df_s1p[unrestricted_col] = pd.to_numeric(df_s1p[unrestricted_col], errors='coerce').fillna(0)
    stock_by_desc = df_s1p.groupby(s1p_desc_col)[unrestricted_col].sum().reset_index()
    stock_by_desc.columns = ['Description', 'Stock']

    s1p_descriptions = set(df_s1p[s1p_desc_col].dropna().astype(str).str.strip().tolist())

    # Find shelf life column
    shelf_col = None
    for col in df_s1p.columns:
        if 'shelf' in str(col).lower() or 'λήξη' in str(col).lower() or 'expir' in str(col).lower():
            shelf_col = col
            break

    # --- Cases Sheet: Historical data and system averages ---
    df_cases_raw = pd.read_excel(uploaded_file, sheet_name='Cases2021 2022 FC', header=None)

    # Extract years from row 8 and months from row 9
    years_row = df_cases_raw.iloc[8, 3:27].values
    months_row = df_cases_raw.iloc[9, 3:27].values

    date_cols = []
    for y, m in zip(years_row, months_row):
        if pd.notna(y) and pd.notna(m):
            try:
                date_cols.append(f"{int(y)}-{int(m):02d}")
            except:
                date_cols.append(None)
        else:
            date_cols.append(None)

    # Get data from row 10
    df_cases = df_cases_raw.iloc[10:].copy()

    # Build column names
    base_cols = ['Category', 'SKU_Code', 'Product_Name']
    col_names = base_cols.copy()
    for i, dc in enumerate(date_cols):
        col_names.append(dc if dc else f'extra_{i}')

    # Add remaining columns (includes averages)
    remaining = len(df_cases.columns) - len(col_names)
    for i in range(remaining):
        col_names.append(f'meta_{i}')

    df_cases.columns = col_names[:len(df_cases.columns)]
    df_cases = df_cases[df_cases['SKU_Code'].notna()].reset_index(drop=True)

    # Find Average columns from raw header
    avg_3m_col = None
    avg_6m_col = None
    header_row = df_cases_raw.iloc[9].values
    for i, val in enumerate(header_row):
        val_str = str(val).lower()
        if '3' in val_str and ('μην' in val_str or 'mhn' in val_str or 'avg' in val_str):
            avg_3m_col = i
        if '6' in val_str and ('μην' in val_str or 'mhn' in val_str or 'avg' in val_str):
            avg_6m_col = i

    # Extract system averages if found
    if avg_3m_col is not None:
        df_cases['Avg_3m_System'] = pd.to_numeric(df_cases_raw.iloc[10:, avg_3m_col].values, errors='coerce')
    else:
        df_cases['Avg_3m_System'] = np.nan

    if avg_6m_col is not None:
        df_cases['Avg_6m_System'] = pd.to_numeric(df_cases_raw.iloc[10:, avg_6m_col].values, errors='coerce')
    else:
        df_cases['Avg_6m_System'] = np.nan

    date_columns = [c for c in df_cases.columns if '-' in str(c) and str(c)[:4].isdigit()]

    return stock_by_desc, df_s1p, s1p_descriptions, df_cases, date_columns, shelf_col, s1p_desc_col

def get_seasonal_forecast(historical_values, date_columns, months_ahead=3):
    """Forecast using same-month historical averages"""
    if len(historical_values) < 12:
        # Not enough data for seasonal, use simple average
        avg = np.nanmean(historical_values)
        return [avg] * months_ahead, None

    # Build month -> values mapping
    month_data = {m: [] for m in range(1, 13)}
    for i, col in enumerate(date_columns):
        try:
            month = int(col.split('-')[1])
            val = historical_values[i]
            if pd.notna(val) and val >= 0:
                month_data[month].append(val)
        except:
            continue

    # Calculate seasonal averages
    seasonal_avg = {}
    for m in range(1, 13):
        if month_data[m]:
            seasonal_avg[m] = np.mean(month_data[m])
        else:
            seasonal_avg[m] = np.nanmean(historical_values)

    # Get forecast for next months
    current_month = datetime.now().month
    forecast = []
    forecast_months = []
    for i in range(months_ahead):
        target_month = ((current_month - 1 + i) % 12) + 1
        forecast.append(seasonal_avg[target_month])
        forecast_months.append(target_month)

    return forecast, forecast_months

def main():
    st.title("Stock Dashboard")

    st.sidebar.header("Ρυθμίσεις")

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Ανέβασε αρχείο Excel", type=['xlsx'])

    if uploaded_file is None:
        st.info("Ανέβασε ένα αρχείο Excel (.xlsx) για να ξεκινήσεις")
        st.markdown("""
        **Απαιτούμενα φύλλα:**
        - `S1P` - με στήλες: Unrestricted Stock, Description, Shelf Life
        - `Cases2021 2022 FC` - ιστορικά δεδομένα
        """)
        return

    if st.sidebar.button("Ανανέωση Δεδομένων"):
        st.cache_data.clear()
        st.rerun()

    try:
        stock_by_desc, df_s1p, s1p_descriptions, df_cases, date_columns, shelf_col, s1p_desc_col = load_all_data(uploaded_file)
    except Exception as e:
        st.error(f"Σφάλμα φόρτωσης: {e}")
        import traceback
        st.code(traceback.format_exc())
        return

    st.sidebar.success(f"Αρχείο: {uploaded_file.name}")

    # --- TABS ---
    tab1, tab2 = st.tabs(["Κίνδυνος OOS", "Προώθηση Πωλήσεων"])

    # ========== TAB 1: OOS RISK ==========
    with tab1:
        st.header("Κίνδυνος Έλλειψης Αποθέματος (OOS)")

        # Build OOS risk data
        oos_data = []
        current_month = datetime.now().month

        for _, row in df_cases.iterrows():
            product_name = str(row['Product_Name']).strip()

            # Get stock from CD (by description)
            stock_match = stock_by_desc[stock_by_desc['Description'].str.strip() == product_name]
            current_stock = stock_match['Stock'].sum() if not stock_match.empty else 0

            # Check if in S1P
            in_s1p = product_name in s1p_descriptions

            # Historical data
            hist_values = row[date_columns].values.astype(float)

            # Seasonal forecast
            forecast, forecast_months = get_seasonal_forecast(hist_values, date_columns, 4)

            # Calculate order needs
            demand_1m = forecast[0] if forecast else 0
            demand_3m = sum(forecast[:3]) if forecast else 0

            order_1m = max(0, demand_1m - current_stock)
            order_3m = max(0, demand_3m - current_stock)

            # System averages
            avg_3m_sys = row.get('Avg_3m_System', np.nan)
            avg_6m_sys = row.get('Avg_6m_System', np.nan)

            # Status
            if current_stock <= 0:
                status = "OOS"
                status_color = "red"
            elif current_stock < demand_1m:
                status = "Κίνδυνος"
                status_color = "orange"
            else:
                status = "OK"
                status_color = "green"

            oos_data.append({
                'Προϊόν': product_name[:50],
                'Απόθεμα': current_stock,
                'Στο S1P': '✓' if in_s1p else '✗ OOS',
                'Πρόβλεψη 1μ': demand_1m,
                'Πρόβλεψη 3μ': demand_3m,
                'Παραγγελία 1μ': order_1m,
                'Παραγγελία 3μ': order_3m,
                'Σύστημα 3μ': avg_3m_sys,
                'Σύστημα 6μ': avg_6m_sys,
                'Κατάσταση': status,
                '_forecast': forecast,
                '_forecast_months': forecast_months,
                '_hist': hist_values,
                '_dates': date_columns
            })

        oos_df = pd.DataFrame(oos_data)

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            show_only_risk = st.checkbox("Μόνο προϊόντα σε κίνδυνο", value=True)
        with col2:
            show_only_not_s1p = st.checkbox("Μόνο εκτός S1P", value=False)

        display_df = oos_df.copy()
        if show_only_risk:
            display_df = display_df[display_df['Κατάσταση'] != 'OK']
        if show_only_not_s1p:
            display_df = display_df[display_df['Στο S1P'] == '✗ OOS']

        # Summary metrics
        st.subheader("Σύνοψη")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Συνολικά Προϊόντα", len(oos_df))
        with col2:
            oos_count = len(oos_df[oos_df['Κατάσταση'] == 'OOS'])
            st.metric("OOS", oos_count, delta_color="inverse")
        with col3:
            risk_count = len(oos_df[oos_df['Κατάσταση'] == 'Κίνδυνος'])
            st.metric("Σε Κίνδυνο", risk_count, delta_color="inverse")
        with col4:
            not_s1p = len(oos_df[oos_df['Στο S1P'] == '✗ OOS'])
            st.metric("Εκτός S1P", not_s1p)

        # Display next 3 months forecast header
        st.subheader(f"Μέση Ζήτηση Επόμενων Μηνών")
        month_names = [MONTH_NAMES_GR[(current_month - 1 + i) % 12] for i in range(4)]
        st.write(f"**{month_names[0]}** → **{month_names[1]}** → **{month_names[2]}** → **{month_names[3]}**")

        # Main table
        st.subheader("Πίνακας Προϊόντων")

        display_cols = ['Προϊόν', 'Απόθεμα', 'Στο S1P', 'Πρόβλεψη 1μ', 'Πρόβλεψη 3μ',
                       'Παραγγελία 1μ', 'Παραγγελία 3μ', 'Σύστημα 3μ', 'Σύστημα 6μ', 'Κατάσταση']

        st.dataframe(
            display_df[display_cols].style.format({
                'Απόθεμα': lambda x: f"{int(x):,}" if pd.notna(x) else "-",
                'Πρόβλεψη 1μ': lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
                'Πρόβλεψη 3μ': lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
                'Παραγγελία 1μ': lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
                'Παραγγελία 3μ': lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
                'Σύστημα 3μ': lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
                'Σύστημα 6μ': lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
            }),
            use_container_width=True,
            height=400
        )

        # Product detail view
        st.subheader("Λεπτομέρειες Προϊόντος")
        product_list = display_df['Προϊόν'].tolist()
        if product_list:
            selected_product = st.selectbox("Επιλογή προϊόντος", product_list)

            if selected_product:
                prod_row = oos_df[oos_df['Προϊόν'] == selected_product].iloc[0]

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Η Πρόβλεψή μας")
                    forecast = prod_row['_forecast']
                    forecast_months = prod_row['_forecast_months']
                    if forecast and forecast_months:
                        for i, (f, m) in enumerate(zip(forecast, forecast_months)):
                            st.metric(MONTH_NAMES_GR[m-1], f"{f:,.0f} κιβ.")

                    st.markdown("### Προτεινόμενη Παραγγελία")
                    st.metric("Για 1 μήνα", f"{prod_row['Παραγγελία 1μ']:,.0f} κιβ.")
                    st.metric("Για 3 μήνες", f"{prod_row['Παραγγελία 3μ']:,.0f} κιβ.")

                with col2:
                    st.markdown("### Το Σύστημα Λέει")
                    st.metric("Μ.Ο. 3 μηνών", f"{prod_row['Σύστημα 3μ']:,.0f}" if pd.notna(prod_row['Σύστημα 3μ']) else "-")
                    st.metric("Μ.Ο. 6 μηνών", f"{prod_row['Σύστημα 6μ']:,.0f}" if pd.notna(prod_row['Σύστημα 6μ']) else "-")

                    st.markdown("### Τρέχον Απόθεμα")
                    st.metric("Διαθέσιμο", f"{prod_row['Απόθεμα']:,} κιβ.")

                # Monthly consumption chart
                st.subheader("Κατανάλωση ανά Μήνα")
                hist_values = prod_row['_hist']
                dates = prod_row['_dates']

                hist_df = pd.DataFrame({
                    'Μήνας': pd.to_datetime([f"{d}-01" for d in dates]),
                    'Κατανάλωση': hist_values
                }).dropna()

                if len(hist_df) > 0:
                    fig = px.bar(hist_df, x='Μήνας', y='Κατανάλωση',
                                title='Μηνιαία Κατανάλωση')
                    fig.update_layout(xaxis_title="Μήνας", yaxis_title="Κιβώτια")
                    st.plotly_chart(fig, use_container_width=True)

    # ========== TAB 2: PROMOTE SALES ==========
    with tab2:
        st.header("Προώθηση Πωλήσεων - Προϊόντα προς Λήξη")

        if shelf_col is None:
            st.warning("Δεν βρέθηκε στήλη ημερομηνίας λήξης στο S1P")
            st.info("Αναζητούμενες στήλες: 'shelf', 'λήξη', 'expir'")
        else:
            # Get products expiring within 1 month with stock >= 3
            today = datetime.now()
            one_month_later = today + timedelta(days=30)

            expiring_products = []

            for _, row in df_s1p.iterrows():
                try:
                    desc = str(row[s1p_desc_col]).strip()
                    expiry = row[shelf_col]

                    # Parse expiry date
                    if pd.isna(expiry):
                        continue

                    if isinstance(expiry, str):
                        try:
                            expiry_date = pd.to_datetime(expiry)
                        except:
                            continue
                    else:
                        expiry_date = pd.to_datetime(expiry)

                    # Check if expiring within 1 month
                    if expiry_date <= one_month_later:
                        # Get stock from CD
                        stock_match = stock_by_desc[stock_by_desc['Description'].str.strip() == desc]
                        stock = stock_match['Stock'].sum() if not stock_match.empty else 0

                        if stock >= 3:
                            days_until = (expiry_date - today).days
                            expiring_products.append({
                                'Προϊόν': desc[:50],
                                'Απόθεμα': stock,
                                'Ημ. Λήξης': expiry_date.strftime('%d/%m/%Y'),
                                'Ημέρες': days_until,
                                'Επείγον': 'ΝΑΙ' if days_until <= 7 else 'ΟΧΙ'
                            })
                except Exception as e:
                    continue

            if expiring_products:
                exp_df = pd.DataFrame(expiring_products)
                exp_df = exp_df.sort_values('Ημέρες')

                # Summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Προϊόντα προς Λήξη", len(exp_df))
                with col2:
                    urgent = len(exp_df[exp_df['Επείγον'] == 'ΝΑΙ'])
                    st.metric("Επείγοντα (<7 ημ.)", urgent, delta_color="inverse")
                with col3:
                    total_stock = exp_df['Απόθεμα'].sum()
                    st.metric("Συνολικό Απόθεμα", f"{int(total_stock):,}")

                st.subheader("Λίστα Προϊόντων")
                st.dataframe(
                    exp_df.style.format({
                        'Απόθεμα': lambda x: f"{int(x):,}"
                    }),
                    use_container_width=True,
                    height=400
                )

                # Chart
                st.subheader("Απόθεμα ανά Προϊόν")
                fig = px.bar(exp_df.head(20), x='Προϊόν', y='Απόθεμα',
                            color='Επείγον',
                            color_discrete_map={'ΝΑΙ': 'red', 'ΟΧΙ': 'orange'},
                            title='Top 20 Προϊόντα προς Λήξη')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("Δεν υπάρχουν προϊόντα προς λήξη με απόθεμα >= 3 κιβώτια")

    st.caption("v5.0 | Κίνδυνος OOS + Προώθηση Πωλήσεων")

if __name__ == "__main__":
    main()

