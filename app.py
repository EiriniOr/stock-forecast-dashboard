"""
Stock Forecasting Dashboard - Cloud Version
Πίνακας Πρόβλεψης Αποθέματος
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Πρόβλεψη Αποθέματος", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_data(uploaded_file):
    """Parse uploaded Excel file - data stays in memory only"""

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
    data = pd.Series(historical_data).dropna()
    if len(data) < 6:
        return None, None, None

    X = np.arange(len(data)).reshape(-1, 1)
    y = data.values

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(data), len(data) + periods_ahead).reshape(-1, 1)
    forecast = model.predict(future_X)

    fitted = model.predict(X)
    residuals = y - fitted
    rmse = np.sqrt(np.mean(residuals**2))

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

        start_month = len(data) % 12
        for i in range(len(forecast)):
            month_idx = (start_month + i) % 12
            forecast[i] *= seasonal_factors[month_idx]

    forecast = np.maximum(forecast, 0)
    return forecast, rmse, model

def calculate_safety_stock(historical_data, service_level=0.95):
    data = pd.Series(historical_data).dropna()
    if len(data) < 3:
        return 0

    z_scores = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
    z = z_scores.get(service_level, 1.65)

    std_dev = data.std()
    lead_time = 1

    safety_stock = z * std_dev * np.sqrt(lead_time)
    return max(0, safety_stock)

# --- MAIN APP ---
def main():
    st.title("Πίνακας Πρόβλεψης Αποθέματος")

    # Sidebar
    st.sidebar.header("Φόρτωση Δεδομένων")
    st.sidebar.info("Τα δεδομένα παραμένουν ιδιωτικά - επεξεργασία μόνο στη μνήμη")

    uploaded_file = st.sidebar.file_uploader(
        "Ανέβασε αρχείο Excel (.xlsx)",
        type=['xlsx'],
        help="Αρχείο με sheets 'Cases2021 2022 FC' και 'Stock'"
    )

    if uploaded_file is None:
        st.info("Ανέβασε το αρχείο Excel στην πλαϊνή μπάρα για να ξεκινήσεις")
        st.markdown("""
        ### Απαιτούμενη δομή Excel:
        - **Sheet 'Cases2021 2022 FC'**: Μηνιαία δεδομένα ζήτησης
        - **Sheet 'Stock'**: Τρέχοντα επίπεδα αποθέματος

        ### Ασφάλεια Δεδομένων:
        - Επεξεργασία μόνο στη μνήμη
        - Τίποτα δεν αποθηκεύεται στον server
        - Τα δεδομένα διαγράφονται όταν κλείσει η σελίδα
        """)
        return

    try:
        data_df, stock_df, date_columns = load_data(uploaded_file)
        st.sidebar.success(f"Φορτώθηκαν {len(data_df)} προϊόντα")
    except Exception as e:
        st.error(f"Σφάλμα φόρτωσης: {e}")
        st.info("Έλεγξε ότι το αρχείο έχει τα απαιτούμενα sheets")
        return

    if st.sidebar.button("Επαναφόρτωση Δεδομένων"):
        st.cache_data.clear()
        st.rerun()

    # Product selection
    st.sidebar.header("Επιλογή Προϊόντος")

    categories = ['Όλα'] + sorted(data_df['Category'].dropna().unique().tolist())
    selected_category = st.sidebar.selectbox("Κατηγορία", categories)

    if selected_category != 'Όλα':
        filtered_df = data_df[data_df['Category'] == selected_category]
    else:
        filtered_df = data_df

    product_options = filtered_df.apply(
        lambda x: f"{x['SKU_Short']} - {str(x['Product_Name'])[:40]}", axis=1
    ).tolist()

    if not product_options:
        st.warning("Δεν υπάρχουν προϊόντα στην επιλεγμένη κατηγορία")
        return

    selected_product = st.sidebar.selectbox("Επιλογή Προϊόντος", product_options)

    selected_idx = product_options.index(selected_product)
    product_row = filtered_df.iloc[selected_idx]

    # Forecast settings
    st.sidebar.header("Ρυθμίσεις Πρόβλεψης")
    forecast_months = st.sidebar.slider("Ορίζοντας Πρόβλεψης (μήνες)", 1, 24, 12)
    service_level = st.sidebar.selectbox("Service Level", [0.90, 0.95, 0.99], index=1)

    # --- MAIN CONTENT ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Κωδικός SKU", product_row['SKU_Short'])
    with col2:
        st.metric("Κατηγορία", product_row['Category'])
    with col3:
        sku_short = product_row['SKU_Short']
        stock_match = stock_df[stock_df['Material Code'].str.contains(sku_short, na=False)]
        if not stock_match.empty:
            current_stock = stock_match.iloc[0]['Unrestricted Stock']
            st.metric("Τρέχον Απόθεμα", f"{int(current_stock):,}")
        else:
            st.metric("Τρέχον Απόθεμα", "Μ/Δ")

    st.subheader(product_row['Product_Name'])

    historical = product_row[date_columns].values.astype(float)
    dates = pd.to_datetime([f"{d}-01" for d in date_columns])

    hist_df = pd.DataFrame({
        'Date': dates,
        'Demand': historical
    })
    hist_df = hist_df.dropna()

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["Ιστορική Ανάλυση", "Πρόβλεψη", "Ανάλυση Αποθέματος", "Σύγκριση"])

    with tab1:
        st.subheader("Ιστορική Ζήτηση")

        if len(hist_df) > 0:
            fig = px.line(hist_df, x='Date', y='Demand',
                         title='Μηνιαία Ζήτηση',
                         markers=True)
            fig.update_layout(xaxis_title="Μήνας", yaxis_title="Κιβώτια")
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Μέση Μηνιαία Ζήτηση", f"{hist_df['Demand'].mean():,.0f}")
            with col2:
                st.metric("Μέγιστη Ζήτηση", f"{hist_df['Demand'].max():,.0f}")
            with col3:
                st.metric("Ελάχιστη Ζήτηση", f"{hist_df['Demand'].min():,.0f}")
            with col4:
                st.metric("Τυπική Απόκλιση", f"{hist_df['Demand'].std():,.0f}")

            st.subheader("Σύγκριση Ετών")
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
                title='Ζήτηση ανά Μήνα (Σύγκριση Ετών)',
                xaxis_title='Μήνας',
                yaxis_title='Κιβώτια',
                xaxis=dict(tickmode='array', tickvals=list(range(1,13)),
                          ticktext=['Ιαν','Φεβ','Μαρ','Απρ','Μάι','Ιουν','Ιουλ','Αυγ','Σεπ','Οκτ','Νοε','Δεκ'])
            )
            st.plotly_chart(fig_yoy, use_container_width=True)

            st.subheader("Μηνιαία Σύνοψη")
            st.dataframe(pivot.round(0), use_container_width=True)
        else:
            st.warning("Δεν υπάρχουν ιστορικά δεδομένα")

    with tab2:
        st.subheader("Πρόβλεψη Ζήτησης")

        if len(hist_df) >= 6:
            forecast, rmse, model = forecast_demand(hist_df['Demand'].values, forecast_months)

            if forecast is not None:
                last_date = hist_df['Date'].max()
                forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                              periods=forecast_months, freq='MS')

                forecast_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Forecast': forecast
                })

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=hist_df['Date'], y=hist_df['Demand'],
                    name='Ιστορικό', mode='lines+markers',
                    line=dict(color='blue')
                ))

                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'], y=forecast_df['Forecast'],
                    name='Πρόβλεψη', mode='lines+markers',
                    line=dict(color='red', dash='dash')
                ))

                upper = forecast_df['Forecast'] + 1.96 * rmse
                lower = np.maximum(forecast_df['Forecast'] - 1.96 * rmse, 0)

                fig.add_trace(go.Scatter(
                    x=list(forecast_df['Date']) + list(forecast_df['Date'][::-1]),
                    y=list(upper) + list(lower[::-1]),
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.1)',
                    line=dict(color='rgba(255,0,0,0)'),
                    name='95% Διάστημα Εμπιστοσύνης'
                ))

                fig.update_layout(
                    title=f'Πρόβλεψη Ζήτησης {forecast_months} Μηνών',
                    xaxis_title='Ημερομηνία',
                    yaxis_title='Κιβώτια'
                )
                st.plotly_chart(fig, use_container_width=True)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Πρόβλεψη Επόμενου Μήνα", f"{forecast[0]:,.0f}")
                with col2:
                    st.metric(f"Σύνολο {forecast_months}μ", f"{forecast.sum():,.0f}")
                with col3:
                    st.metric("RMSE Πρόβλεψης", f"{rmse:,.0f}")
                with col4:
                    safety = calculate_safety_stock(hist_df['Demand'].values, service_level)
                    st.metric("Safety Stock", f"{safety:,.0f}")

                st.subheader("Λεπτομέρειες Πρόβλεψης")
                forecast_df['Μήνας'] = forecast_df['Date'].dt.strftime('%Y-%m')
                forecast_df['Πρόβλεψη'] = forecast_df['Forecast'].round(0).astype(int)
                st.dataframe(forecast_df[['Μήνας', 'Πρόβλεψη']], use_container_width=True)
            else:
                st.warning("Αδυναμία δημιουργίας πρόβλεψης")
        else:
            st.warning("Απαιτούνται τουλάχιστον 6 μήνες δεδομένων")

    with tab3:
        st.subheader("Ανάλυση Θέσης Αποθέματος")

        if not stock_match.empty:
            stock_row = stock_match.iloc[0]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Τρέχοντα Επίπεδα Αποθέματος")
                st.metric("Διαθέσιμο Απόθεμα", f"{int(stock_row['Unrestricted Stock']):,}")
                st.metric("Καθαρή Διαθεσιμότητα", f"{int(stock_row['Available Quantity']):,}")
                st.metric("Ποιοτικός Έλεγχος", f"{int(stock_row['Quality Inspection Stock']):,}")

            with col2:
                st.markdown("### Παραγγελίες & Παραδόσεις")
                st.metric("Παραγγελίες Πελατών", f"{int(stock_row['Cust.Ord.Total Qty']):,}")
                st.metric("Επιβεβαιωμένες Παραγγελίες", f"{int(stock_row['Cust.Conf.Total Qty']):,}")
                st.metric("Σε Παράδοση", f"{int(stock_row['Quantities on Deliveries']):,}")
                st.metric("Παραλαβές", f"{int(stock_row['Goods Receipts Total Qty']):,}")

            if len(hist_df) > 0:
                avg_monthly = hist_df['Demand'].mean()
                current = stock_row['Unrestricted Stock']

                if avg_monthly > 0:
                    months_of_stock = current / avg_monthly

                    st.markdown("### Κάλυψη Αποθέματος")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Μέση Μηνιαία Ζήτηση", f"{avg_monthly:,.0f}")
                    with col2:
                        st.metric("Μήνες Αποθέματος", f"{months_of_stock:.1f}")
                    with col3:
                        safety = calculate_safety_stock(hist_df['Demand'].values, service_level)
                        reorder_point = avg_monthly + safety
                        st.metric("Σημείο Αναπαραγγελίας", f"{reorder_point:,.0f}")

                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=months_of_stock,
                        title={'text': "Μήνες Κάλυψης Αποθέματος"},
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
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Δεν βρέθηκαν δεδομένα αποθέματος για αυτό το προϊόν")

    with tab4:
        st.subheader("Σύγκριση Προϊόντων")

        compare_products = st.multiselect(
            "Επιλογή προϊόντων για σύγκριση",
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
                title='Σύγκριση Ζήτησης Προϊόντων',
                xaxis_title='Ημερομηνία',
                yaxis_title='Κιβώτια'
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- SUMMARY TABLE ---
    st.header("Σύνοψη Όλων των Προϊόντων")

    summary_data = []
    for _, row in data_df.iterrows():
        hist = pd.Series(row[date_columns].values.astype(float)).dropna()

        sku = row['SKU_Short']
        stock_match_row = stock_df[stock_df['Material Code'].str.contains(sku, na=False)]
        current_stock = stock_match_row.iloc[0]['Unrestricted Stock'] if not stock_match_row.empty else None

        if len(hist) >= 6:
            fc, _, _ = forecast_demand(hist.values, 3)
            next_3m = fc.sum() if fc is not None else None
        else:
            next_3m = None

        avg_demand = hist.mean() if len(hist) > 0 else None
        latest = hist.iloc[-1] if len(hist) > 0 else None

        summary_data.append({
            'SKU': sku,
            'Προϊόν': str(row['Product_Name'])[:35],
            'Κατηγορία': row['Category'],
            'Τρέχον Απόθεμα': current_stock,
            'Μέση Μηνιαία': avg_demand,
            'Τελευταίος Μήνας': latest,
            'Πρόβλεψη 3μ': next_3m,
            'Μήνες Κάλυψης': current_stock / avg_demand if current_stock and avg_demand and avg_demand > 0 else None
        })

    summary_df = pd.DataFrame(summary_data)

    st.dataframe(
        summary_df.style.format({
            'Τρέχον Απόθεμα': lambda x: f"{int(x):,}" if pd.notna(x) else "-",
            'Μέση Μηνιαία': lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
            'Τελευταίος Μήνας': lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
            'Πρόβλεψη 3μ': lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
            'Μήνες Κάλυψης': lambda x: f"{x:.1f}" if pd.notna(x) else "-"
        }),
        use_container_width=True,
        height=400
    )

if __name__ == "__main__":
    main()
