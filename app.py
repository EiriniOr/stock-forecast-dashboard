import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Πρόβλεψη Αποθέματος", page_icon="📊", layout="wide")

# Greek translations
TRANSLATIONS = {
    'title': 'Σύστημα Διαχείρισης Αποθέματος',
    'upload': 'Μεταφόρτωση αρχείου Excel',
    'upload_help': 'Επιλέξτε το αρχείο Excel με τα δεδομένα αποθέματος',
    'tab1': 'Πρόβλεψη Αποθέματος',
    'tab2': 'Προϊόντα προς Προώθηση',
    'select_product': 'Επιλέξτε Προϊόν',
    'current_stock': 'Τρέχον Απόθεμα',
    'order_30days': 'Παραγγελία για 30 ημέρες',
    'order_3months': 'Παραγγελία για 3 μήνες',
    'sales_per_month': 'Πωλήσεις ανά Μήνα',
    'present_system': 'Τρέχον Σύστημα',
    'status': 'Κατάσταση',
    'normal': 'Κανονική',
    'scheduled_delivery': 'Προγραμματισμένη Παράδοση',
    'expected_cases': 'Αναμενόμενες Κιβώτια',
    'avg_3month': 'Μέσος Όρος 3μηνου',
    'avg_6month': 'Μέσος Όρος 6μηνου',
    'product': 'Προϊόν',
    'stock': 'Απόθεμα',
    'expiration': 'Λήξη',
    'push_desc': 'Προϊόντα με απόθεμα >10 και λήξη εντός 6 μηνών',
    'no_products': 'Δεν βρέθηκαν προϊόντα προς προώθηση',
    'order_now': 'ΠΑΡΑΓΓΕΙΛΕ ΤΩΡΑ',
    'units': 'μονάδες'
}


def load_excel_data(uploaded_file):
    """Load and parse the Excel file."""
    xlsx = pd.ExcelFile(uploaded_file)

    # Load CD sheet (header at row 1, data starts row 2)
    df_cd_raw = pd.read_excel(xlsx, sheet_name='CD', header=None)
    # Find header row
    df_cd = pd.DataFrame()
    df_cd['category'] = df_cd_raw.iloc[2:, 0].values
    df_cd['trade'] = df_cd_raw.iloc[2:, 1].values
    df_cd['mrdr'] = df_cd_raw.iloc[2:, 2].values
    df_cd['description'] = df_cd_raw.iloc[2:, 3].values
    df_cd['status'] = df_cd_raw.iloc[2:, 6].values
    df_cd['scheduled_delivery'] = df_cd_raw.iloc[2:, 7].values
    df_cd['expected_cases'] = df_cd_raw.iloc[2:, 8].values
    df_cd = df_cd.dropna(subset=['mrdr'])
    df_cd['mrdr'] = df_cd['mrdr'].astype(str).str.strip()
    # Remove header row if present
    df_cd = df_cd[df_cd['mrdr'] != 'MRDR']

    # Load S1P sheet
    df_s1p = pd.read_excel(xlsx, sheet_name='S1P')
    df_s1p['Material'] = df_s1p['Material'].astype(str).str.strip()

    # Load Cases sheet for sales history
    df_cases_raw = pd.read_excel(xlsx, sheet_name='Cases2021 2022 FC', header=None)

    # Parse Cases sheet structure
    # Row 7: Year headers (2024, 2025)
    # Row 8: Month numbers (1-12)
    # Row 9: Column names (Category, SKU Code, Product description...)
    # Data starts at row 10

    years_row = df_cases_raw.iloc[8, 3:26].values  # columns 3-25
    months_row = df_cases_raw.iloc[9, 3:26].values  # columns 3-25

    sales_data = []
    for idx in range(10, len(df_cases_raw)):
        row = df_cases_raw.iloc[idx]
        mrdr = str(row[28]).strip() if pd.notna(row[28]) else None  # Column AB (28)
        if mrdr and mrdr != 'nan':
            product_desc = row[2]  # Column C - Product description
            avg_3m = row[29] if pd.notna(row[29]) else 0  # Column AD
            avg_6m = row[30] if pd.notna(row[30]) else 0  # Column AE

            # Monthly sales data
            monthly_sales = []
            for i, (year, month) in enumerate(zip(years_row, months_row)):
                if pd.notna(year) and pd.notna(month):
                    value = row[3 + i]
                    if pd.notna(value):
                        monthly_sales.append({
                            'year': int(year),
                            'month': int(month),
                            'sales': float(value) if value != '' else 0
                        })

            sales_data.append({
                'mrdr': mrdr,
                'product_desc_cases': product_desc,
                'avg_3m': avg_3m,
                'avg_6m': avg_6m,
                'monthly_sales': monthly_sales
            })

    df_cases = pd.DataFrame(sales_data)

    return df_cd, df_s1p, df_cases


def get_oos_products(df_cd, df_s1p):
    """Get products marked as OOS or OOS Risk that exist in both CD and S1P."""
    # Filter CD to only OOS and OOS Risk status
    oos_cd = df_cd[df_cd['status'].isin(['OOS', 'OOS Risk'])].copy()

    cd_mrdrs = set(oos_cd['mrdr'].dropna().unique())
    s1p_materials = set(df_s1p['Material'].dropna().unique())
    common = cd_mrdrs.intersection(s1p_materials)
    return oos_cd[oos_cd['mrdr'].isin(common)].copy()


def calculate_stock_by_mrdr(df_s1p, mrdr):
    """Sum unrestricted stock for all items with the same MRDR."""
    filtered = df_s1p[df_s1p['Material'] == mrdr]
    return filtered['Unrestricted stock'].sum()


def get_sales_chart_data(df_cases, mrdr):
    """Get sales data for chart by MRDR."""
    if mrdr not in df_cases['mrdr'].values:
        return None

    row = df_cases[df_cases['mrdr'] == mrdr].iloc[0]
    return row['monthly_sales']


def calculate_order_quantity(sales_data, current_stock, months=1):
    """Calculate recommended order quantity for lean production."""
    if not sales_data:
        return 0

    # Calculate average monthly sales from positive values
    relevant_sales = [e['sales'] for e in sales_data if e['sales'] > 0]

    if not relevant_sales:
        return 0

    avg_monthly_sales = np.mean(relevant_sales)

    # Calculate needed for the period
    needed = avg_monthly_sales * months

    # Subtract current stock
    order_qty = max(0, needed - current_stock)

    return int(np.ceil(order_qty))


def get_expiring_products(df_s1p, df_cd):
    """Get products with >10 stock and shelf life within 6 months from now."""
    now = datetime.now()
    six_months_later = now + timedelta(days=180)

    # Convert shelf life date column to datetime
    df_s1p_copy = df_s1p.copy()
    df_s1p_copy['Short shelf life date'] = pd.to_datetime(
        df_s1p_copy['Short shelf life date'], errors='coerce'
    )

    # Aggregate stock and get earliest shelf date by Material, keep Description
    agg_data = df_s1p_copy.groupby('Material').agg({
        'Unrestricted stock': 'sum',
        'Short shelf life date': 'min',
        'Description': 'first'
    }).reset_index()

    # Filter: stock > 10, shelf life between now and 6 months from now
    expiring = []
    for _, row in agg_data.iterrows():
        material = row['Material']
        stock = row['Unrestricted stock']
        shelf_date = row['Short shelf life date']
        desc = row['Description']

        if stock > 10 and pd.notna(shelf_date):
            try:
                # Only include if shelf date is in the future and within 6 months
                if now <= shelf_date <= six_months_later:
                    # Try to get Greek description from CD, fallback to S1P Description
                    cd_match = df_cd[df_cd['mrdr'] == material]
                    if not cd_match.empty:
                        desc = cd_match.iloc[0]['description']

                    expiring.append({
                        'description': desc,
                        'stock': int(stock),
                        'expiration': shelf_date.strftime('%Y-%m-%d')
                    })
            except:
                pass

    # Sort by expiration date
    expiring.sort(key=lambda x: x['expiration'])
    return expiring


def create_sales_chart(sales_data, current_stock, mrdr):
    """Create a plotly chart of sales per month by year."""
    if not sales_data:
        return None

    fig = go.Figure()

    # Group by year
    years_data = {}
    for entry in sales_data:
        year = entry['year']
        if year not in years_data:
            years_data[year] = {'months': [], 'sales': []}
        years_data[year]['months'].append(entry['month'])
        years_data[year]['sales'].append(max(0, entry['sales']))

    colors = {'2024': '#3498db', '2025': '#e74c3c', '2026': '#2ecc71'}

    for year in sorted(years_data.keys()):
        data = years_data[year]
        # Sort by month
        sorted_data = sorted(zip(data['months'], data['sales']))
        months, sales = zip(*sorted_data) if sorted_data else ([], [])

        fig.add_trace(go.Scatter(
            x=list(months),
            y=list(sales),
            mode='lines+markers',
            name=str(year),
            line=dict(color=colors.get(str(year), '#95a5a6'), width=2),
            marker=dict(size=8)
        ))

    # Add current stock as a star in the current month
    current_month = datetime.now().month
    fig.add_trace(go.Scatter(
        x=[current_month],
        y=[current_stock],
        mode='markers',
        name='Τρέχον Απόθεμα',
        marker=dict(
            symbol='star',
            size=20,
            color='#f1c40f',
            line=dict(color='#000000', width=2)
        )
    ))

    month_names = ['Ιαν', 'Φεβ', 'Μαρ', 'Απρ', 'Μάι', 'Ιουν',
                   'Ιουλ', 'Αυγ', 'Σεπ', 'Οκτ', 'Νοε', 'Δεκ']

    fig.update_layout(
        title=TRANSLATIONS['sales_per_month'],
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=month_names,
            title='Μήνας'
        ),
        yaxis=dict(title='Πωλήσεις (μονάδες)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=400
    )

    return fig


def main():
    st.title(f"📊 {TRANSLATIONS['title']}")

    # Landing page - File upload
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

    uploaded_file = st.file_uploader(
        TRANSLATIONS['upload'],
        type=['xlsx', 'xls'],
        help=TRANSLATIONS['upload_help']
    )

    if uploaded_file is not None:
        try:
            with st.spinner('Φόρτωση δεδομένων...'):
                df_cd, df_s1p, df_cases = load_excel_data(uploaded_file)
                st.session_state.df_cd = df_cd
                st.session_state.df_s1p = df_s1p
                st.session_state.df_cases = df_cases
                st.session_state.data_loaded = True
        except Exception as e:
            st.error(f'Σφάλμα φόρτωσης: {str(e)}')
            st.session_state.data_loaded = False

    if st.session_state.data_loaded:
        df_cd = st.session_state.df_cd
        df_s1p = st.session_state.df_s1p
        df_cases = st.session_state.df_cases

        # Create tabs
        tab1, tab2 = st.tabs([f"📈 {TRANSLATIONS['tab1']}", f"⚠️ {TRANSLATIONS['tab2']}"])

        # Tab 1: Stock Prediction
        with tab1:
            # Get OOS/OOS Risk products only
            oos_products = get_oos_products(df_cd, df_s1p)

            if oos_products.empty:
                st.warning("Δεν βρέθηκαν προϊόντα με κατάσταση OOS ή OOS Risk")
            else:
                # Product selector
                product_options = oos_products[['mrdr', 'description']].drop_duplicates()
                product_options['display'] = product_options['description'].astype(str)

                selected = st.selectbox(
                    TRANSLATIONS['select_product'],
                    options=product_options['mrdr'].tolist(),
                    format_func=lambda x: product_options[product_options['mrdr'] == x]['display'].values[0]
                )

                if selected:
                    # Get product info
                    product_info = df_cd[df_cd['mrdr'] == selected].iloc[0]
                    current_stock = calculate_stock_by_mrdr(df_s1p, selected)
                    sales_data = get_sales_chart_data(df_cases, selected)

                    # Calculate order quantities (months)
                    order_1m = calculate_order_quantity(sales_data, current_stock, 1)
                    order_3m = calculate_order_quantity(sales_data, current_stock, 3)

                    # Two columns: Left (predictions), Right (present system)
                    col_left, col_right = st.columns([2, 1])

                    with col_left:
                        st.subheader(f"📦 {product_info['description']}")

                        # Order recommendations at top, smaller
                        col_o1, col_o2, col_o3 = st.columns([1, 1, 1])
                        with col_o1:
                            st.metric(TRANSLATIONS['current_stock'], f"{current_stock:,}")
                        with col_o2:
                            st.metric(TRANSLATIONS['order_30days'], f"{order_1m:,}")
                        with col_o3:
                            st.metric(TRANSLATIONS['order_3months'], f"{order_3m:,}")

                        # Sales chart
                        if sales_data:
                            fig = create_sales_chart(sales_data, current_stock, selected)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Δεν υπάρχουν δεδομένα πωλήσεων για αυτό το προϊόν")

                    with col_right:
                        st.subheader(f"📋 {TRANSLATIONS['present_system']}")

                        # Status
                        status = product_info['status']
                        if pd.isna(status) or status == '' or str(status).strip() == '':
                            status = TRANSLATIONS['normal']
                        st.metric(TRANSLATIONS['status'], status)

                        # Scheduled delivery
                        sched_del = product_info['scheduled_delivery']
                        if pd.notna(sched_del):
                            if hasattr(sched_del, 'strftime'):
                                sched_del = sched_del.strftime('%Y-%m-%d')
                        else:
                            sched_del = 'TBC'
                        st.metric(TRANSLATIONS['scheduled_delivery'], sched_del)

                        # Expected cases
                        exp_cases = product_info['expected_cases']
                        if pd.isna(exp_cases):
                            exp_cases = 0
                        st.metric(TRANSLATIONS['expected_cases'], f"{int(exp_cases):,}")

                        # Averages from Cases sheet
                        cases_match = df_cases[df_cases['mrdr'] == selected]
                        if not cases_match.empty:
                            avg_3m = cases_match.iloc[0]['avg_3m']
                            avg_6m = cases_match.iloc[0]['avg_6m']

                            if pd.notna(avg_3m):
                                st.metric(TRANSLATIONS['avg_3month'], f"{int(avg_3m):,}")
                            else:
                                st.metric(TRANSLATIONS['avg_3month'], "N/A")

                            if pd.notna(avg_6m):
                                st.metric(TRANSLATIONS['avg_6month'], f"{int(avg_6m):,}")
                            else:
                                st.metric(TRANSLATIONS['avg_6month'], "N/A")
                        else:
                            st.metric(TRANSLATIONS['avg_3month'], "N/A")
                            st.metric(TRANSLATIONS['avg_6month'], "N/A")

        # Tab 2: Products to Push
        with tab2:
            st.subheader(f"⚠️ {TRANSLATIONS['tab2']}")
            st.caption(TRANSLATIONS['push_desc'])

            expiring = get_expiring_products(df_s1p, df_cd)

            if expiring:
                df_expiring = pd.DataFrame(expiring)
                df_expiring.columns = [TRANSLATIONS['product'], TRANSLATIONS['stock'], TRANSLATIONS['expiration']]

                st.dataframe(
                    df_expiring,
                    use_container_width=True,
                    hide_index=True
                )

                st.info(f"Βρέθηκαν {len(expiring)} προϊόντα προς προώθηση")
            else:
                st.success(TRANSLATIONS['no_products'])


if __name__ == '__main__':
    main()
