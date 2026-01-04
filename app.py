"""
Stock Dashboard - Historical Data + Manual Estimation
Πίνακας Αποθέματος - Ιστορικά Δεδομένα + Χειροκίνητη Εκτίμηση
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Πίνακας Αποθέματος", layout="wide")

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

    # 2. S1P sheet - current stock
    uploaded_file.seek(0)
    df_s1p = pd.read_excel(uploaded_file, sheet_name='S1P')
    stock_by_material = df_s1p.groupby('Material').agg({
        'Unrestricted stock': 'sum',
        'Description': 'first',
        'Category': 'first'
    }).reset_index()
    stock_by_material['Material'] = stock_by_material['Material'].astype(str)

    # 3. SOP sheet - system predictions
    uploaded_file.seek(0)
    df_sop = pd.read_excel(uploaded_file, sheet_name='SOP', header=2)
    df_sop.columns = ['Category_SOP', 'Trade', 'MRDR', 'Product_Desc', 'Link', 'Status',
                      'Avg_Daily_Sales', 'Stock_Days', 'Days_Till_Avail', 'Available_Stock'] + \
                     [f'extra_{i}' for i in range(len(df_sop.columns) - 10)]
    df_sop['MRDR'] = df_sop['MRDR'].astype(str).str.strip()

    # Aggregate by Product_Name
    agg_dict = {col: 'sum' for col in date_columns}
    agg_dict['Category'] = 'first'
    agg_dict['Material'] = lambda x: list(x)

    grouped_df = data_df.groupby('Product_Name').agg(agg_dict).reset_index()
    grouped_df['Material_Codes'] = grouped_df['Material']
    grouped_df['Material'] = grouped_df['Material_Codes'].apply(lambda x: x[0] if x else '')
    grouped_df['Num_Codes'] = grouped_df['Material_Codes'].apply(len)

    return grouped_df, stock_by_material, df_sop, date_columns, data_df


def get_total_stock(product_row, stock_df):
    """Get total stock across all material codes"""
    material_codes = product_row.get('Material_Codes', [product_row.get('Material', '')])
    if isinstance(material_codes, str):
        material_codes = [material_codes]

    total = 0
    for mat in material_codes:
        match = stock_df[stock_df['Material'] == str(mat)]
        if not match.empty:
            total += int(match['Unrestricted stock'].values[0])
    return total


def get_system_data(product_row, sop_df):
    """Get system daily sales and OOS status"""
    material_codes = product_row.get('Material_Codes', [product_row.get('Material', '')])
    if isinstance(material_codes, str):
        material_codes = [material_codes]

    total_daily = 0
    has_oos = False

    for mat in material_codes:
        match = sop_df[sop_df['MRDR'].str.contains(str(mat), na=False)]
        if not match.empty:
            daily = match['Avg_Daily_Sales'].values[0]
            if pd.notna(daily):
                total_daily += daily
            if match['Status'].values[0] == "OOS":
                has_oos = True

    return total_daily if total_daily > 0 else None, has_oos


def calculate_historical_stats(historical_values, date_columns):
    """Calculate historical statistics for decision making"""
    current_month = datetime.now().month
    current_year = datetime.now().year

    # Parse dates
    dates = pd.to_datetime([f"{d}-01" for d in date_columns])
    df = pd.DataFrame({'date': dates, 'sales': historical_values})
    df = df.dropna()
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    # Only positive sales
    positive = df[df['sales'] > 0]

    if len(positive) == 0:
        return None

    stats = {}

    # Overall average (monthly, converted to daily)
    stats['avg_monthly'] = positive['sales'].mean()
    stats['avg_daily'] = stats['avg_monthly'] / 30

    # Last 3 months average
    last_3 = positive.tail(3)
    if len(last_3) > 0:
        stats['last_3m_avg_monthly'] = last_3['sales'].mean()
        stats['last_3m_avg_daily'] = stats['last_3m_avg_monthly'] / 30
    else:
        stats['last_3m_avg_monthly'] = stats['avg_monthly']
        stats['last_3m_avg_daily'] = stats['avg_daily']

    # Same month last year
    same_month_ly = positive[(positive['month'] == current_month) & (positive['year'] == current_year - 1)]
    if len(same_month_ly) > 0:
        stats['same_month_ly'] = same_month_ly['sales'].values[0]
        stats['same_month_ly_daily'] = stats['same_month_ly'] / 30
    else:
        stats['same_month_ly'] = None
        stats['same_month_ly_daily'] = None

    # Last month
    last_month = positive.iloc[-1] if len(positive) > 0 else None
    if last_month is not None:
        stats['last_month'] = last_month['sales']
        stats['last_month_daily'] = stats['last_month'] / 30
    else:
        stats['last_month'] = None
        stats['last_month_daily'] = None

    # Trend (compare last 3 months to previous 3 months)
    if len(positive) >= 6:
        recent = positive.tail(3)['sales'].mean()
        previous = positive.iloc[-6:-3]['sales'].mean()
        if previous > 0:
            trend_pct = ((recent - previous) / previous) * 100
            if trend_pct > 10:
                stats['trend'] = 'up'
                stats['trend_pct'] = trend_pct
            elif trend_pct < -10:
                stats['trend'] = 'down'
                stats['trend_pct'] = trend_pct
            else:
                stats['trend'] = 'stable'
                stats['trend_pct'] = trend_pct
        else:
            stats['trend'] = 'stable'
            stats['trend_pct'] = 0
    else:
        stats['trend'] = 'unknown'
        stats['trend_pct'] = 0

    # Volatility (coefficient of variation)
    if len(positive) >= 3:
        stats['volatility'] = (positive['sales'].std() / positive['sales'].mean()) * 100
    else:
        stats['volatility'] = 0

    return stats


# --- MAIN APP ---
def main():
    st.title("📦 Πίνακας Αποθέματος")
    st.caption("Ιστορικά δεδομένα + χειροκίνητη εκτίμηση παραγγελίας")

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

    # --- FILTERING ---
    def is_inactive(row):
        recent = date_columns[-6:] if len(date_columns) >= 6 else date_columns
        values = pd.Series(row[recent].values.astype(float)).dropna()
        if len(values) == 0:
            return True
        return (values <= 0).all()

    grouped_df['_inactive'] = grouped_df.apply(is_inactive, axis=1)

    st.markdown("---")

    show_inactive = st.checkbox("Εμφάνιση ανενεργών προϊόντων (χωρίς πωλήσεις 6μ)", value=False)

    working_df = grouped_df if show_inactive else grouped_df[~grouped_df['_inactive']]

    st.caption(f"Εμφανίζονται {len(working_df)}/{len(grouped_df)} προϊόντα")

    col1, col2 = st.columns([2, 3])

    with col1:
        categories = ['Όλες'] + sorted(working_df['Category'].dropna().unique().tolist())
        selected_category = st.selectbox("Κατηγορία", categories)

    filtered_df = working_df if selected_category == 'Όλες' else working_df[working_df['Category'] == selected_category]
    filtered_df = filtered_df.reset_index(drop=True)

    product_options = filtered_df.apply(
        lambda x: f"{str(x['Product_Name'])[:50]}", axis=1
    ).tolist()

    with col2:
        selected_product = st.selectbox("Προϊόν", product_options)

    if not product_options:
        st.warning("Δεν βρέθηκαν προϊόντα")
        return

    # Get data
    selected_idx = product_options.index(selected_product)
    product_row = filtered_df.iloc[selected_idx]

    current_stock = get_total_stock(product_row, stock_df)
    sys_daily, sys_oos = get_system_data(product_row, sop_df)

    historical = product_row[date_columns].values.astype(float)
    stats = calculate_historical_stats(historical, date_columns)

    # ================================================================
    # MAIN DISPLAY
    # ================================================================
    st.markdown("---")
    st.markdown(f"## 📊 {product_row['Product_Name']}")

    if stats is None:
        st.warning("Δεν υπάρχουν αρκετά ιστορικά δεδομένα")
        return

    # --- CURRENT STATUS ---
    st.markdown("### 📦 Τρέχουσα Κατάσταση")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Απόθεμα Τώρα", f"{current_stock:,} κιβώτια")

    with col2:
        if stats['last_3m_avg_daily'] > 0:
            days_coverage = current_stock / stats['last_3m_avg_daily']
            color = "🟢" if days_coverage > 14 else "🟡" if days_coverage > 7 else "🔴"
            st.metric(f"{color} Κάλυψη (με πρόσφατο ρυθμό)", f"{days_coverage:.0f} ημέρες")
        else:
            st.metric("Κάλυψη", "—")

    with col3:
        if sys_daily and sys_daily > 0:
            sys_days = current_stock / sys_daily
            st.metric("Κάλυψη (σύστημα)", f"{sys_days:.0f} ημέρες")
            if sys_oos:
                st.error("⚠️ Σύστημα: OOS")
        else:
            st.metric("Κάλυψη (σύστημα)", "—")

    # --- HISTORICAL DATA ---
    st.markdown("---")
    st.markdown("### 📈 Ιστορικά Δεδομένα Πωλήσεων")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Μ.Ο. Ημερήσιος (τελευταίοι 3μ)", f"{stats['last_3m_avg_daily']:.1f}")

    with col2:
        st.metric("Μ.Ο. Ημερήσιος (συνολικός)", f"{stats['avg_daily']:.1f}")

    with col3:
        if stats['same_month_ly_daily']:
            st.metric("Ίδιος μήνας πέρυσι", f"{stats['same_month_ly_daily']:.1f}/ημ")
        else:
            st.metric("Ίδιος μήνας πέρυσι", "—")

    with col4:
        trend_icon = "📈" if stats['trend'] == 'up' else "📉" if stats['trend'] == 'down' else "➡️"
        trend_text = f"{stats['trend_pct']:+.0f}%" if stats['trend'] != 'unknown' else "?"
        st.metric(f"{trend_icon} Τάση", trend_text)

    # Volatility warning
    if stats['volatility'] > 50:
        st.warning(f"⚠️ Υψηλή μεταβλητότητα ({stats['volatility']:.0f}%) - οι εκτιμήσεις έχουν αβεβαιότητα")

    # --- MANUAL ORDER CALCULATOR ---
    st.markdown("---")
    st.markdown("### 🧮 Υπολογισμός Παραγγελίας")

    col1, col2 = st.columns(2)

    with col1:
        target_days = st.slider(
            "Πόσες ημέρες θέλεις να καλύψεις;",
            min_value=7, max_value=60, value=14, step=7
        )

        rate_option = st.radio(
            "Με ποιο ρυθμό πωλήσεων;",
            ["Πρόσφατος (3μ)", "Συνολικός Μ.Ο.", "Σύστημα", "Χειροκίνητα"]
        )

    with col2:
        if rate_option == "Πρόσφατος (3μ)":
            daily_rate = stats['last_3m_avg_daily']
        elif rate_option == "Συνολικός Μ.Ο.":
            daily_rate = stats['avg_daily']
        elif rate_option == "Σύστημα":
            daily_rate = sys_daily if sys_daily else stats['last_3m_avg_daily']
        else:
            daily_rate = st.number_input("Ημερήσιος ρυθμός:", min_value=0.0, value=stats['last_3m_avg_daily'])

        st.metric("Επιλεγμένος ρυθμός", f"{daily_rate:.1f} κιβ./ημέρα")

    # Calculate
    needed_stock = daily_rate * target_days
    order_quantity = max(0, needed_stock - current_stock)

    st.markdown("---")

    if order_quantity > 0:
        st.markdown(f"""
        <div style="background-color: #fff3cd; padding: 20px; border-radius: 10px; border: 2px solid #ffc107; text-align: center;">
        <h2 style="color: #856404; margin: 0;">📋 Προτεινόμενη Παραγγελία: {order_quantity:,.0f} κιβώτια</h2>
        <p style="color: #856404; margin: 10px 0 0 0;">Για κάλυψη {target_days} ημερών με ρυθμό {daily_rate:.1f} κιβ./ημέρα</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; border: 2px solid #28a745; text-align: center;">
        <h2 style="color: #155724; margin: 0;">✅ Επαρκές απόθεμα</h2>
        <p style="color: #155724; margin: 10px 0 0 0;">Καλύπτεις {target_days} ημέρες χωρίς παραγγελία</p>
        </div>
        """, unsafe_allow_html=True)

    # Breakdown
    st.markdown("")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Απαιτούμενο απόθεμα", f"{needed_stock:,.0f}")
    with col2:
        st.metric("Τρέχον απόθεμα", f"{current_stock:,}")
    with col3:
        st.metric("Διαφορά (παραγγελία)", f"{order_quantity:,.0f}")

    # --- CHART ---
    st.markdown("---")
    st.markdown("### 📊 Ιστορικό Πωλήσεων")

    dates = pd.to_datetime([f"{d}-01" for d in date_columns])
    hist_df = pd.DataFrame({'Date': dates, 'Sales': historical})
    hist_df = hist_df.dropna()

    if len(hist_df) > 0:
        hist_df['Year'] = hist_df['Date'].dt.year
        hist_df['Month'] = hist_df['Date'].dt.month

        fig = go.Figure()

        colors = {2024: '#AB63FA', 2025: '#FFA15A', 2026: '#19D3F3'}
        for year in sorted(hist_df['Year'].unique()):
            year_data = hist_df[hist_df['Year'] == year]
            fig.add_trace(go.Scatter(
                x=year_data['Month'],
                y=year_data['Sales'] / 30,
                name=str(year),
                mode='lines+markers',
                line=dict(color=colors.get(year, '#1f77b4'), width=2),
                marker=dict(size=8)
            ))

        # Average line
        fig.add_hline(y=stats['avg_daily'], line_dash="dash", line_color="gray",
                     annotation_text=f"Μ.Ο.: {stats['avg_daily']:.1f}")

        # Current month
        fig.add_vline(x=datetime.now().month, line_dash="dot", line_color="red",
                     annotation_text="Τώρα")

        fig.update_layout(
            xaxis_title='Μήνας',
            yaxis_title='Κιβώτια/Ημέρα',
            height=350,
            legend=dict(orientation="h", y=1.1),
            xaxis=dict(tickmode='array', tickvals=list(range(1,13)),
                      ticktext=['Ι','Φ','Μ','Α','Μ','Ι','Ι','Α','Σ','Ο','Ν','Δ'])
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- SUMMARY TABLE ---
    st.markdown("---")
    st.subheader("📋 Όλα τα Προϊόντα - Γρήγορη Επισκόπηση")

    summary_data = []
    for idx, row in filtered_df.iterrows():
        pname = str(row['Product_Name'])[:35]
        stock = get_total_stock(row, stock_df)
        s_daily, s_oos = get_system_data(row, sop_df)

        hist = row[date_columns].values.astype(float)
        pstats = calculate_historical_stats(hist, date_columns)

        if pstats:
            daily = pstats['last_3m_avg_daily']
            days = stock / daily if daily > 0 else 999
            need_14d = max(0, (daily * 14) - stock)
        else:
            daily = 0
            days = 999
            need_14d = 0

        status = "🔴" if days < 7 else "🟡" if days < 14 else "🟢"

        summary_data.append({
            '': status,
            'Προϊόν': pname,
            'Απόθεμα': stock,
            'Ημ/σιος': round(daily, 1) if daily > 0 else None,
            'Ημέρες': int(days) if days < 999 else None,
            'Για 14ημ': int(need_14d) if need_14d > 0 else 0,
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Ημέρες', ascending=True)

    st.dataframe(
        summary_df.style.format({
            'Απόθεμα': lambda x: f"{x:,}" if pd.notna(x) else "—",
            'Ημ/σιος': lambda x: f"{x:.1f}" if pd.notna(x) else "—",
            'Ημέρες': lambda x: f"{x:,}" if pd.notna(x) else "999+",
            'Για 14ημ': lambda x: f"{x:,}" if x > 0 else "✅",
        }),
        use_container_width=True,
        height=400
    )

    st.caption("🔴 <7 ημ | 🟡 7-14 ημ | 🟢 >14 ημ | Ημ/σιος = Μ.Ο. τελευταίων 3μ | Για 14ημ = Παραγγελία για 14 ημέρες κάλυψη")
    st.caption("v4.0 - Ιστορικά + Χειροκίνητη Εκτίμηση")

if __name__ == "__main__":
    main()
