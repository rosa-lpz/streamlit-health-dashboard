"""
Health-Related Sustainable Development Goals Dashboard
Using WHO Global Health Observatory (GHO) OData API
"""

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="Health SDG Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# WHO GHO API Base URL
API_BASE_URL = "https://ghoapi.azureedge.net/api"

# Health-related SDG Indicators
SDG_INDICATORS = {
    "SDG 3.1 - Maternal Mortality": {
        "code": "MDG_0000000001",
        "description": "Maternal mortality ratio (per 100,000 live births)"
    },
    "SDG 3.2 - Under-5 Mortality": {
        "code": "MDG_0000000007",
        "description": "Under-five mortality rate (per 1,000 live births)"
    },
    "SDG 3.2 - Neonatal Mortality": {
        "code": "MDG_0000000003",
        "description": "Neonatal mortality rate (per 1,000 live births)"
    },
    "SDG 3.3 - HIV Incidence": {
        "code": "HIV_0000000026",
        "description": "New HIV infections (per 1,000 uninfected population)"
    },
    "SDG 3.3 - Tuberculosis Incidence": {
        "code": "MDG_0000000020",
        "description": "Tuberculosis incidence (per 100,000 population)"
    },
    "SDG 3.3 - Malaria Incidence": {
        "code": "MALARIA_INCIDENCE",
        "description": "Malaria incidence (per 1,000 population at risk)"
    },
    "SDG 3.4 - NCD Mortality": {
        "code": "NCDMORT3070",
        "description": "Probability of dying from cardiovascular disease, cancer, diabetes, or chronic respiratory disease between ages 30 and 70"
    },
    "SDG 3.4 - Suicide Mortality": {
        "code": "SDGSUICIDE",
        "description": "Suicide mortality rate (per 100,000 population)"
    },
    "SDG 3.5 - Alcohol Consumption": {
        "code": "SA_0000001688",
        "description": "Total alcohol per capita consumption (liters of pure alcohol)"
    },
    "SDG 3.6 - Road Traffic Deaths": {
        "code": "RS_198",
        "description": "Road traffic mortality rate (per 100,000 population)"
    },
    "SDG 3.7 - Adolescent Birth Rate": {
        "code": "MDG_0000000005",
        "description": "Adolescent birth rate (per 1,000 women aged 15-19)"
    },
    "SDG 3.8 - UHC Service Coverage": {
        "code": "UHC_INDEX_REPORTED",
        "description": "Universal Health Coverage service coverage index"
    },
    "SDG 3.a - Tobacco Use": {
        "code": "M_Est_smk_curr_std",
        "description": "Age-standardized prevalence of current tobacco smoking among persons aged 15 years and older"
    },
    "SDG 3.b - DTP3 Immunization": {
        "code": "WHS4_100",
        "description": "Diphtheria-tetanus-pertussis (DTP3) immunization coverage among 1-year-olds (%)"
    },
    "SDG 3.c - Health Worker Density": {
        "code": "HWF_0001",
        "description": "Medical doctors (per 10,000 population)"
    },
    "Life Expectancy at Birth": {
        "code": "WHOSIS_000001",
        "description": "Life expectancy at birth (years)"
    }
}


@st.cache_data(ttl=3600)
def fetch_countries() -> pd.DataFrame:
    """Fetch list of countries from WHO API"""
    try:
        url = f"{API_BASE_URL}/DIMENSION/COUNTRY/DimensionValues"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "value" in data:
            df = pd.DataFrame(data["value"])
            cols = ["Code", "Title"]
            if "ParentCode" in df.columns:
                cols.append("ParentCode")
            if "ParentTitle" in df.columns:
                cols.append("ParentTitle")
            df = df[cols].rename(columns={
                "Code": "code",
                "Title": "name",
                "ParentCode": "region_code",
                "ParentTitle": "region_name"
            })
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching countries: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_regions() -> pd.DataFrame:
    """Fetch list of WHO regions"""
    try:
        url = f"{API_BASE_URL}/DIMENSION/REGION/DimensionValues"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "value" in data:
            df = pd.DataFrame(data["value"])
            return df[["Code", "Title"]].rename(columns={"Code": "code", "Title": "name"})
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching regions: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_indicator_data(indicator_code: str, country_code: Optional[str] = None) -> pd.DataFrame:
    """Fetch indicator data from WHO API"""
    try:
        url = f"{API_BASE_URL}/{indicator_code}"
        
        if country_code:
            url += f"?$filter=SpatialDim eq '{country_code}'"
        
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        if "value" in data and len(data["value"]) > 0:
            df = pd.DataFrame(data["value"])
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()


def process_indicator_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process and clean indicator data"""
    if df.empty:
        return df
    
    # Select relevant columns
    cols_to_keep = []
    available_cols = df.columns.tolist()
    
    desired_cols = ["SpatialDim", "TimeDim", "NumericValue", "Value", "Dim1", "Dim2", "Low", "High"]
    for col in desired_cols:
        if col in available_cols:
            cols_to_keep.append(col)
    
    if not cols_to_keep:
        return df
    
    df = df[cols_to_keep].copy()

    # Handle potential duplicate value columns
    if "NumericValue" in df.columns and "Value" in df.columns:
        df["Value"] = df["NumericValue"].combine_first(df["Value"])
        df = df.drop(columns=["NumericValue"])
    
    # Rename columns for clarity
    rename_map = {
        "SpatialDim": "Country_Code",
        "TimeDim": "Year",
        "NumericValue": "Value",
        "Dim1": "Dimension_1",
        "Dim2": "Dimension_2"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    # Convert Year to numeric if exists
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    
    # Convert Value to numeric if exists
    if "Value" in df.columns:
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    
    return df


def create_time_series_chart(df: pd.DataFrame, title: str, country_name: str) -> go.Figure:
    """Create a time series chart"""
    fig = px.line(
        df,
        x="Year",
        y="Value",
        title=f"{title} - {country_name}",
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white"
    )
    
    return fig


def create_comparison_chart(df: pd.DataFrame, title: str, year: int) -> go.Figure:
    """Create a bar chart comparing countries"""
    df_year = df[df["Year"] == year].copy()
    df_year = df_year.nlargest(20, "Value")
    
    fig = px.bar(
        df_year,
        x="Country_Code",
        y="Value",
        title=f"{title} - Top 20 Countries ({year})",
        color="Value",
        color_continuous_scale="Viridis"
    )
    
    fig.update_layout(
        xaxis_title="Country",
        yaxis_title="Value",
        xaxis_tickangle=45,
        template="plotly_white"
    )
    
    return fig


def create_world_map(df: pd.DataFrame, title: str, year: int) -> go.Figure:
    """Create a choropleth world map"""
    df_year = df[df["Year"] == year].copy()
    
    fig = px.choropleth(
        df_year,
        locations="Country_Code",
        color="Value",
        hover_name="Country_Code",
        color_continuous_scale="RdYlGn_r",
        title=f"{title} ({year})"
    )
    
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type="natural earth"
        ),
        template="plotly_white"
    )
    
    return fig


def main():
    # Header
    st.title("🏥 Health-Related Sustainable Development Goals Dashboard")
    st.markdown("""
    This dashboard displays key health indicators related to the **United Nations Sustainable Development Goals (SDGs)**.
    Data is sourced from the **World Health Organization (WHO) Global Health Observatory**.
    
    ---
    """)
    
    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    
    # Indicator selection
    st.sidebar.subheader("Select Indicator")
    selected_indicator_name = st.sidebar.selectbox(
        "Health Indicator",
        options=list(SDG_INDICATORS.keys()),
        index=0
    )
    
    indicator_info = SDG_INDICATORS[selected_indicator_name]
    indicator_code = indicator_info["code"]
    indicator_description = indicator_info["description"]
    
    # Display indicator description
    st.sidebar.info(f"📋 {indicator_description}")
    
    # View mode selection
    st.sidebar.subheader("View Mode")
    view_mode = st.sidebar.radio(
        "Select View",
        ["🌍 Global Overview", "🔍 Country Analysis", "📈 Comparison"]
    )
    
    # Fetch countries for selection
    countries_df = fetch_countries()
    regions_df = fetch_regions()
    
    # Region selection (before country selection)
    selected_region_code = None
    if not regions_df.empty:
        region_options = {"All Regions": None}
        region_options.update(dict(zip(regions_df["name"], regions_df["code"])))
        selected_region_name = st.sidebar.selectbox(
            "Select Region",
            options=list(region_options.keys()),
            index=0
        )
        selected_region_code = region_options[selected_region_name]

    if view_mode == "🔍 Country Analysis":
        if not countries_df.empty:
            filtered_countries_df = countries_df
            if selected_region_code and "region_code" in countries_df.columns:
                filtered_countries_df = countries_df[countries_df["region_code"] == selected_region_code]
            if filtered_countries_df.empty:
                filtered_countries_df = countries_df
            country_options = dict(zip(filtered_countries_df["name"], filtered_countries_df["code"]))
            selected_country_name = st.sidebar.selectbox(
                "Select Country",
                options=list(country_options.keys()),
                index=list(country_options.keys()).index("India") if "India" in country_options.keys() else 0
            )
            selected_country_code = country_options[selected_country_name]
        else:
            selected_country_name = "India"
            selected_country_code = "IND"
    
    # Main content area
    st.header(f"📊 {selected_indicator_name}")
    
    # Show loading spinner while fetching data
    with st.spinner("Fetching data from WHO Global Health Observatory..."):
        if view_mode == "🔍 Country Analysis":
            df = fetch_indicator_data(indicator_code, selected_country_code)
        else:
            df = fetch_indicator_data(indicator_code)
    
    if df.empty:
        st.warning("⚠️ No data available for this indicator. Please try another indicator.")
        return
    
    # Process data
    df = process_indicator_data(df)
    
    if df.empty or "Value" not in df.columns:
        st.warning("⚠️ Unable to process data for this indicator.")
        return
    
    # Remove NaN values
    df = df.dropna(subset=["Value"])
    
    if df.empty:
        st.warning("⚠️ No valid data values available.")
        return

    # Apply region filter when possible (global/comparison views)
    if selected_region_code and "Country_Code" in df.columns:
        if "region_code" in countries_df.columns:
            region_countries = countries_df[countries_df["region_code"] == selected_region_code]["code"].unique()
            df = df[df["Country_Code"].isin(region_countries)]
        else:
            st.sidebar.info("Region filter is unavailable (country-to-region mapping not provided by API).")
    
    # Get available years
    if "Year" in df.columns:
        available_years = sorted(df["Year"].dropna().unique().astype(int))
        if available_years:
            min_year = int(min(available_years))
            max_year = int(max(available_years))
        else:
            min_year = max_year = 2020
    else:
        min_year = max_year = 2020
        available_years = [2020]
    
    # Different views
    if view_mode == "🌍 Global Overview":
        st.subheader("🗺️ Global Map View")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            selected_year = st.selectbox(
                "Select Year",
                options=available_years,
                index=len(available_years) - 1 if available_years else 0
            )
        
        with col1:
            # World map
            if "Country_Code" in df.columns:
                fig_map = create_world_map(df, selected_indicator_name, selected_year)
                st.plotly_chart(fig_map, use_container_width=True)
        
        # Statistics
        st.subheader("📈 Global Statistics")
        
        df_year = df[df["Year"] == selected_year] if "Year" in df.columns else df
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Countries with Data", len(df_year["Country_Code"].unique()) if "Country_Code" in df_year.columns else "N/A")
        
        with col2:
            st.metric("Global Average", f"{df_year['Value'].mean():.2f}" if not df_year.empty else "N/A")
        
        with col3:
            st.metric("Minimum", f"{df_year['Value'].min():.2f}" if not df_year.empty else "N/A")
        
        with col4:
            st.metric("Maximum", f"{df_year['Value'].max():.2f}" if not df_year.empty else "N/A")
        
        # Top and bottom performers
        st.subheader("🏆 Rankings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top 10 (Highest Values)**")
            top_10 = df_year.nlargest(10, "Value")[["Country_Code", "Value"]].reset_index(drop=True)
            top_10.index = top_10.index + 1
            st.dataframe(top_10, use_container_width=True)
        
        with col2:
            st.markdown("**Bottom 10 (Lowest Values)**")
            bottom_10 = df_year.nsmallest(10, "Value")[["Country_Code", "Value"]].reset_index(drop=True)
            bottom_10.index = bottom_10.index + 1
            st.dataframe(bottom_10, use_container_width=True)
    
    elif view_mode == "🔍 Country Analysis":
        st.subheader(f"📍 {selected_country_name}")
        
        # Time series chart
        if "Year" in df.columns and len(df) > 1:
            # Check for dimension filters
            if "Dimension_1" in df.columns:
                unique_dims = df["Dimension_1"].dropna().unique()
                if len(unique_dims) > 1:
                    selected_dim = st.selectbox("Filter by Dimension", options=["All"] + list(unique_dims))
                    if selected_dim != "All":
                        df = df[df["Dimension_1"] == selected_dim]
            
            df_trend = df.groupby("Year")["Value"].mean().reset_index()
            fig = create_time_series_chart(df_trend, selected_indicator_name, selected_country_name)
            st.plotly_chart(fig, use_container_width=True)

        # Year filter for Country Analysis
        df_filtered = df
        if "Year" in df.columns and available_years:
            selected_year = st.selectbox(
                "Select Year",
                options=available_years,
                index=len(available_years) - 1 if available_years else 0
            )
            df_filtered = df[df["Year"] == selected_year]
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_value = df_filtered["Value"].sum()
            st.metric("Total Value", f"{total_value:.2f}" if pd.notna(total_value) else "N/A")

        with col2:
            latest_value = df_filtered["Value"].mean() if not df_filtered.empty else float("nan")
            st.metric("Latest Value", f"{latest_value:.2f}" if pd.notna(latest_value) else "N/A")

        

        with col3:
            avg_value = df_filtered["Value"].mean()
            st.metric("Average (All Years)", f"{avg_value:.2f}" if pd.notna(avg_value) else "N/A")
        with col4:
            if "Year" in df.columns and len(df["Year"].unique()) > 1:
                years = sorted(df["Year"].unique())
                if len(years) >= 2:
                    first_year = years[0]
                    last_year = years[-1]
                    first_value = df[df["Year"] == first_year]["Value"].mean()
                    last_value = df[df["Year"] == last_year]["Value"].mean()
                    if pd.notna(first_value) and pd.notna(last_value) and first_value != 0:
                        change = ((last_value - first_value) / first_value) * 100
                        st.metric(f"Change ({first_year}-{last_year})", f"{change:+.1f}%")
                    else:
                        st.metric("Change", "N/A")
                else:
                    st.metric("Change", "N/A")
            else:
                st.metric("Change", "N/A")
        
        # Data table
        st.subheader("📋 Raw Data")
        st.dataframe(df_filtered.sort_values("Year", ascending=False) if "Year" in df_filtered.columns else df_filtered, use_container_width=True)
    
    elif view_mode == "📈 Comparison":
        st.subheader("📊 Country Comparison")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            selected_year = st.selectbox(
                "Select Year for Comparison",
                options=available_years,
                index=len(available_years) - 1 if available_years else 0
            )
        
        with col1:
            if "Country_Code" in df.columns:
                fig_bar = create_comparison_chart(df, selected_indicator_name, selected_year)
                st.plotly_chart(fig_bar, use_container_width=True)
        
        # Distribution chart
        st.subheader("📊 Value Distribution")
        
        df_year = df[df["Year"] == selected_year] if "Year" in df.columns else df
        
        fig_hist = px.histogram(
            df_year,
            x="Value",
            nbins=30,
            title=f"Distribution of {selected_indicator_name} ({selected_year})",
            color_discrete_sequence=["#636EFA"]
        )
        fig_hist.update_layout(
            xaxis_title="Value",
            yaxis_title="Number of Countries",
            template="plotly_white"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Data Source: <a href='https://www.who.int/data/gho' target='_blank'>WHO Global Health Observatory</a></p>
        <p>API: <a href='https://www.who.int/data/gho/info/gho-odata-api' target='_blank'>GHO OData API</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
