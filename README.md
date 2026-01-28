# Health SDG Dashboard

A Streamlit dashboard for visualizing Health-Related Sustainable Development Goals (SDGs) using data from the World Health Organization (WHO) Global Health Observatory.

## Features

- 🌍 **Global Overview**: Interactive world map showing health indicators by country
- 🔍 **Country Analysis**: Detailed time-series analysis for individual countries
- 📈 **Comparison View**: Compare health metrics across countries
- 📊 **Multiple SDG Indicators**: 
  - Maternal Mortality (SDG 3.1)
  - Under-5 & Neonatal Mortality (SDG 3.2)
  - HIV, TB, Malaria Incidence (SDG 3.3)
  - NCD & Suicide Mortality (SDG 3.4)
  - Alcohol Consumption (SDG 3.5)
  - Road Traffic Deaths (SDG 3.6)
  - Adolescent Birth Rate (SDG 3.7)
  - Universal Health Coverage (SDG 3.8)
  - Tobacco Use (SDG 3.a)
  - Immunization Coverage (SDG 3.b)
  - Health Worker Density (SDG 3.c)
  - Life Expectancy at Birth

## Setup

1. Create and activate the virtual environment:
```bash
python3 -m venv envenv
source envenv/bin/activate  # On Linux/Mac
# or
envenv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

## Data Source

Data is fetched in real-time from the [WHO Global Health Observatory OData API](https://www.who.int/data/gho/info/gho-odata-api).

## Technologies Used

- **Streamlit** - Web application framework
- **Pandas** - Data manipulation
- **Plotly** - Interactive visualizations
- **Requests** - API calls

## License

MIT License
