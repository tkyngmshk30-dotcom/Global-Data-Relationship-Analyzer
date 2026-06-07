import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import wbgapi as wb
import plotly.express as px
import requests
import certifi
from io import StringIO
from datetime import datetime

# ==================================================
# App settings
# ==================================================
st.set_page_config(
    page_title="Global Data Relationship Analyzer",
    layout="wide"
)

st.title("Global Data Relationship Analyzer")
st.write(
    "Select countries, years, databases, indicators, and an analysis method. "
    "This tool helps users explore relationships between economic and development indicators."
)

if "search_history" not in st.session_state:
    st.session_state["search_history"] = []

# ==================================================
# Country groups
# ==================================================
COUNTRY_GROUPS = {
    "ASEAN": {
        "IDN": "Indonesia",
        "VNM": "Vietnam",
        "THA": "Thailand",
        "MYS": "Malaysia",
        "PHL": "Philippines",
        "SGP": "Singapore",
        "KHM": "Cambodia",
        "LAO": "Laos",
        "MMR": "Myanmar",
        "BRN": "Brunei",
    },
    "East Asia": {
        "CHN": "China",
        "JPN": "Japan",
        "KOR": "South Korea",
        "HKG": "Hong Kong",
        "MAC": "Macao",
    },
    "South Asia": {
        "IND": "India",
        "BGD": "Bangladesh",
        "PAK": "Pakistan",
        "LKA": "Sri Lanka",
        "NPL": "Nepal",
    },
    "Europe": {
        "DEU": "Germany",
        "FRA": "France",
        "GBR": "United Kingdom",
        "ITA": "Italy",
        "ESP": "Spain",
        "NLD": "Netherlands",
        "SWE": "Sweden",
        "NOR": "Norway",
        "FIN": "Finland",
    },
    "North America": {
        "USA": "United States",
        "CAN": "Canada",
        "MEX": "Mexico",
    },
    "Oceania": {
        "AUS": "Australia",
        "NZL": "New Zealand",
    },
    "Middle East": {
        "ARE": "United Arab Emirates",
        "SAU": "Saudi Arabia",
        "QAT": "Qatar",
        "TUR": "Türkiye",
        "ISR": "Israel",
    },
    "Africa": {
        "ZAF": "South Africa",
        "EGY": "Egypt",
        "NGA": "Nigeria",
        "KEN": "Kenya",
        "ETH": "Ethiopia",
        "GHA": "Ghana",
    },
    "Latin America": {
        "BRA": "Brazil",
        "ARG": "Argentina",
        "CHL": "Chile",
        "COL": "Colombia",
        "PER": "Peru",
    },
}

# ==================================================
# Indicator lists by database
# ==================================================
WORLD_BANK_INDICATORS = {
    "FDI inflows (current US$)": {
        "code": "BX.KLT.DINV.CD.WD",
        "short": "WB_FDI_Inflow",
    },
    "High-technology exports (% of manufactured exports)": {
        "code": "TX.VAL.TECH.MF.ZS",
        "short": "WB_HighTech_Exports",
    },
    "GDP per capita (current US$)": {
        "code": "NY.GDP.PCAP.CD",
        "short": "WB_GDP_per_capita",
    },
    "GDP growth (annual %)": {
        "code": "NY.GDP.MKTP.KD.ZG",
        "short": "WB_GDP_growth",
    },
    "Trade (% of GDP)": {
        "code": "NE.TRD.GNFS.ZS",
        "short": "WB_Trade_GDP",
    },
    "Exports of goods and services (% of GDP)": {
        "code": "NE.EXP.GNFS.ZS",
        "short": "WB_Exports_GDP",
    },
    "Imports of goods and services (% of GDP)": {
        "code": "NE.IMP.GNFS.ZS",
        "short": "WB_Imports_GDP",
    },
    "Manufacturing, value added (% of GDP)": {
        "code": "NV.IND.MANF.ZS",
        "short": "WB_Manufacturing_GDP",
    },
    "Industry, value added (% of GDP)": {
        "code": "NV.IND.TOTL.ZS",
        "short": "WB_Industry_GDP",
    },
    "Services, value added (% of GDP)": {
        "code": "NV.SRV.TOTL.ZS",
        "short": "WB_Services_GDP",
    },
    "Internet users (% of population)": {
        "code": "IT.NET.USER.ZS",
        "short": "WB_Internet_Users",
    },
    "Mobile cellular subscriptions (per 100 people)": {
        "code": "IT.CEL.SETS.P2",
        "short": "WB_Mobile_Subscriptions",
    },
    "Tertiary school enrollment (% gross)": {
        "code": "SE.TER.ENRR",
        "short": "WB_Tertiary_Enrollment",
    },
    "Education expenditure (% of GDP)": {
        "code": "SE.XPD.TOTL.GD.ZS",
        "short": "WB_Education_Expenditure",
    },
    "R&D expenditure (% of GDP)": {
        "code": "GB.XPD.RSDV.GD.ZS",
        "short": "WB_R_and_D",
    },
    "Patent applications, residents": {
        "code": "IP.PAT.RESD",
        "short": "WB_Patent_Applications",
    },
    "CO2 emissions per capita": {
        "code": "EN.ATM.CO2E.PC",
        "short": "WB_CO2_per_capita",
    },
    "Urban population (% of total population)": {
        "code": "SP.URB.TOTL.IN.ZS",
        "short": "WB_Urban_Population",
    },
    "Population, total": {
        "code": "SP.POP.TOTL",
        "short": "WB_Population",
    },
    "Unemployment (% of labor force)": {
        "code": "SL.UEM.TOTL.ZS",
        "short": "WB_Unemployment",
    },
}

OWID_INDICATORS = {
    "Life expectancy": {
        "slug": "life-expectancy",
        "short": "OWID_Life_Expectancy",
    },
    "Population": {
        "slug": "population",
        "short": "OWID_Population",
    },
    "GDP per capita": {
        "slug": "gdp-per-capita-worldbank",
        "short": "OWID_GDP_per_capita",
    },
    "CO2 emissions per capita": {
        "slug": "co2-emissions-per-capita",
        "short": "OWID_CO2_per_capita",
    },
    "Annual CO2 emissions": {
        "slug": "annual-co2-emissions-per-country",
        "short": "OWID_Annual_CO2",
    },
    "Share of population using the Internet": {
        "slug": "share-of-individuals-using-the-internet",
        "short": "OWID_Internet_Users",
    },
}

GAPMINDER_INDICATORS = {
    "GDP per capita": {
        "type": "column",
        "column": "gdpPercap",
        "short": "GAP_GDP_per_capita",
    },
    "Life expectancy": {
        "type": "column",
        "column": "lifeExp",
        "short": "GAP_Life_Expectancy",
    },
    "Population": {
        "type": "column",
        "column": "pop",
        "short": "GAP_Population",
    },
    "Total GDP estimated": {
        "type": "derived",
        "short": "GAP_Total_GDP",
    },
    "Population growth rate (%)": {
        "type": "derived",
        "short": "GAP_Population_Growth",
    },
    "GDP per capita growth rate (%)": {
        "type": "derived",
        "short": "GAP_GDP_per_capita_Growth",
    },
    "Life expectancy change": {
        "type": "derived",
        "short": "GAP_Life_Expectancy_Change",
    },
}

DATABASES = {
    "World Bank": WORLD_BANK_INDICATORS,
    "Our World in Data": OWID_INDICATORS,
    "Gapminder": GAPMINDER_INDICATORS,
}

# ==================================================
# Utility functions
# ==================================================
def clean_variable_name(name):
    return (
        name.replace(" ", "_")
        .replace("-", "_")
        .replace("%", "Percent")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace(",", "")
        .replace(".", "")
        .replace("$", "USD")
    )


@st.cache_data
def fetch_world_bank(indicator_code, variable_name, countries, start_year, end_year):
    data = wb.data.DataFrame(
        indicator_code,
        countries,
        time=range(start_year, end_year + 1),
        labels=False
    )

    data = data.stack().reset_index()
    data.columns = ["Country", "Year", variable_name]
    data["Year"] = data["Year"].str.replace("YR", "").astype(int)

    return data


@st.cache_data
def fetch_owid(slug, variable_name, countries, start_year, end_year):
    url = f"https://ourworldindata.org/grapher/{slug}.csv"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=30,
            verify=certifi.where()
        )
        response.raise_for_status()

    except requests.exceptions.SSLError:
        response = requests.get(
            url,
            headers=headers,
            timeout=30,
            verify=False
        )
        response.raise_for_status()

    data = pd.read_csv(StringIO(response.text))

    if not {"Entity", "Code", "Year"}.issubset(data.columns):
        raise ValueError("This OWID dataset format is not supported.")

    value_cols = [
        col for col in data.columns
        if col not in ["Entity", "Code", "Year"]
    ]

    if len(value_cols) == 0:
        raise ValueError("No data value column was found.")

    value_col = value_cols[0]

    data = data.rename(
        columns={
            "Entity": "Country_Name",
            "Code": "Country",
            value_col: variable_name,
        }
    )

    data = data[["Country", "Country_Name", "Year", variable_name]]
    data = data[data["Country"].isin(countries)]
    data = data[(data["Year"] >= start_year) & (data["Year"] <= end_year)]

    return data


@st.cache_data
def fetch_gapminder(indicator_info, variable_name, countries, start_year, end_year):
    data = px.data.gapminder()

    data = data.rename(
        columns={
            "iso_alpha": "Country",
            "country": "Country_Name",
            "year": "Year",
            "gdpPercap": "GDP_per_capita",
            "lifeExp": "Life_Expectancy",
            "pop": "Population",
        }
    )

    data = data.sort_values(["Country", "Year"]).copy()

    data["Total_GDP"] = data["GDP_per_capita"] * data["Population"]
    data["Population_Growth"] = data.groupby("Country")["Population"].pct_change() * 100
    data["GDP_per_capita_Growth"] = data.groupby("Country")["GDP_per_capita"].pct_change() * 100
    data["Life_Expectancy_Change"] = data.groupby("Country")["Life_Expectancy"].diff()

    if indicator_info["type"] == "column":
        source_col = {
            "gdpPercap": "GDP_per_capita",
            "lifeExp": "Life_Expectancy",
            "pop": "Population",
        }[indicator_info["column"]]
    else:
        source_col = {
            "GAP_Total_GDP": "Total_GDP",
            "GAP_Population_Growth": "Population_Growth",
            "GAP_GDP_per_capita_Growth": "GDP_per_capita_Growth",
            "GAP_Life_Expectancy_Change": "Life_Expectancy_Change",
        }[indicator_info["short"]]

    data = data.rename(columns={source_col: variable_name})

    data = data[["Country", "Country_Name", "Year", variable_name]]
    data = data[data["Country"].isin(countries)]
    data = data[(data["Year"] >= start_year) & (data["Year"] <= end_year)]

    return data


def fetch_indicator(database_type, indicator_name, countries, start_year, end_year):
    info = DATABASES[database_type][indicator_name]
    variable_name = clean_variable_name(info["short"])

    if database_type == "World Bank":
        return fetch_world_bank(
            info["code"],
            variable_name,
            countries,
            start_year,
            end_year
        ), variable_name

    if database_type == "Our World in Data":
        return fetch_owid(
            info["slug"],
            variable_name,
            countries,
            start_year,
            end_year
        ), variable_name

    if database_type == "Gapminder":
        return fetch_gapminder(
            info,
            variable_name,
            countries,
            start_year,
            end_year
        ), variable_name

    raise ValueError("Unsupported database type.")


def add_lag(df, x_var, lag_years):
    if lag_years == 0:
        return df, x_var

    lag_var = f"{x_var}_lag{lag_years}"
    df = df.sort_values(["Country", "Year"]).copy()
    df[lag_var] = df.groupby("Country")[x_var].shift(lag_years)

    return df, lag_var


def build_formula(y_var, x_var, method):
    if method == "Simple OLS":
        return f"{y_var} ~ {x_var}"

    if method == "OLS + country fixed effects":
        return f"{y_var} ~ {x_var} + C(Country)"

    if method == "OLS + year fixed effects":
        return f"{y_var} ~ {x_var} + C(Year)"

    if method == "OLS + country and year fixed effects":
        return f"{y_var} ~ {x_var} + C(Country) + C(Year)"

    if method == "Quadratic OLS":
        return f"{y_var} ~ {x_var} + I({x_var} ** 2)"

    return None


def prepare_merged_data(df_x, df_y):
    df = pd.merge(
        df_x,
        df_y,
        on=["Country", "Year"],
        how="inner"
    )

    country_name_cols = [col for col in df.columns if "Country_Name" in col]

    if len(country_name_cols) > 0:
        df["Country_Name"] = df[country_name_cols[0]]
    else:
        df["Country_Name"] = df["Country"]

    return df


# ==================================================
# Sidebar
# ==================================================
st.sidebar.header("Analysis Settings")

group = st.sidebar.selectbox(
    "Select a country group",
    list(COUNTRY_GROUPS.keys())
)

country_options = COUNTRY_GROUPS[group]

selected_countries = st.sidebar.multiselect(
    "Select countries",
    options=list(country_options.keys()),
    default=list(country_options.keys()),
    format_func=lambda code: f"{code} - {country_options[code]}"
)

custom_codes = st.sidebar.text_input(
    "Add custom ISO3 country codes, separated by commas",
    value="",
    placeholder="Example: AUS,NZL,CAN"
)

start_year = st.sidebar.number_input(
    "Start year",
    min_value=1950,
    max_value=2026,
    value=2000
)

end_year = st.sidebar.number_input(
    "End year",
    min_value=1950,
    max_value=2026,
    value=2023
)

st.sidebar.markdown("---")

x_database_type = st.sidebar.selectbox(
    "Select X database",
    list(DATABASES.keys()),
    index=0
)

x_indicator_names = list(DATABASES[x_database_type].keys())

x_indicator_name = st.sidebar.selectbox(
    "Select X variable / explanatory variable",
    x_indicator_names,
    index=0
)

y_database_type = st.sidebar.selectbox(
    "Select Y database",
    list(DATABASES.keys()),
    index=0
)

y_indicator_names = list(DATABASES[y_database_type].keys())

y_default_index = 1 if len(y_indicator_names) > 1 else 0

y_indicator_name = st.sidebar.selectbox(
    "Select Y variable / dependent variable",
    y_indicator_names,
    index=y_default_index
)

st.sidebar.markdown("---")

log_x = st.sidebar.checkbox(
    "Log-transform X variable",
    value=False
)

lag_years = st.sidebar.selectbox(
    "Lag X variable by years",
    options=[0, 1, 2, 3, 5],
    index=0
)

analysis_method = st.sidebar.selectbox(
    "Select analysis method",
    [
        "Correlation only",
        "Simple OLS",
        "OLS + country fixed effects",
        "OLS + year fixed effects",
        "OLS + country and year fixed effects",
        "Quadratic OLS",
    ],
    index=2
)

run_button = st.sidebar.button("Run analysis")

# ==================================================
# Main analysis
# ==================================================
if run_button:
    countries = selected_countries.copy()

    if custom_codes.strip():
        additional_codes = [
            code.strip().upper()
            for code in custom_codes.split(",")
            if code.strip()
        ]
        countries.extend(additional_codes)

    countries = list(dict.fromkeys(countries))

    if len(countries) < 2:
        st.error("Please select at least two countries.")

    elif start_year >= end_year:
        st.error("The end year must be later than the start year.")

    else:
        try:
            with st.spinner("Fetching and preparing data..."):
                df_x, x_var = fetch_indicator(
                    x_database_type,
                    x_indicator_name,
                    countries,
                    int(start_year),
                    int(end_year)
                )

                df_y, y_var = fetch_indicator(
                    y_database_type,
                    y_indicator_name,
                    countries,
                    int(start_year),
                    int(end_year)
                )

                df = prepare_merged_data(df_x, df_y)
                df = df.dropna(subset=[x_var, y_var]).copy()

                if x_var == y_var:
                    st.error(
                        "The selected X and Y variables have the same internal variable name. "
                        "Please select a different combination."
                    )
                    st.stop()

                df, x_model_var = add_lag(df, x_var, lag_years)
                df = df.dropna(subset=[x_model_var, y_var]).copy()

                if log_x:
                    log_var = f"log_{x_model_var}"
                    df[log_var] = np.log(df[x_model_var].where(df[x_model_var] > 0))
                    df = df.dropna(subset=[log_var]).copy()
                    x_model_var = log_var

                if len(df) == 0:
                    st.error("No available data for the selected settings.")
                    st.stop()

                correlation = df[[x_model_var, y_var]].corr().iloc[0, 1]

                formula = build_formula(y_var, x_model_var, analysis_method)

                model = None
                if analysis_method != "Correlation only":
                    model = smf.ols(formula=formula, data=df).fit()
                    df["Predicted"] = model.predict(df)

                history_record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "country_group": group,
                    "countries": ", ".join(countries),
                    "start_year": int(start_year),
                    "end_year": int(end_year),
                    "x_database": x_database_type,
                    "x_indicator": x_indicator_name,
                    "y_database": y_database_type,
                    "y_indicator": y_indicator_name,
                    "log_x": log_x,
                    "lag_years": lag_years,
                    "analysis_method": analysis_method,
                    "observations": len(df),
                    "correlation": correlation,
                    "formula": formula if formula is not None else "Correlation only",
                }

                if model is not None:
                    coef = model.params.get(x_model_var, np.nan)
                    p_value = model.pvalues.get(x_model_var, np.nan)
                    r_squared = model.rsquared
                    adj_r_squared = model.rsquared_adj

                    history_record.update({
                        "x_coefficient": coef,
                        "x_p_value": p_value,
                        "r_squared": r_squared,
                        "adj_r_squared": adj_r_squared,
                    })
                else:
                    history_record.update({
                        "x_coefficient": np.nan,
                        "x_p_value": np.nan,
                        "r_squared": np.nan,
                        "adj_r_squared": np.nan,
                    })

                st.session_state["search_history"].append(history_record)

            st.success("Analysis completed.")

            # ==================================================
            # 1. Data overview
            # ==================================================
            st.subheader("1. Data Overview")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Observations", len(df))
            col2.metric("Countries", len(countries))
            col3.metric("X database", x_database_type)
            col4.metric("Y database", y_database_type)

            st.write("Selected X variable:")
            st.code(f"{x_database_type} | {x_indicator_name}")

            st.write("Selected Y variable:")
            st.code(f"{y_database_type} | {y_indicator_name}")

            with st.expander("Show cleaned panel data"):
                st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="Download cleaned data as CSV",
                data=csv,
                file_name="global_data_analysis.csv",
                mime="text/csv"
            )

            # ==================================================
            # 2. Analysis method
            # ==================================================
            st.subheader("2. Analysis Method")

            st.write(f"Selected method: **{analysis_method}**")

            if lag_years > 0:
                st.write(
                    f"The X variable is lagged by **{lag_years} year(s)**. "
                    "This means the model uses past X values to explain current Y values."
                )

            if log_x:
                st.write(
                    "The X variable is log-transformed to reduce the impact of large scale differences."
                )

            if formula is not None:
                st.write("Regression formula:")
                st.code(formula)

            # ==================================================
            # 3. Scatter chart
            # ==================================================
            st.subheader("3. Relationship between X and Y")

            st.scatter_chart(
                df,
                x=x_model_var,
                y=y_var,
                color="Country"
            )

            # ==================================================
            # 4. Correlation
            # ==================================================
            st.subheader("4. Correlation")

            st.write(f"The correlation coefficient is **{correlation:.4f}**.")

            if correlation > 0:
                st.write("→ The two variables show a positive linear relationship.")
            elif correlation < 0:
                st.write("→ The two variables show a negative linear relationship.")
            else:
                st.write("→ The two variables show almost no linear relationship.")

            # ==================================================
            # 5. Regression results
            # ==================================================
            if model is not None:
                st.subheader("5. Regression Results")

                st.text(model.summary())

                result_table = pd.DataFrame({
                    "Variable": model.params.index,
                    "Coefficient": model.params.values,
                    "P-value": model.pvalues.values
                })

                st.write("Key coefficients:")
                st.dataframe(result_table)

                st.subheader("6. Actual vs Predicted")

                st.scatter_chart(
                    df,
                    x=y_var,
                    y="Predicted",
                    color="Country"
                )

                st.subheader("7. Simple Interpretation")

                coef = model.params.get(x_model_var, None)
                p_value = model.pvalues.get(x_model_var, None)

                if coef is not None and p_value is not None:
                    st.write(f"- The coefficient of **{x_model_var}** is **{coef:.4f}**.")
                    st.write(f"- The p-value of **{x_model_var}** is **{p_value:.4f}**.")

                    if coef > 0:
                        st.write(
                            "→ The coefficient is positive. Higher X values are associated with higher Y values in this model."
                        )
                    elif coef < 0:
                        st.write(
                            "→ The coefficient is negative. Higher X values are associated with lower Y values in this model."
                        )
                    else:
                        st.write(
                            "→ The coefficient is close to zero."
                        )

                    if p_value < 0.05:
                        st.write("→ The relationship is statistically significant at the 5% level.")
                    else:
                        st.write("→ The relationship is not statistically significant at the 5% level.")

            st.info(
                "Important note: This tool shows correlation, not strict causality. "
                "Even if a result is statistically significant, it does not prove that X directly causes Y."
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    st.info(
        "Select countries, years, X/Y databases, indicators, and an analysis method from the sidebar, then click 'Run analysis'."
    )

# ==================================================
# Search history
# ==================================================
st.sidebar.markdown("---")
st.sidebar.subheader("Search History")

if len(st.session_state["search_history"]) > 0:
    history_df = pd.DataFrame(st.session_state["search_history"])

    with st.sidebar.expander("Show search history"):
        st.dataframe(history_df)

    history_csv = history_df.to_csv(index=False).encode("utf-8-sig")

    st.sidebar.download_button(
        label="Download search history",
        data=history_csv,
        file_name="search_history.csv",
        mime="text/csv"
    )

    if st.sidebar.button("Clear search history"):
        st.session_state["search_history"] = []
        st.rerun()
else:
    st.sidebar.write("No search history yet.")
