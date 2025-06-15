from typing import Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone
import pytz
from scipy import stats
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .utils import retrieve_data


def make_plots(
    auth_token: str, start_date: Optional[str] = None, end_date: Optional[str] = None
):
    # Convert to ISO8601 format with time
    if start_date:
        start_date = (
            datetime.strptime(start_date, "%Y-%m-%d")
            .replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    else:
        start_date = "2015-08-01T00:00:00.000Z"

    if end_date:
        end_date = (
            datetime.strptime(end_date, "%Y-%m-%d")
            .replace(
                hour=23, minute=59, second=59, microsecond=999000, tzinfo=timezone.utc
            )
            .isoformat()
            .replace("+00:00", "Z")
        )
    else:
        end_date = (
            datetime.now(timezone.utc)
            .replace(hour=23, minute=59, second=59, microsecond=999000)
            .isoformat()
            .replace("+00:00", "Z")
        )
    # === Data Preparation ===
    data = retrieve_data(auth_token, start_date, end_date)
    df = pd.json_normalize(data)

    # Extract timezone from location.timeZone
    # For multiple locations, we'll handle dynamically below
    # Convert createdAt for each record based on its location.timeZone
    def convert_to_local_timezone(row):
        try:
            tz = pytz.timezone(row["location.timeZone"])
        except pytz.exceptions.UnknownTimeZoneError:
            tz = pytz.UTC
        # Parse the timestamp
        ts = pd.to_datetime(row["createdAt"])
        # If already timezone-aware, convert directly; otherwise, localize to UTC first
        if ts.tzinfo is not None:
            return ts.tz_convert(tz)
        return ts.tz_localize("UTC").tz_convert(tz)

    df["createdAt"] = df.apply(convert_to_local_timezone, axis=1)
    df["month"] = df["createdAt"].dt.to_period("M").astype(str)
    df["date"] = df["createdAt"].dt.date
    df["day_of_week"] = df["createdAt"].dt.day_name()
    df["location_name"] = df["location.name"]
    df["hour"] = df["createdAt"].dt.hour  # Now in each location's timezone
    df["year"] = df["createdAt"].dt.year
    df["customer_name"] = df["customer.firstName"] + " " + df["customer.lastName"]

    # Handle pass type and membership
    def get_pass_type(row):
        if pd.notna(row.get("pass.passType.name")):
            return row["pass.passType.name"].strip()
        elif pd.notna(row.get("membership.membershipType.name")):
            return row["membership.membershipType.name"].strip()
        return "No Pass"

    def get_membership_id(row):
        return str(row["membershipId"]) if pd.notna(row.get("membershipId")) else "None"

    def get_membership_name(row):
        return (
            row["membership.membershipType.name"].strip()
            if pd.notna(row.get("membership.membershipType.name"))
            else "None"
        )

    df["pass_type"] = df.apply(get_pass_type, axis=1)
    df["membership_id"] = df.apply(get_membership_id, axis=1)
    df["membership_name"] = df.apply(get_membership_name, axis=1).where(
        df["membership_id"] != "None", "None"
    )

    # Prepare data for forecasting
    # 1. Prepare Monthly Data
    monthly_counts = (
        df.groupby(df["createdAt"].dt.to_period("M"))["id"]
        .count()
        .reset_index(name="y")
    )
    monthly_counts["ds"] = monthly_counts["createdAt"].dt.to_timestamp()
    # Fill missing months
    full_index = pd.date_range(
        start=monthly_counts["ds"].min(), end=monthly_counts["ds"].max(), freq="MS"
    )
    forecast_data = pd.DataFrame({"ds": full_index})
    forecast_data = forecast_data.merge(
        monthly_counts[["ds", "y"]], on="ds", how="left"
    )
    forecast_data["y"] = forecast_data["y"].fillna(0)

    # Handle prices
    def get_pass_price(row):
        if pd.notna(row.get("pass.passType.taxInclusivePrice")):
            return float(row["pass.passType.taxInclusivePrice"])
        elif pd.notna(row.get("membership.taxInclusivePrice")):
            return float(row["membership.taxInclusivePrice"])
        return 0.0

    df["pass_price"] = df.apply(get_pass_price, axis=1)

    # Calculate unique membership prices (once per membership_id)
    unique_memberships = df[df["membership_id"] != "None"][
        ["membership_id", "pass_price", "membership_name"]
    ].drop_duplicates("membership_id")
    print(
        f"Unique memberships: {unique_memberships[['membership_id', 'pass_price']].to_dict(orient='records')}"
    )

    # Set price to 0 for failed visits
    df.loc[df["status"] != "approved", "pass_price"] = 0.0

    # Dynamic time aggregation
    time_span = (df["createdAt"].max() - df["createdAt"].min()).days
    if time_span <= 30:
        df["time_period"] = df["date"]
        time_label = "date"
        period_name = "Daily"
    elif time_span <= 365:
        df["time_period"] = pd.PeriodIndex(df["createdAt"], freq="W").start_time
        time_label = "week"
        period_name = "Weekly"
    else:
        df["time_period"] = (
            df["createdAt"].dt.to_period("M").apply(lambda x: x.to_timestamp())
        )
        time_label = "month"
        period_name = "Monthly"

    # === Advanced Analysis ===
    summary = {
        "total_visits": int(len(df)),
        "failed_visits": int((df["status"] != "approved").sum()),
        "earliest_visit_date": df["createdAt"].min().isoformat(),
        "latest_visit_date": df["createdAt"].max().isoformat(),
        "average_monthly_visits": df.groupby("month").size().mean(),
        "unique_locations": int(df["location_name"].nunique()),
        "most_frequent_location": df["location_name"].mode().iloc[0]
        if not df["location_name"].empty
        else "N/A",
        "peak_visit_hour": int(df["hour"].mode().iloc[0])
        if not df["hour"].empty
        else 0,
        "unique_customers": int(df["customer_name"].nunique()),
        "most_frequent_customer": df["customer_name"].mode().iloc[0]
        if not df["customer_name"].empty
        else "N/A",
        "most_common_pass_type": df["pass_type"].mode().iloc[0]
        if not df["pass_type"].empty
        else "N/A",
    }

    # Membership Analysis
    membership_counts = (
        df[df["membership_id"] != "None"]
        .groupby(["membership_id", "membership_name"])
        .size()
        .reset_index(name="count")
    )
    summary["total_membership_entries"] = int(membership_counts["count"].sum())
    summary["membership_breakdown"] = {
        f"ID: {row['membership_id']} ({row['membership_name']})": int(row["count"])
        for _, row in membership_counts.iterrows()
    }

    # Revenue Analysis
    casual_revenue = df[df["membership_id"] == "None"]["pass_price"].sum()
    membership_revenue = unique_memberships["pass_price"].sum()
    revenue_by_type = {}
    for pass_type in df["pass_type"].unique():
        if pass_type == "No Pass":
            revenue_by_type[pass_type] = 0.0
        elif pass_type in unique_memberships["membership_name"].values:
            revenue_by_type[pass_type] = unique_memberships[
                unique_memberships["membership_name"] == pass_type
            ]["pass_price"].sum()
        else:
            revenue_by_type[pass_type] = df[
                (df["pass_type"] == pass_type) & (df["membership_id"] == "None")
            ]["pass_price"].sum()

    print(f"Casual revenue: ${casual_revenue}")
    print(f"Membership revenue: ${membership_revenue}")
    print(f"Revenue by type: {revenue_by_type}")

    summary["revenue_by_entry_type"] = {
        k: round(v, 2) for k, v in revenue_by_type.items()
    }
    summary["total_revenue"] = round(casual_revenue + membership_revenue, 2)

    # Customer Visit Summary
    customer_summary = (
        df.groupby("customer_name")
        .agg(
            visit_count=("id", "count"),
            locations_visited=("location_name", "nunique"),
            last_visit=("createdAt", "max"),
            pass_types_used=("pass_type", "nunique"),
            avg_days_between_visits=(
                "createdAt",
                lambda x: x.sort_values().diff().dt.days.mean() if len(x) > 1 else None,
            ),
        )
        .reset_index()
    )
    customer_summary["last_visit"] = customer_summary["last_visit"].apply(
        lambda x: x.isoformat()
    )
    customer_summary["avg_days_between_visits"] = customer_summary[
        "avg_days_between_visits"
    ].round(2)
    summary["customer_summary"] = customer_summary.to_dict(orient="records")

    # Trend Analysis
    visits_over_time = df.groupby("time_period").size().reset_index(name="visit_count")
    visits_over_time["time_period"] = visits_over_time["time_period"].apply(
        lambda x: x if isinstance(x, datetime) else pd.Timestamp(x)
    )
    visits_over_time["smoothed"] = (
        visits_over_time["visit_count"]
        .rolling(window=3, center=True, min_periods=1)
        .mean()
    )
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        range(len(visits_over_time)), visits_over_time["visit_count"]
    )
    summary["trend_slope"] = round(slope, 4)
    summary["trend_r_squared"] = round(r_value**2, 4)
    summary["trend_p_value"] = round(p_value, 4)
    trend_line = intercept + slope * np.arange(len(visits_over_time))

    # Seasonality Analysis
    if len(visits_over_time) > 24:
        if time_label == "date":
            # Assume weekly seasonality (e.g., visits peak every Saturday)
            period = 7
            seasonality_period_desc = "weekly (~every 7 days)"
        elif time_label == "week":
            # Assume quarterly seasonality
            period = 13  # 13 weeks ≈ 1 quarter
            seasonality_period_desc = "quarterly (~every 13 weeks)"
        elif time_label == "month":
            # Annual seasonality
            period = 12
            seasonality_period_desc = "yearly (~every 12 months)"
        else:
            period = 1  # fallback
            seasonality_period_desc = "unknown"
        decomposition = seasonal_decompose(
            visits_over_time["visit_count"], model="additive", period=period
        )
        seasonal_strength = round(
            np.var(decomposition.seasonal) / np.var(visits_over_time["visit_count"]), 4
        )

        seasonal = decomposition.seasonal
        trend = decomposition.trend.dropna()
        mean_trend = trend.mean()
        peak_to_trough = seasonal.max() - seasonal.min()
        seasonal_effect_pct = round((peak_to_trough / mean_trend) * 100, 2)
        summary["estimated_seasonal_effect_%"] = (
            f"~{seasonal_effect_pct}% between seasonal high and low"
        )
        summary["seasonal_strength"] = seasonal_strength
        summary["seasonality_period"] = seasonality_period_desc

    else:
        summary["seasonal_strength"] = "Insufficient data for seasonality analysis"
        summary["seasonality_period"] = "-"
        summary["estimated_seasonal_effect_%"] = "-"

    # Forecasting Analysis
    MINIMUM_FORECAST = 6
    if len(monthly_counts) >= MINIMUM_FORECAST:
        ts_data = forecast_data.set_index("ds")["y"]
        future_dates = pd.date_range(
            start=ts_data.index[-1] + pd.offsets.MonthBegin(0), periods=12, freq="MS"
        )

        def try_exponential_smoothing():
            model = ExponentialSmoothing(
                ts_data, trend="add", seasonal="add", seasonal_periods=12
            ).fit()
            forecast = model.forecast(12)
            return forecast

        def try_linear_regression():
            ts_data_reset = ts_data.reset_index()
            ts_data_reset["t"] = np.arange(len(ts_data_reset))
            reg = LinearRegression().fit(ts_data_reset[["t"]], ts_data_reset["y"])
            future_t = np.arange(len(ts_data_reset), len(ts_data_reset) + 12).reshape(
                -1, 1
            )
            forecast = reg.predict(future_t)
            return pd.Series(forecast, index=future_dates)

        def try_sarimax():
            model = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=len(future_dates))
            return pd.Series(forecast, index=future_dates)

        # Try Prophet if available and more than 12 months of data
        if len(monthly_counts) >= 12:
            print(f"Prophet failed: {e}. Falling back to Exponential Smoothing.")
            forecast_series = try_exponential_smoothing()
            forecast_df = pd.DataFrame({"ds": future_dates, "y": forecast_series})
        else:
            try:
                forecast_series = try_sarimax()
                forecast_df = pd.DataFrame({"ds": future_dates, "y": forecast_series})
            except Exception as e:
                print(f"SARIMAX failed: {e}. Falling back to Linear Regression.")
                forecast_series = try_linear_regression()
                forecast_df = pd.DataFrame({"ds": future_dates, "y": forecast_series})

        # Add forecast type
        forecast_df["type"] = "Forecast"
        forecast_df["yhat_lower"] = forecast_df["y"] * 0.9
        forecast_df["yhat_upper"] = forecast_df["y"] * 1.1

        # Combine with historical
        forecast_data["type"] = "Historical"
        combined_data = pd.concat(
            [
                forecast_data[["ds", "y", "type"]],
                forecast_df[["ds", "y", "type", "yhat_lower", "yhat_upper"]],
            ],
            ignore_index=True,
        )

        # Add summary metric
        summary["total_forecasted_visits_next_year"] = int(
            forecast_df["y"].sum().round()
        )

    else:
        print(
            f"⚠️ Insufficient data for forecasting (need at least {MINIMUM_FORECAST} months)."
        )
        forecast_data["type"] = "Historical"
        combined_data = forecast_data.copy()
        summary["total_forecasted_visits_next_year"] = "Insufficient data"

    figs = []
    # Plot 1: Key Statistics Table
    stats_data = [
        ["Total Visits", summary["total_visits"]],
        ["Failed Visits", summary["failed_visits"]],
        ["Total Revenue ($)", summary["total_revenue"]],
        [
            "Revenue by Entry Type",
            ", ".join(
                [f"{k}: ${v}" for k, v in summary["revenue_by_entry_type"].items()]
            ),
        ],
        ["Total Membership Entries", summary["total_membership_entries"]],
        ["Unique Customers", summary["unique_customers"]],
        ["Most Frequent Customer", summary["most_frequent_customer"]],
        ["Most Common Pass Type", summary["most_common_pass_type"]],
        ["Peak Visit Hour", f"{summary['peak_visit_hour']}:00"],
        ["Trend Slope", f"{summary['trend_slope']:.4f} increase {period_name}"],
        ["Trend R²", f"{summary['trend_r_squared']:.4f}"],
        [
            "Seasonal Strength",
            f"{summary['seasonal_strength']} {summary['seasonality_period']}",
        ],
        ["Forecasted Visits (Next Year)", summary["total_forecasted_visits_next_year"]],
    ]
    fig = go.Figure(
        data=go.Table(
            header=dict(values=["Metric", "Value"], align="left", font=dict(size=14)),
            cells=dict(
                values=[list(x) for x in zip(*stats_data)],
                align="left",
                font=dict(size=12),
            ),
        ),
        layout=dict(
            margin=dict(l=0, r=0, t=0, b=10),  # Reduce margins, small bottom margin
            height=400,
        ),
    )
    figs.append(fig)

    # Plot 1b: Membership stats
    membership_table_data = {
        "Membership ID": [],
        "Type": [],
        "Cost": [],
        "Visits": [],
        "Avg Cost/Visit": [],
        "Adj Avg (if active)": [],
    }

    now = datetime.now()

    for _, row in membership_counts.iterrows():
        mid = row["membership_id"]
        membership_row = df[df["membership_id"] == mid].iloc[0]

        cost = unique_memberships[unique_memberships["membership_id"] == mid][
            "pass_price"
        ].iloc[0]
        visits = row["count"]
        avg_cost = round(cost / visits, 2) if visits > 0 else 0.0

        start = pd.to_datetime(
            membership_row["membership.startEffectiveDate"]
        ).tz_localize(None)
        end = pd.to_datetime(membership_row["membership.endEffectiveDate"]).tz_localize(
            None
        )

        if end > now and (end - start).days > 0:
            total_days = (end - start).days
            used_days = max((now - start).days, 1)
            proportional_cost = cost * (used_days / total_days)
            adjusted_avg = round(proportional_cost / visits, 2) if visits > 0 else 0.0
            adj_text = f"${adjusted_avg} ({used_days}/{total_days} days)"
        else:
            adj_text = "N/A"

        membership_table_data["Membership ID"].append(mid)
        membership_table_data["Type"].append(row["membership_name"])
        membership_table_data["Cost"].append(f"${cost}")
        membership_table_data["Visits"].append(visits)
        membership_table_data["Avg Cost/Visit"].append(f"${avg_cost}")
        membership_table_data["Adj Avg (if active)"].append(adj_text)

    # Membership breakdown table
    fig = go.Figure(
        data=go.Table(
            header=dict(
                values=list(membership_table_data.keys()),
                align="left",
                font=dict(size=14),
            ),
            cells=dict(
                values=list(membership_table_data.values()),
                align="left",
                font=dict(size=12),
            ),
        ),
        layout=dict(
            margin=dict(l=0, r=0, t=0, b=10),  # Reduce margins, small bottom margin
            height=max(60 * len(membership_counts), 100),
        ),
    )
    figs.append(fig)

    # Plot 3: Visit Status Pie Chart
    status_counts = df["status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]
    fig = px.pie(
        status_counts,
        names="status",
        values="count",
        title="Visit Status Distribution",
        hole=0,  # set >0 for donut chart if you want
        color_discrete_sequence=["green", "red", "purple"],  # In order of appearance
    )
    # Customize text info to show percent and label
    fig.update_traces(textinfo="percent+label", textfont_size=12, showlegend=False)
    figs.append(fig)

    # Plot 4: Visits by Day of Week
    dow_counts = (
        df["day_of_week"]
        .value_counts()
        .reindex(
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
        )
        .reset_index()
    )
    dow_counts.columns = ["day", "count"]
    fig = px.bar(
        dow_counts,
        x="day",
        y="count",
        text="count",
        title="Visits by Day of Week",
        labels={"day": "Day of Week", "count": "Visit Count"},
        color="day",  # adds colors automatically
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    # Show counts on bars
    fig.update_traces(textposition="auto", showlegend=False)
    figs.append(fig)

    # Plot 5: Visits Over Time
    # Base line plot for visit counts
    fig = px.line(
        visits_over_time,
        x="time_period",
        y="visit_count",
        markers=True,
        title=f"{period_name} Visits Over Time",
        labels={"time_period": "Time Period", "visit_count": "Visit Count"},
    )

    # Update style for base line + markers
    fig.update_traces(
        line=dict(width=3, color="blue"),
        marker=dict(size=10),
        name=f"{period_name} Visits",
    )
    # Add smoothed trend line
    fig.add_trace(
        go.Scatter(
            x=visits_over_time["time_period"],
            y=visits_over_time["smoothed"],
            mode="lines",
            line=dict(width=2, color="red", dash="dash"),
            name="Smoothed Trend",
        )
    )
    # Add linear trend line
    fig.add_trace(
        go.Scatter(
            x=visits_over_time["time_period"],
            y=trend_line,
            mode="lines",
            line=dict(width=2, color="green", dash="dot"),
            name="Linear Trend",
        )
    )
    figs.append(fig)

    # Plot 6: Visits by Entry Type with Membership IDs
    # Prepare data grouped by pass_type and membership_id
    pass_type_counts = (
        df.groupby(["pass_type", "membership_id"]).size().reset_index(name="count")
    )
    # Replace missing membership_ids (or None) with string "None" for consistent coloring
    pass_type_counts["membership_id"] = (
        pass_type_counts["membership_id"].fillna("None").astype(str)
    )
    # Define a color sequence for membership IDs, with gray for "None"
    unique_ids = pass_type_counts["membership_id"].unique()
    color_seq = px.colors.qualitative.Plotly
    color_map = {mid: color_seq[i % len(color_seq)] for i, mid in enumerate(unique_ids)}
    color_map["None"] = "#808080"  # gray for None
    # Map colors to the membership_id column
    pass_type_counts["color"] = pass_type_counts["membership_id"].map(color_map)
    # Create the stacked bar chart
    fig = px.bar(
        pass_type_counts,
        x="pass_type",
        y="count",
        color="membership_id",
        color_discrete_map=color_map,
        text="count",
        title="Visits by Entry Type with Membership IDs",
        labels={
            "pass_type": "Pass Type",
            "count": "Visit Count",
            "membership_id": "Membership ID",
        },
        barmode="stack",
    )
    # Customize text position for better readability
    fig.update_traces(textposition="auto")
    figs.append(fig)

    # Plot 7: Visit Forecast
    fig = go.Figure()
    # Historical data
    fig.add_trace(
        go.Scatter(
            x=combined_data[combined_data["type"] == "Historical"]["ds"],
            y=combined_data[combined_data["type"] == "Historical"]["y"],
            mode="lines+markers",
            name="Historical Visits",
            line=dict(color="blue", width=3),
            marker=dict(size=8),
        )
    )
    # Forecasted data
    if len(monthly_counts) >= MINIMUM_FORECAST:
        fig.add_trace(
            go.Scatter(
                x=combined_data[combined_data["type"] == "Forecast"]["ds"],
                y=combined_data[combined_data["type"] == "Forecast"]["y"],
                mode="lines+markers",
                name="Forecasted Visits",
                line=dict(color="orange", width=3, dash="dash"),
                marker=dict(size=8),
            )
        )
        # Uncertainty intervals
        fig.add_trace(
            go.Scatter(
                x=combined_data[combined_data["type"] == "Forecast"]["ds"],
                y=combined_data[combined_data["type"] == "Forecast"]["yhat_upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=combined_data[combined_data["type"] == "Forecast"]["ds"],
                y=combined_data[combined_data["type"] == "Forecast"]["yhat_lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(255, 165, 0, 0.2)",
                name="Uncertainty Interval",
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        title="Historical and Forecasted Monthly Visits",
        xaxis_title="Date",
        yaxis_title="Visit Count",
        showlegend=True,
        height=500,
        margin=dict(l=50, r=50, t=80, b=50),
    )
    fig.update_xaxes(tickformat="%b %Y")
    figs.append(fig)

    return figs
