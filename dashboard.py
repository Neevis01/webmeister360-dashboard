"""Chef-first kitchen waste dashboard (native Streamlit, no CSS).

Run:  python -m streamlit run dashboard.py

Data source is chosen automatically:
  - .streamlit/secrets.toml has a REAL SUPABASE_URL and SUPABASE_KEY -> Supabase
  - otherwise                                                      -> waste_log.csv
Optional in secrets.toml:  SUPABASE_TABLE = "waste_log"  (default name)
CSV/table columns: created_at, weight_g, food_type, category
"""

from pathlib import Path
import math
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

KITCHEN_TIMEZONE = "Europe/Berlin"  # Change if the kitchen is in another timezone.
CSV_PATH = Path(__file__).resolve().with_name("waste_log.csv")
COLUMNS = ["created_at", "weight_g", "food_type", "category"]
LABELS = {
    "preparation_waste": "Preparation waste",
    "spoiled_stock": "Spoiled stock",
    "plate_leftovers": "Plate leftovers",
    "unmapped": "Unmapped",
}
SHORT_LABELS = {
    "preparation_waste": "Prep",
    "spoiled_stock": "Spoiled",
    "plate_leftovers": "Leftovers",
    "unmapped": "Unmapped",
}
VALID_CATEGORIES = tuple(key for key in LABELS if key != "unmapped")
UNKNOWN_FOODS = {"", "mixed", "unknown", "nan", "none", "n/a"}


def fmt_kg(grams):
    """The ONE place kilograms are formatted, so every number on screen matches."""
    return f"{grams / 1000:.2f} kg"


def display_food(name):
    if name in UNKNOWN_FOODS:
        return "Mixed/Unidentifiable"
    return name.capitalize()


def get_secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return default


def supabase_configured():
    url = str(get_secret("SUPABASE_URL", ""))
    key = str(get_secret("SUPABASE_KEY", ""))
    placeholder = ("your-url", "your-anon-key", "YOUR_PROJECT", "YOUR_READ_KEY")
    return (
        url.startswith("https://")
        and bool(key)
        and not any(token in url + key for token in placeholder)
    )


def data_source():
    # Optional override in secrets.toml:  DATA_SOURCE = "csv"  forces local test data.
    if str(get_secret("DATA_SOURCE", "")).lower() == "csv":
        return "csv"
    return "supabase" if supabase_configured() else "csv"


def explain_error(error):
    """Plain-language cause and fix for the most common data-loading failures."""
    text = str(error)
    low = text.lower()
    if isinstance(error, ModuleNotFoundError) or "no module named" in low:
        return "The Supabase package is missing. In the terminal run: python -m pip install supabase"
    if "missing columns" in low:
        return "Column names do not match. Required: created_at, weight_g, food_type, category. " + text
    if "pgrst205" in low or "could not find the table" in low:
        return 'Supabase table not found. Set the exact table name: SUPABASE_TABLE = "..." in secrets.toml.'
    if "42501" in low or "permission denied" in low or "row-level security" in low:
        return "Supabase refuses to read this table. Ask your teammate to add a SELECT policy for the anon role."
    if "42703" in low or ("column" in low and "does not exist" in low):
        return "A column name does not exist in the Supabase table. Check created_at, weight_g, food_type, category."
    if "invalid api key" in low or "jwt" in low or "apikey" in low:
        return "Supabase rejected the key. Check SUPABASE_KEY (use the anon/publishable key)."
    if any(t in low for t in ("getaddrinfo", "name or service", "connecterror", "timed out", "connection")):
        return "Cannot reach Supabase. Check SUPABASE_URL and the internet connection."
    return "Unexpected data error. See the technical message below."


def load_data(start_utc, end_utc):
    if data_source() == "supabase":
        return fetch_supabase_data(start_utc, end_utc)
    return pd.read_csv(CSV_PATH)


@st.cache_resource
def init_supabase():
    from supabase import create_client

    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


def fetch_supabase_data(start_utc, end_utc):
    client = init_supabase()
    table_name = get_secret("SUPABASE_TABLE", "waste_log")
    records = []
    offset = 0
    page_size = 500
    while True:
        response = (
            client.table(table_name)
            .select(",".join(COLUMNS))
            .gte("created_at", start_utc.isoformat())
            .lte("created_at", end_utc.isoformat())
            .order("created_at")
            .order("food_type")
            .order("category")
            .order("weight_g")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        page = response.data
        if page is None:
            raise RuntimeError("Supabase returned no data payload.")
        if not page:
            break
        records.extend(page)
        offset += len(page)
    return pd.DataFrame(records, columns=COLUMNS)


# Cloud notes:
# - Reads only; assumes the public schema and the configured table's four fields.
# - Pages avoid silently truncating a busy day's events at an API row limit.
# - The time cutoff is fixed for each refresh. Ordering all selected fields
#   resolves timestamp ties without assuming an additional id column exists.
# - Intended for an append-only event log; concurrent edits/deletions during
#   paging require a database-side snapshot/aggregation for strict consistency.
# - An empty cloud result may also mean the read policy hides rows: verify
#   access with a known inserted event before demonstrating to the jury.


def parse_created_at(value):
    try:
        value = pd.Timestamp(value)
        if pd.isna(value):
            return pd.NaT
        if value.tzinfo is None:
            value = value.tz_localize(
                KITCHEN_TIMEZONE, ambiguous="NaT", nonexistent="NaT"
            )
        return value.tz_convert("UTC")
    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def prepare_data(raw):
    """Preserve rows; normalize values and flag invalid input explicitly."""
    df = raw.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower()
    if df.columns.duplicated().any():
        raise ValueError("Duplicate CSV column names.")
    missing = [column for column in COLUMNS if column not in df.columns]
    if missing:
        message = "Missing columns: " + ", ".join(missing)
        if "created_at" in missing and "timestamp" in df.columns:
            message += "; rename timestamp to created_at in the CSV."
        raise ValueError(message)
    df["created_at"] = pd.to_datetime(
        df["created_at"].map(parse_created_at), utc=True, errors="coerce"
    )
    df["weight_g"] = pd.to_numeric(df["weight_g"], errors="coerce")
    df["category"] = df["category"].fillna("").astype(str).str.strip().str.lower()
    df["food_type"] = (
        df["food_type"].fillna("").astype(str)
        .map(lambda value: re.sub(r"\s+", " ", value).strip().lower())
    )
    df["valid_weight"] = df["weight_g"].map(
        lambda value: pd.notna(value) and math.isfinite(value) and value >= 0
    ).astype(bool)
    df["unmapped"] = ~df["category"].isin(VALID_CATEGORIES)
    df["stream"] = df["category"].where(~df["unmapped"], "unmapped")
    return df


def pick_priority(today):
    totals = (
        today.groupby("stream")["weight_g"].sum()
        .reindex(list(LABELS), fill_value=0.0)
    )
    if totals.max() <= 0:
        return totals, None, None
    category = totals.idxmax()
    candidates = today.loc[
        today["stream"].eq(category)
        & ~today["food_type"].isin(UNKNOWN_FOODS)
        & today["weight_g"].gt(0)
    ]
    # Heaviest named food in the winning stream; mixed only supplies a fallback.
    foods = candidates.groupby("food_type", sort=True)["weight_g"].sum()
    food = None if foods.empty else foods.idxmax()
    return totals, category, food


def action_text(category, food):
    """One chef-style sentence. Falls back to a general phrase for mixed/unknown food."""
    if category == "preparation_waste":
        if food:
            return f"Before prep, have one cook trim a batch of {food} with the sharp knife and keep every usable edge."
        return "Before prep, run a 10-minute knife-skills and trimming check at every station."
    if category == "spoiled_stock":
        if food:
            return f"Before service, do a first-in-first-out audit of the fridge and use the oldest {food} first."
        return "Before service, do a first-in-first-out audit of every fridge and use the oldest stock first."
    if category == "plate_leftovers":
        if food:
            return f"Serve {food} in a 15% smaller default portion and offer a free top-up."
        return "Review the standard plate portion sizes and trial 15% smaller defaults."
    if category == "unmapped":
        return "Fix the unmapped category labels before choosing a kitchen action."
    return "Log a disposal to receive an action."


CO2_KG_PER_KG = 2.5  # Rough average kg CO2e per kg of food waste (estimate, not measured).


def playbook_markdown(category):
    """Three specific tips for the winning category, shown inside a collapsed expander."""
    if category == "preparation_waste":
        return (
            "1. **Knife-skills training:** 10 minutes at each station before prep. "
            "Show the trim line once; thick cuts at the ends are where the kilos go.\n"
            "2. **Repurpose offcuts into stock:** peels, ends and trimmings go into a "
            "labelled stock container instead of the bin. Free flavour, zero extra cost.\n"
            "3. **Audit peeler blade sharpness:** a dull peeler takes twice the skin. "
            "Test every blade tonight and replace the worn ones."
        )
    if category == "spoiled_stock":
        return (
            "1. **Strict FIFO fridge organization:** oldest stock at the front with a "
            "date label, newest at the back. Nothing is opened before the older one is used.\n"
            "2. **Daily temperature audits:** check and log every fridge at open and "
            "close against your HACCP limits. A drifting fridge spoils stock silently.\n"
            "3. **Tighter purchasing cycles:** order smaller amounts more often, based on "
            "the last 3 days of real usage instead of gut feeling."
        )
    if category == "plate_leftovers":
        return (
            "1. **Standardize scoop sizes:** one colour-coded scoop per side dish, "
            "so every plate gets the same portion no matter who plates it.\n"
            "2. **Menu engineering:** find the side dishes that come back most often "
            "uneaten. Rework them or take them off the menu.\n"
            "3. **Smaller default portions with free top-ups:** guests who want more "
            "ask for it, and everyone else stops leaving food on the plate."
        )
    return None


CATEGORY_COLORS = {  # Colour-blind-safe palette, readable on light and dark themes.
    "preparation_waste": "#56B4E9",
    "spoiled_stock": "#CC79A7",
    "plate_leftovers": "#009E73",
    "unmapped": "#8A8F98",
}
ACCENT = "#E69F00"
MIXED_GREY = "#8A8F98"
TOP_FOODS = 7


def build_donut(totals, winner):
    """Overview donut: share of each category in total weight."""
    shown = totals[totals > 0]
    frame = pd.DataFrame({
        "key": shown.index,
        "Category": [LABELS[key] for key in shown.index],
        "kg": (shown.values / 1000),
    })
    fig = px.pie(
        frame, names="Category", values="kg", hole=0.45,
        color="Category",
        color_discrete_map={LABELS[k]: CATEGORY_COLORS[k] for k in LABELS},
    )
    fig.update_traces(
        sort=False,
        pull=[0.06 if key == winner else 0 for key in frame["key"]],
        texttemplate="%{percent:.0%}",
        textposition="inside",
        textfont_size=20,
        hovertemplate="%{label}: %{value:.2f} kg (%{percent})<extra></extra>",
    )
    fig.update_layout(
        height=300,
        font=dict(size=16),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5),
        annotations=[dict(
            text=f"<b>{fmt_kg(totals.sum())}</b>", x=0.5, y=0.5,
            showarrow=False, font=dict(size=26),
        )],
    )
    return fig


def food_breakdown(today, category):
    """Weight per food_type inside one category, heaviest first (top N + Other)."""
    stream = today.loc[today["stream"].eq(category)]
    foods = stream.groupby("food_type")["weight_g"].sum().sort_values(ascending=False)
    foods = foods[foods > 0]
    if len(foods) > TOP_FOODS:
        other = foods.iloc[TOP_FOODS:].sum()
        foods = pd.concat([foods.iloc[:TOP_FOODS], pd.Series({"other": other})])
    return foods


def build_food_bars(foods, category):
    """Drill-down: horizontal bars for the foods of one category."""
    total = float(foods.sum())
    names = ["Other foods" if name == "other" else display_food(name) for name in foods.index]
    top_named = next((n for n in foods.index if n not in UNKNOWN_FOODS and n != "other"), None)
    colors = [
        ACCENT if n == top_named else MIXED_GREY if n in UNKNOWN_FOODS or n == "other"
        else CATEGORY_COLORS[category]
        for n in foods.index
    ]
    fig = go.Figure(go.Bar(
        x=foods.values / 1000, y=names, orientation="h", marker_color=colors,
        text=[f"{v / 1000:.2f} kg · {v / total:.0%}" for v in foods.values],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.3f} kg<extra></extra>",
    ))
    fig.update_layout(
        height=max(200, 20 + 42 * len(foods)),
        font=dict(size=16),
        margin=dict(l=0, r=110, t=5, b=0),
        showlegend=False, bargap=0.4,
        xaxis=dict(visible=False, range=[0, float(foods.max()) / 1000 * 1.5],
                   fixedrange=True),
        yaxis=dict(title=None, autorange="reversed", fixedrange=True, automargin=True),
    )
    return fig


def render_dashboard():
    heading, refresh = st.columns([4, 1])
    heading.title("Kitchen waste")
    refresh.button("Refresh", use_container_width=True)

    now = pd.Timestamp.now(tz="UTC")
    try:
        midnight_utc = now.tz_convert(KITCHEN_TIMEZONE).normalize().tz_convert("UTC")
        start_utc = min(midnight_utc, now - pd.Timedelta(hours=24))
        raw = load_data(start_utc, now)
        df = prepare_data(raw)
    except FileNotFoundError:
        st.error("Put waste_log.csv beside dashboard.py.")
        return
    except pd.errors.EmptyDataError:
        st.info("Waiting for disposal data.")
        return
    except Exception as error:
        # Never fall back to mock data after a cloud failure.
        st.error(explain_error(error))
        st.caption(f"Source: {data_source()}")
        st.code(f"{type(error).__name__}: {error}")
        return

    if len(raw) == 0 and data_source() == "supabase":
        st.warning("Supabase returned 0 rows for the last 24 hours. Check that data exists and that the anon key may read it.")

    # Rare operational warnings are retained; routine explainer text is gone.
    if df["unmapped"].any():
        st.warning(f"{int(df['unmapped'].sum())} rows have unmapped categories; inspect the raw log.")
    invalid = (
        ~df["valid_weight"] | df["created_at"].isna() | df["created_at"].gt(now)
    )
    if invalid.any():
        st.warning(f"{int(invalid.sum())} invalid or future-dated rows excluded; inspect the raw log.")

    valid = df.loc[~invalid]
    # Rolling 24 hours: never resets to a near-empty screen at midnight.
    today = valid.loc[valid["created_at"].ge(now - pd.Timedelta(hours=24))].copy()
    period = "Last 24 hours"
    totals, winner, food = pick_priority(today)
    source = "Live (Supabase)" if data_source() == "supabase" else "Test data (CSV)"
    st.caption(f"{period} · {source}")

    waste, count, category, co2 = st.columns(4)
    waste.metric("Total Waste", fmt_kg(today["weight_g"].sum()))
    count.metric("Disposals", f"{len(today):,}")
    category.metric(
        "Largest Category", SHORT_LABELS[winner] if winner else "—",
        help=LABELS[winner] if winner else "No waste in this period.",
    )

    co2.metric(
        "Est. CO2 Footprint",
        f"{today['weight_g'].sum() / 1000 * CO2_KG_PER_KG:.1f} kg CO2e",
        help=f"Estimate: total waste in kg x {CO2_KG_PER_KG} kg CO2e per kg of food waste.",
    )

    # Native Markdown heading inside the native alert: large text, no HTML/CSS.
    st.warning(f"## {action_text(winner, food)}", icon="⚡")

    playbook = playbook_markdown(winner)
    if playbook:
        with st.expander(f"💡 Tactical Playbook: {LABELS[winner]}"):
            st.markdown(playbook)
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Where the waste comes from")
        if totals.sum() > 0:
            st.plotly_chart(
                build_donut(totals, winner), theme="streamlit",
                use_container_width=True, key="donut",
                config={"displayModeBar": False, "responsive": True},
            )
        else:
            st.info("No waste logged in this period.")

    with right:
        st.subheader("What is inside?")
        options = list(VALID_CATEGORIES)
        choice = st.radio(
            "Category", options,
            index=options.index(winner) if winner in options else 0,
            format_func=lambda key: SHORT_LABELS[key],
            horizontal=True, key="drill_category", label_visibility="collapsed",
        )
        foods = food_breakdown(today, choice)
        if foods.empty:
            st.info(f"No {LABELS[choice].lower()} in this period.")
        else:
            named = [n for n in foods.index if n not in UNKNOWN_FOODS and n != "other"]
            if named:
                top = named[0]
                st.markdown(f"**{top.capitalize()}** is {foods[top] / foods.sum():.0%} of {LABELS[choice].lower()}.")
            else:
                st.markdown("Only mixed/unidentifiable waste logged here.")
            st.plotly_chart(
                build_food_bars(foods, choice), theme="streamlit",
                use_container_width=True, key=f"foods_{choice}",
                config={"displayModeBar": False, "responsive": True},
            )

    with st.expander("Raw data (newest first)"):
        log = today.sort_values("created_at", ascending=False).copy()
        log["time"] = log["created_at"].dt.tz_convert(KITCHEN_TIMEZONE).dt.strftime("%d %b %H:%M")
        log["category"] = log["category"].map(lambda c: LABELS.get(c, c))
        st.dataframe(
            log[["time", "category", "food_type", "weight_g"]],
            hide_index=True, use_container_width=True,
        )


def main():
    st.set_page_config(page_title="Kitchen waste", page_icon="🍽️", layout="wide")
    try:
        render_dashboard()
    except Exception as error:  # Never leave the page blank: show the real error.
        st.exception(error)


if __name__ == "__main__":
    main()