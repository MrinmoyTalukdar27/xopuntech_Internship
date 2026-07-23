"""A compact Streamlit interface for the Amazon product recommender."""

# Standard-library helpers for file paths, safe HTML, and SVG image encoding.
from pathlib import Path
from urllib.parse import quote
import html

# Third-party libraries used for data handling, model loading, and the web UI.
import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Product Recommender", page_icon="◆", layout="wide")

# Resolve all project paths from this file, so the app works no matter where the
# `streamlit run` command is issued from.
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
DATA_PATH = PROJECT_DIR / "Datasets" / "Amazon_feature_dataset.csv"
MODEL_PATH = PROJECT_DIR / "Models" / "knn_model.pkl"
HOME_HERO_IMAGE_PATH = APP_DIR / "assets" / "home-page-hero.png"
PROJECT_PAGE_IMAGE_PATH = APP_DIR / "assets" / "project-page-visual.png"
# Inline SVG shown whenever an external product image URL is missing or blocked.
PLACEHOLDER_IMAGE = "data:image/svg+xml;utf8," + quote(
    "<svg xmlns='http://www.w3.org/2000/svg' width='96' height='96'>"
    "<rect width='100%' height='100%' rx='10' fill='#1e293b'/>"
    "<path d='M25 69l17-18 12 12 9-9 10 15H25z' fill='#64748b'/>"
    "<circle cx='37' cy='35' r='7' fill='#64748b'/>"
    "</svg>"
)

# Global CSS: gives Streamlit widgets and custom cards the dark professional theme.
st.markdown(
    """
    <style>
        .stApp { background: #0b1220; color: #e5e7eb; }
        .block-container { max-width: 1120px; padding: 2.4rem 2.4rem 3rem; margin-top: 1.5rem;
                           margin-bottom: 1.5rem; background: #0d1728; border: 1px solid #243047;
                           border-radius: 18px; box-shadow: 0 18px 45px rgba(0, 0, 0, .20); }
        h1, h2, h3, p, label, [data-testid="stCaptionContainer"] { color: #e5e7eb !important; }
        [data-testid="stTextInput"] input, [data-testid="stSelectbox"] > div > div {
            background: #111c30 !important; color: #f8fafc !important;
            border-color: #334155 !important;
        }
        [data-testid="stTextInput"] input::placeholder { color: #94a3b8 !important; opacity: 1 !important; }
        [data-baseweb="select"] * { color: #f8fafc !important; }
        .stButton > button {
            background: #2563eb; color: white; border: 0; border-radius: 8px;
            font-weight: 600; padding: 0.55rem 1.2rem;
        }
        .stButton > button:hover { background: #1d4ed8; color: white; }
        .hero { border-bottom: 1px solid #243047; padding-bottom: 1.5rem; margin-bottom: 1.6rem; }
        .eyebrow { color: #60a5fa; font-size: .75rem; font-weight: 700; letter-spacing: .12em; }
        .card { background: #111c30; border: 1px solid #243047; border-radius: 12px; padding: 1rem; }
        .product-name { color: #f8fafc; font-weight: 700; font-size: 1rem; }
        .meta { color: #94a3b8; font-size: .85rem; }
        .score { color: #60a5fa; font-weight: 700; font-size: .85rem; }
        .image-frame { width: 96px; height: 96px; display: flex; align-items: center; justify-content: center; }
        .image-frame img { max-width: 96px; max-height: 96px; object-fit: contain; border-radius: 10px; }
        .stat { background: #111c30; border: 1px solid #243047; border-radius: 12px; padding: 1rem; }
        .stat-label { color: #94a3b8; font-size: .78rem; text-transform: uppercase; letter-spacing: .07em; }
        .stat-value { color: #f8fafc; font-weight: 700; font-size: 1.55rem; margin-top: .25rem; }
        .home-hero { background: linear-gradient(125deg, #10192c 0%, #182a51 55%, #3730a3 100%);
                     border: 1px solid #293b65; border-radius: 20px; padding: 3rem; margin: .6rem 0 1.8rem; }
        .home-hero h1 { font-size: 2.7rem; line-height: 1.18; max-width: 850px; margin: .55rem 0 1.2rem; }
        .home-hero p { color: #cbd5e1 !important; max-width: 760px; font-size: 1.05rem; line-height: 1.7; }
        [data-testid="stImage"] img { border: 1px solid #293b65; border-radius: 20px; }
        .chip { display: inline-block; color: #93c5fd; background: #172554; border: 1px solid #334f83;
                border-radius: 99px; padding: .28rem .65rem; font-size: .8rem; margin: .2rem .35rem .2rem 0; }
        .step-title { color: #f8fafc; font-weight: 700; margin-bottom: .45rem; }
        .step-copy { color: #94a3b8; font-size: .88rem; line-height: 1.5; }
        .catalog-hero { background: linear-gradient(90deg, rgba(17, 28, 48, .98) 0%, rgba(17, 28, 48, .90) 48%,
                              rgba(17, 28, 48, .42) 100%),
                              url("/app/static/recommendations-header-background.png") right center / 52% auto no-repeat, #111c30;
                        border: 1px solid #243047; border-radius: 16px;
                        padding: 1.5rem 2rem; margin: .6rem 0 1.4rem; display: flex;
                        align-items: center; justify-content: space-between; gap: 1.5rem; }
        .catalog-hero h1 { margin: .35rem 0; font-size: 2rem; }
        .catalog-copy { flex: 1; }
        .product-tile { background: #111c30; border: 1px solid #243047; border-radius: 12px;
                        overflow: hidden; min-height: 365px; margin-bottom: 1rem; }
        .product-tile:hover { border-color: #3b82f6; }
        .product-tile-image { height: 175px; display: flex; justify-content: center; align-items: center;
                              background: #0f172a; padding: .75rem; }
        .product-tile-image img { max-height: 155px; max-width: 100%; object-fit: contain; }
        .product-tile-body { padding: 1rem; }
        .product-tile-name { color: #f8fafc; font-weight: 700; line-height: 1.4; height: 4.2em; overflow: hidden; }
        .product-tile-category { color: #94a3b8; font-size: .78rem; margin: .45rem 0; }
        .product-tile-price { color: #f8fafc; font-size: 1.05rem; font-weight: 700; margin-top: .5rem; }
        .product-tile-rating { color: #fbbf24; font-size: .85rem; }
        .product-tile-score { color: #60a5fa; font-size: .82rem; font-weight: 700; margin-top: .45rem; }
        .project-list { color: #cbd5e1; line-height: 1.75; }
        .project-list b { color: #f8fafc; }
        .section-label { color: #f8fafc; font-size: 1.35rem; font-weight: 700; margin: 2.1rem 0 1rem; }
        .section-badge { color: #a5b4fc; background: #1e1b4b; border-radius: 999px; padding: .25rem .65rem;
                         font-size: .78rem; margin-left: .5rem; vertical-align: middle; }
        .pipeline-number { color: #818cf8; font-size: .8rem; font-weight: 700; }
        section[data-testid="stSidebar"] { border-right: 1px solid #243047; }
        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            background: linear-gradient(rgba(7, 14, 28, .88), rgba(7, 14, 28, .95)),
                        url("/app/static/navigation-sidebar-background.png") center / cover no-repeat;
        }
        section[data-testid="stSidebar"] .stButton > button { background: #111c30; border: 1px solid #243047;
            color: #cbd5e1; text-align: left; width: 100%; margin-bottom: .35rem; padding: .6rem .75rem;
            font-size: .88rem; white-space: nowrap; }
        section[data-testid="stSidebar"] .stButton > button:hover { background: #172554; border-color: #3b82f6; }
        section[data-testid="stSidebar"] .stButton > button[kind="primary"] { background: #2563eb; color: #ffffff; }
        .side-brand { padding: .8rem .15rem 1.5rem; border-bottom: 1px solid #243047; margin-bottom: 1.2rem; }
        .side-brand-title { color: #f8fafc; font-weight: 700; font-size: 1.1rem; }
        .side-brand-sub { color: #94a3b8; font-size: .75rem; margin-top: .2rem; }
        header[data-testid="stHeader"] { background: #0b1220 !important; border-bottom: 1px solid #243047; }
        [data-testid="stSidebarCollapsedControl"], header[data-testid="stHeader"] button {
            background: #38bdf8 !important; border: 1px solid #7dd3fc !important; border-radius: 9px !important;
            min-width: 42px !important; min-height: 42px !important; margin: .35rem !important;
        }
        [data-testid="stSidebarCollapsedControl"]:hover, header[data-testid="stHeader"] button:hover {
            background: #0ea5e9 !important;
        }
        header[data-testid="stHeader"] svg, [data-testid="stSidebarCollapsedControl"] svg {
            fill: #0b1220 !important; color: #0b1220 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    """Load the product CSV once and prepare display-friendly numeric fields."""
    data = pd.read_csv(path).reset_index(drop=True)
    # Prices and rating counts were log-transformed during feature engineering.
    # Convert them back only for readable UI display; the model keeps its own data.
    for column in ("discount_price", "actual_price", "no_of_ratings"):
        if column in data:
            data[f"{column}_display"] = np.expm1(data[column]).clip(lower=0)
    data["ratings"] = pd.to_numeric(data.get("ratings", 0), errors="coerce").fillna(0)
    data["name"] = data["name"].fillna("Unnamed product").astype(str)
    return data


@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    """Load and cache the trained KNN model to avoid reloading it on every rerun."""
    return joblib.load(path)


def format_price(value) -> str:
    """Format a numeric price as Indian Rupees, handling missing values safely."""
    return "—" if pd.isna(value) else f"₹{value:,.0f}"


def product_image_html(value) -> str:
    """Render remote images with a local fallback when a URL is unavailable."""
    image_url = value if isinstance(value, str) and value.startswith("http") else PLACEHOLDER_IMAGE
    image_url = html.escape(image_url, quote=True)
    return (
        '<div class="image-frame"><img src="' + image_url + '" alt="Product image" '
        "onerror=\"this.onerror=null;this.src='" + PLACEHOLDER_IMAGE + "';\" /></div>"
    )


def product_tile_html(product: pd.Series) -> str:
    """Return a compact marketplace-style card for one recommendation."""
    image_url = product.get("image", "")
    image_url = image_url if isinstance(image_url, str) and image_url.startswith("http") else PLACEHOLDER_IMAGE
    image_url = html.escape(image_url, quote=True)
    name = html.escape(str(product["name"]))
    category = html.escape(str(product.get("sub_category", product.get("main_category", ""))).title())
    price = format_price(product.get("discount_price_display", np.nan))
    return (
        '<div class="product-tile">'
        '<div class="product-tile-image"><img src="' + image_url + '" alt="Product image" '
        "onerror=\"this.onerror=null;this.src='" + PLACEHOLDER_IMAGE + "';\" /></div>"
        '<div class="product-tile-body">'
        f'<div class="product-tile-name">{name}</div>'
        f'<div class="product-tile-category">{category}</div>'
        f'<div class="product-tile-rating">★ {product["ratings"]:.1f}</div>'
        f'<div class="product-tile-price">{price}</div>'
        f'<div class="product-tile-score">{product["score"] * 100:.1f}% match</div>'
        '</div></div>'
    )


def recommendations(data: pd.DataFrame, model, seed_index: int, count: int) -> pd.DataFrame:
    """Query the KNN model and return the requested number of similar products."""
    # The fitted model contains the hybrid matrix, so no duplicate feature file
    # needs to be loaded into memory.
    matrix = model._fit_X
    if len(data) != matrix.shape[0]:
        raise ValueError("The dataset and model were created from different product lists.")

    # Query nearest neighbours using the vector for the chosen product.
    seed_vector = matrix[seed_index]
    if getattr(seed_vector, "ndim", 2) == 1:
        seed_vector = seed_vector.reshape(1, -1)
    distances, indices = model.kneighbors(seed_vector, n_neighbors=min(count + 1, len(data)))

    # Remove the selected product itself and convert cosine distance to a score.
    mask = indices[0] != seed_index
    result = data.iloc[indices[0][mask][:count]].copy()
    result["score"] = np.clip(1 - distances[0][mask][:count], 0, 1)
    return result


# Validate required files before trying to read the large dataset/model.
if not DATA_PATH.exists() or not MODEL_PATH.exists():
    st.error("Required dataset or model file is missing.")
    st.stop()

# Load the large dataset and model. Streamlit caching avoids repeating this work.
with st.spinner("Loading product catalog and recommendation model..."):
    df = load_data(DATA_PATH)
    knn_model = load_model(MODEL_PATH)

if not hasattr(knn_model, "_fit_X"):
    st.error("The saved KNN model is incomplete. Export it again from the training notebook.")
    st.stop()

# Session state remembers the active page after every widget interaction reruns the app.
if "simple_page" not in st.session_state:
    st.session_state.simple_page = "Home"

# Sidebar navigation: each button selects a page and then reruns the interface.
with st.sidebar:
    st.markdown(
        """
        <div class="side-brand">
            <div class="eyebrow">PRODUCT DISCOVERY</div>
            <div class="side-brand-title">Recommender</div>
            <div class="side-brand-sub">Hybrid product matching</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    nav_labels = {
        "Home": "⌂  Home",
        "Recommendations": "✦  Recommendations",
        "Project": "◇  Project",
    }
    for item in ("Home", "Recommendations", "Project"):
        if st.button(
            nav_labels[item],
            key=f"nav_{item}",
            use_container_width=True,
            type="primary" if st.session_state.simple_page == item else "secondary",
        ):
            st.session_state.simple_page = item
            st.rerun()

# Read the active page selected by the sidebar navigation buttons.
page = st.session_state.simple_page

# ------------------------------- HOME PAGE ---------------------------------
if page == "Home":
    hero_text, hero_visual = st.columns([1.35, 0.65])
    with hero_text:
        st.markdown(
            """
            <div class="home-hero">
                <div class="eyebrow">HYBRID · CONTENT-BASED · COSINE KNN</div>
                <h1>Product recommendations built from what items <em>are</em>, not just what people clicked.</h1>
                <p>This recommender combines product names and categories with price, popularity, and rating signals
                in one hybrid vector, then finds the nearest products across the complete catalog.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hero_visual:
        if HOME_HERO_IMAGE_PATH.exists():
            st.image(str(HOME_HERO_IMAGE_PATH), use_container_width=True)
    first, second, third, fourth = st.columns(4)
    stats = [
        ("Catalog size", f"{len(df):,}"),
        ("Main categories", f"{df['main_category'].nunique():,}"),
        ("Sub-categories", f"{df['sub_category'].nunique():,}"),
        ("Feature dimensions", f"{knn_model._fit_X.shape[1]:,}"),
    ]
    for column, (label, value) in zip((first, second, third, fourth), stats):
        with column:
            st.markdown(
                f'<div class="stat"><div class="stat-label">{label}</div><div class="stat-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-label">Pipeline <span class="section-badge">4 stages</span></div>', unsafe_allow_html=True)
    step_one, step_two, step_three, step_four = st.columns(4)
    steps = [
        ("01", "Data cleaning", "Prepared product text, prices, ratings, and popularity fields for modelling."),
        ("02", "Exploratory analysis", "Reviewed category mix, price ranges, and rating distributions."),
        ("03", "Feature engineering", "Combined semantic text embeddings with scaled numeric shopping signals."),
        ("04", "Model training", "Used cosine K-Nearest Neighbours to retrieve the closest products."),
    ]
    for column, (number, title, copy) in zip((step_one, step_two, step_three, step_four), steps):
        with column:
            st.markdown(
                f'<div class="card"><div class="pipeline-number">{number}</div>'
                f'<div class="step-title">{title}</div><div class="step-copy">{copy}</div></div>',
                unsafe_allow_html=True,
            )
    st.stop()

# ----------------------------- PROJECT PAGE --------------------------------
if page == "Project":
    project_text, project_visual = st.columns([1.25, 0.75])
    with project_text:
        st.markdown(
            """
            <div class="home-hero">
                <div class="eyebrow">ABOUT THIS PROJECT</div>
                <h1>A faster way to explore a large product catalog.</h1>
                <p>This portfolio project demonstrates how machine learning can make product discovery more useful when shoppers do not know exactly what to search for.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with project_visual:
        if PROJECT_PAGE_IMAGE_PATH.exists():
            st.image(str(PROJECT_PAGE_IMAGE_PATH), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.markdown("### What this app helps with")
        st.markdown(
            '<div class="card"><div class="project-list"><b>Product discovery:</b> find alternatives to an item you already like<br>'
            '<b>Comparison:</b> identify similar products across a large catalog<br>'
            '<b>Exploration:</b> browse by category before narrowing a search</div></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("### Data scope")
        st.markdown(
            f'<div class="card"><div class="project-list"><b>Catalog records:</b> {len(df):,}<br>'
            f'<b>Product categories:</b> {df["main_category"].nunique():,}<br>'
            '<b>Available information:</b> title, category, price, rating, popularity, and image links</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("### Design considerations")
    st.write(
        "The app is intentionally designed as a decision-support tool, not a purchasing system. Similarity scores "
        "indicate how close two products are in the model's feature space; they do not guarantee availability, "
        "current price, product quality, or personal suitability."
    )
    st.stop()

# ------------------------- RECOMMENDATIONS PAGE ----------------------------
# The remaining code runs only when the Recommendations page is selected.
st.markdown(
    """
    <div class="catalog-hero">
        <div class="catalog-copy">
            <div class="eyebrow">PRODUCT DISCOVERY</div>
            <h1>Find products you will like</h1>
            <p>Search the catalog, select a product, and compare its closest alternatives.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Let the user narrow the product catalog by category and name.
filter_col, search_col = st.columns([1, 2])
with filter_col:
    categories = ["All categories"] + sorted(df["main_category"].dropna().unique().tolist())
    selected_category = st.selectbox("Category", categories)
with search_col:
    query = st.text_input("Search products", placeholder="Type at least 2 letters to search")

# Apply the optional category filter before building the product dropdown.
catalog = df if selected_category == "All categories" else df[df["main_category"] == selected_category]

# Keep the dropdown fast: show popular products by default, or up to 500 search matches.
if len(query.strip()) < 2:
    candidates = catalog.nlargest(500, "no_of_ratings_display")
    helper = "Showing 500 popular products. Type at least 2 letters to search."
else:
    candidates = catalog[catalog["name"].str.contains(query.strip(), case=False, regex=False, na=False)].head(500)
    helper = f"Showing {len(candidates):,} matching products (maximum 500)."

st.caption(helper)

if candidates.empty:
    st.info("No products found. Try a shorter search.")
    st.stop()

# Store row indices in the dropdown; the index is also the matching model row.
options = candidates.index.tolist()
selected_index = st.selectbox(
    "Select a product",
    options,
    format_func=lambda index: df.at[index, "name"][:110],
)
number = st.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10,
    step=1,
    help="Choose how many similar products to display, then click Get recommendations.",
)

# Run model inference only when the user clicks the action button.
if st.button("Get recommendations", type="primary"):
    with st.spinner("Finding similar products..."):
        st.session_state["simple_recommendations"] = recommendations(df, knn_model, selected_index, number)
        st.session_state["simple_request"] = (selected_index, number)

# Render results only when they belong to the currently selected product and count.
result = st.session_state.get("simple_recommendations")
if result is not None and st.session_state.get("simple_request") == (selected_index, number):
    st.markdown(f"### Recommended products ({len(result)})")
    for start in range(0, len(result), 3):
        columns = st.columns(3)
        for column, (_, product) in zip(columns, result.iloc[start : start + 3].iterrows()):
            with column:
                st.markdown(product_tile_html(product), unsafe_allow_html=True)
