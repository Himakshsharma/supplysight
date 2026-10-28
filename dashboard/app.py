"""SupplySight dashboard entry point: ``streamlit run dashboard/app.py``."""

from __future__ import annotations

import streamlit as st

PAGES = [
    st.Page("overview.py", title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
    st.Page("pages/1_Anomaly_Review.py", title="Anomaly review", icon=":material/policy:", url_path="Anomaly_Review"),
    st.Page(
        "pages/2_Vendor_Duplicates.py", title="Vendor duplicates", icon=":material/group:", url_path="Vendor_Duplicates"
    ),
    st.Page(
        "pages/3_Demand_and_Reorder.py",
        title="Demand & reorder",
        icon=":material/trending_up:",
        url_path="Demand_and_Reorder",
    ),
    st.Page("pages/4_Copilot.py", title="Copilot", icon=":material/forum:", url_path="Copilot"),
    st.Page("pages/5_Model_Health.py", title="Model health", icon=":material/monitor_heart:", url_path="Model_Health"),
]

st.navigation(PAGES).run()
