"""Suspected duplicate vendor master records, side by side."""

from __future__ import annotations

import streamlit as st
from common import client, inr, page_setup

user = page_setup("Vendor Duplicates", "👥")
st.caption("Fuzzy name/address similarity plus GSTIN, PAN and bank-account matches (SOP-VEN-001). "
           "The newer record is the suspect. GSTIN and bank details are masked unless you are admin.")

rows = client().duplicates()
if not rows:
    st.info("No suspected duplicates.")
    st.stop()

if user["role"] != "admin":
    from supplysight.copilot.guardrails import mask_value

    for r in rows:
        for k in ("gstin_a", "gstin_b", "bank_a", "bank_b"):
            r[k] = mask_value(r[k])

thr = st.slider("Minimum similarity", 0.70, 1.0, 0.72, 0.01)
rows = [r for r in rows if r["similarity"] >= thr]
st.caption(f"{len(rows)} pair(s)")
for r in rows:
    with st.container(border=True):
        st.markdown(f"**Similarity {r['similarity']:.0%}** · {r['match_on']}")
        a, b = st.columns(2)
        for col, side, label in ((a, "a", "Existing vendor"), (b, "b", "Suspect (newer) vendor")):
            with col:
                st.markdown(f"**{label}: {r['name_' + side]}** ({r['lifnr_' + side]})")
                st.markdown(f"- Address: {r['street_' + side]}, {r['city_' + side]}\n"
                            f"- GSTIN: `{r['gstin_' + side]}`\n- Bank: `{r['bank_' + side]}`\n"
                            f"- Created: {r['created_' + side]}")
        st.caption(f"POs placed with the suspect vendor: {r['pos_b']} · spend {inr(r['spend_b'] or 0)}")
