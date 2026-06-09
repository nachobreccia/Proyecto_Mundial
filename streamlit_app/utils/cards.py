import streamlit as st

def metric_card(label, value, subtitle=None):
    subtitle_html = f"<div class='small-muted'>{subtitle}</div>" if subtitle else ""

    st.markdown(
        f'''
        <div class="custom-card">
            <div class="section-label">{label}</div>
            <div class="big-number">{value}</div>
            {subtitle_html}
        </div>
        ''',
        unsafe_allow_html=True
    )

def text_card(label, text):
    st.markdown(
        f'''
        <div class="custom-card">
            <div class="section-label">{label}</div>
            <div style="font-size:15px; line-height:1.6; color:#374151;">
                {text}
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

def list_card(label, items):
    if items is None:
        items = []

    if isinstance(items, str):
        try:
            import ast
            parsed = ast.literal_eval(items)
            items = parsed if isinstance(parsed, list) else [items]
        except Exception:
            items = [items]

    if len(items) == 0:
        items = ["No data available"]

    items_html = "".join([f"<li>{item}</li>" for item in items])

    st.markdown(
        f'''
        <div class="custom-card">
            <div class="section-label">{label}</div>
            <ul style="font-size:15px; line-height:1.7; color:#374151; margin-bottom:0;">
                {items_html}
            </ul>
        </div>
        ''',
        unsafe_allow_html=True
    )

def app_footer():
    st.divider()
    st.caption(
        "Developed by Juan Ignacio Breccia | Football Analytics · Scouting · Data Science | Powered by StatsBomb Open Data"
    )