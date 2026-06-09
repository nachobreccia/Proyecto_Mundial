import streamlit as st

def apply_global_style():
    st.markdown(
        '''
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1320px;
        }

        h1 {
            font-weight: 800;
            letter-spacing: -0.04em;
        }

        .hero-card {
            background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
            border: 1px solid #E5E7EB;
            border-radius: 24px;
            padding: 28px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.05);
            margin-bottom: 24px;
        }

        .custom-card {
            background: white;
            border: 1px solid #E5E7EB;
            border-radius: 18px;
            padding: 20px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.05);
            margin-bottom: 18px;
        }

        .section-label {
            color: #6B7280;
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .big-number {
            font-size: 34px;
            font-weight: 800;
            letter-spacing: -0.04em;
        }

        .small-muted {
            color: #6B7280;
            font-size: 14px;
        }

        section[data-testid="stSidebar"] {
            background-color: #F3F4F6;
            border-right: 1px solid #E5E7EB;
        }

        .stTabs [data-baseweb="tab"] {
            font-weight: 700;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
        }
        </style>
        ''',
        unsafe_allow_html=True
    )