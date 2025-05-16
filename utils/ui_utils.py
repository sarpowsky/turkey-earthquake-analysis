# utils/ui_utils.py
# Path: /utils/ui_utils.py
import streamlit as st

def add_footer():
    """Add a personal footer to all pages"""
    st.markdown("""
    <div style="position: fixed; bottom: 0; width: 100%; background-color: #2e4053; 
    padding: 5px 0; text-align: center; font-size: 0.8rem; border-top: 1px solid #ddd;">
        <p style="color: #ffffff; margin: 0;">Developed by sarpowsky | Global AI Hub ML Bootcamp 2025 | 
        <a href="https://github.com/sarpowsky" target="_blank" style="color: #8de5ee;">GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)