import streamlit as st

def military_authentication():
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔐 Command Authentication")
    auth_status = st.sidebar.selectbox("Authentication Level",
                                       ["UNCLASSIFIED", "CONFIDENTIAL", "SECRET", "TOP SECRET"],
                                       key="auth_level_select")
    if auth_status == "TOP SECRET":
        st.sidebar.success("✅ TOP SECRET clearance granted")
        st.sidebar.info("Full system access enabled")
        return 4
    elif auth_status == "SECRET":
        st.sidebar.warning("⚠️ SECRET clearance granted")
        st.sidebar.info("Limited system access")
        return 3
    elif auth_status == "CONFIDENTIAL":
        st.sidebar.warning("⚠️ CONFIDENTIAL clearance granted")
        st.sidebar.info("Restricted system access")
        return 2
    else:
        st.sidebar.error("🔒 UNCLASSIFIED access only")
        st.sidebar.info("Basic dashboard view only")
        return 1