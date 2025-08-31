import hashlib
import streamlit as st
from quantum_utils import generate_bb84_key, create_qkd_circuit
from visualization import visualize_bb84_protocol, visualize_qkd_summary, display_circuit

def render(view_mode: str, qkd_enabled: bool):
    st.markdown(f'<div class="subheader"><h2>Quantum Key Distribution (QKD) - {view_mode}</h2></div>', unsafe_allow_html=True)
    if not qkd_enabled:
        st.warning("QKD Encryption is currently disabled. Enable it in the sidebar to see QKD operations.")
        return

    eve_present = st.checkbox("Simulate Eavesdropper (Adversary)", False, key="eve_present_check")
    qkd_data = generate_bb84_key(1024, eve_present)
    st.success("🔒 QKD Encryption is active and securing tactical communications")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("BB84 Protocol Execution")
        st.plotly_chart(visualize_bb84_protocol(qkd_data), use_container_width=True)
        st.subheader("Key Generation Metrics")
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Raw Key Length", f"{len(qkd_data['alice_bits'])} bits")
            st.metric("Sifted Key Length", f"{len(qkd_data['sifted_key'])} bits")
        with col_stat2:
            st.metric("Final Key Length", f"{len(qkd_data['final_key'])} bits")
            rate = (len(qkd_data['final_key'])/len(qkd_data['alice_bits'])*100) if len(qkd_data['alice_bits']) else 0
            st.metric("Key Rate", f"{rate:.1f}%")

        if qkd_data["eve_present"]:
            st.error(f"🕵️ Adversary detected! {len(qkd_data['error_positions'])} errors found")
        else:
            st.success("✅ No eavesdropper detected")

        if len(qkd_data['final_key']) >= 256:
            st.success(f"Final 256-bit Key: {qkd_data['hex_key']}")
        else:
            st.warning("Insufficient matching bits for full key - continuing protocol")

    with col2:
        st.subheader("Key Generation Process")
        st.plotly_chart(visualize_qkd_summary(qkd_data), use_container_width=True)
        st.subheader("QKD Circuit Diagram")
        qc = create_qkd_circuit(eve_present)
        display_circuit(qc)

        st.subheader("Message Authentication")
        message = st.text_input("Enter message to authenticate:", "Tactical alert: perimeter breach detected", key="message_auth_input")
        if st.button("Generate Quantum HMAC", key="hmac_button"):
            if qkd_data and len(qkd_data['final_key']) >= 256:
                key_bytes = bytes(''.join(map(str, qkd_data['final_key'])), 'utf-8')
                message_bytes = bytes(message, 'utf-8')
                hmac = hashlib.sha256(key_bytes + message_bytes).hexdigest()
                st.success("Message authenticated with quantum key:")
                st.code(hmac)
            else:
                st.error("Not enough key material for authentication")

def api_render(view_mode: str, qkd_enabled: bool):
    if not qkd_enabled:
        return {"status": "QKD Encryption is disabled"}
    
    # Simulate QKD key generation (replace with your actual logic)
    from data_generation import generate_bb84_key
    qkd_data = generate_bb84_key(1024, eve_present=False)
    
    return {
        "status": "QKD Encryption is active",
        "metrics": {
            "raw_key_length": len(qkd_data["alice_bits"]),
            "sifted_key_length": len(qkd_data["sifted_key"]),
            "final_key_length": len(qkd_data["final_key"]),
            "key_rate": f"{len(qkd_data['final_key'])/len(qkd_data['alice_bits'])*100:.1f}%" if len(qkd_data["alice_bits"]) else "0%",
            "error_rate": f"{qkd_data['error_rate']*100:.2f}%",
            "hex_key": qkd_data["hex_key"]
        },
        "summary": {
            "errors_detected": len(qkd_data["error_positions"]),
            "eve_present": qkd_data["eve_present"]
        }
    }