# Military-specific configuration
MILITARY_BRANCHES = ["Army", "Navy", "Air Force", "Marines", "Coast Guard", "Space Force"]
MILITARY_RANKS = ["General", "Colonel", "Major", "Captain", "Lieutenant", "Sergeant", "Corporal"]
TACTICAL_NETWORKS = [
    "Tactical LAN", "SATCOM", "Radio Network", "Drone Command",
    "Weapons System", "Surveillance Feed"
]
MILITARY_INSTALLATIONS = [
    "Fort Bragg", "Naval Base San Diego", "MacDill AFB", "Camp Lejeune",
    "Pearl Harbor", "Buckley SFB", "NORAD", "Pentagon Network"
]

# Color theme
primary_color = "#FFD700"    # Gold
secondary_color = "#FFA500"  # Orange
background_color = "#FFFFE0" # Light Yellow
text_color = "#000000"       # Black
accent_color = "#8B4513"     # SaddleBrown

def app_css() -> str:
    return f"""
<style>
    .main {{
        background-color: {background_color};
        color: {text_color};
    }}
    .sidebar .sidebar-content {{
        background-color: {primary_color};
        color: {text_color};
    }}
    .metric-box {{
        background-color: {secondary_color};
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid {accent_color};
        color: {text_color};
    }}
    .threat-card {{
        background-color: #FFA07A;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #FF6347;
        color: {text_color};
    }}
    .agent-card {{
        background-color: #F0E68C;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #DAA520;
        color: {text_color};
    }}
    .quantum-card {{
        background-color: #E6E6FA;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #9370DB;
        color: {text_color};
    }}
    .qkd-card {{
        background-color: #F5DEB3;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #D2B48C;
        color: {text_color};
    }}
    .chat-card {{
        background-color: #FAFAD2;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #DAA520;
        color: {text_color};
    }}
    .stAlert {{
        background-color: {secondary_color} !important;
        color: {text_color};
    }}
    .st-bb {{ background-color: {secondary_color}; }}
    .st-at {{ background-color: {primary_color}; }}
    .css-1aumxhk {{
        background-color: {primary_color};
        background-image: none;
        color: {text_color};
    }}
    .scanning-animation {{
        border: 2px solid {accent_color};
        border-radius: 50%;
        animation: scanning 2s linear infinite;
        position: relative;
    }}
    @keyframes scanning {{
        0%   {{ transform: scale(0.8); opacity: 0.7; }}
        50%  {{ transform: scale(1.1); opacity: 1;   }}
        100% {{ transform: scale(0.8); opacity: 0.7; }}
    }}
    .agent-button {{
        background-color: {primary_color};
        border: 2px solid {accent_color};
        border-radius: 3px;
        padding: 5px 10px;
        margin: 5px;
        cursor: pointer;
        color: {text_color};
    }}
    .agent-button.active {{
        background-color: {accent_color};
        color: white;
        font-weight: bold;
    }}
    .chat-message {{
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        max-width: 80%;
    }}
    .user-message {{
        background-color: {primary_color};
        color: {text_color};
        margin-left: auto;
    }}
    .assistant-message {{
        background-color: {accent_color};
        color: white;
        margin-right: auto;
    }}
    .metric-button {{
        background-color: {primary_color};
        border: 2px solid {accent_color};
        border-radius: 3px;
        padding: 10px;
        margin: 5px;
        cursor: pointer;
        text-align: center;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        color: {text_color};
    }}
    .metric-button:hover {{
        background-color: {accent_color};
        color: white;
    }}
    .metric-button h3 {{
        font-size: 1.1rem;
        margin-bottom: 5px;
        color: {text_color};
    }}
    .metric-button p {{
        font-size: 1.4rem;
        font-weight: bold;
        margin: 5px 0;
        color: {text_color};
    }}
    .metric-button small {{ font-size: 0.8rem; color: #666; }}
    .header {{
        background-color: {primary_color};
        color: {text_color};
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }}
    .subheader {{
        background-color: {secondary_color};
        color: {text_color};
        padding: 10px;
        border-radius: 3px;
        margin: 10px 0;
    }}
    .filter-container {{
        background-color: {primary_color};
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }}
    .threat-table {{
        background-color: {background_color};
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid {accent_color};
    }}
</style>
"""