"""IBM Carbon Design styles for MemPulse (Gray 100 dark theme).

Generated from the /carbon-streamlit template and adapted:
  * THEME switched to Carbon Gray 100 tokens (PRD §7.1)
  * MemPulse-specific classes appended: pipeline strip, memory tier cards,
    context gauge, document ribbon, state diff, health tiles.

Override THEME values or pass a dict to get_carbon_css() to re-theme.
"""

THEME: dict[str, str] = {
    "primary": "#0f62fe",          # $interactive  (Blue 60)
    "primary_hover": "#0353e9",
    "primary_active": "#002d9c",
    "highlight": "#4589ff",        # $highlight    (Blue 40)
    "bg_primary": "#161616",       # $background   (Gray 100)
    "bg_secondary": "#262626",     # $layer-01     (Gray 90)
    "bg_layer2": "#353535",        # $layer-02     (Gray 80)
    "text_primary": "#f4f4f4",     # $text-01      (Gray 10)
    "text_secondary": "#c6c6c6",   # $text-02      (Gray 30)
    "text_helper": "#8d8d8d",
    "text_placeholder": "#6f6f6f",
    "border": "#393939",
    "success": "#42be65",
    "warning": "#f1c21b",
    "error": "#fa4d56",
    "overlap_bg": "#684600",       # overlap highlight (PRD §7.1)
    "overlap_fg": "#f1c21b",
    "font_sans": "'IBM Plex Sans', -apple-system, sans-serif",
    "font_mono": "'IBM Plex Mono', 'Courier New', monospace",
}


def get_carbon_css(theme: dict[str, str] | None = None) -> str:
    """Return a <style> block with Carbon Gray 100 styles + MemPulse widgets."""
    t = {**THEME, **(theme or {})}
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {{ font-family: {t["font_sans"]}; }}
    #MainMenu, footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
    .stMainBlockContainer {{ padding: 1rem 1.5rem !important; max-width: 100% !important; }}

    h1, h2, h3, h4 {{ font-family: {t["font_sans"]} !important; color: {t["text_primary"]} !important; letter-spacing: -0.01em; }}
    h1 {{ font-weight: 600 !important; }}
    .subtitle {{ font-size: 0.9rem; color: {t["text_secondary"]}; margin-top: -0.6rem; }}
    .stMarkdown {{ font-family: {t["font_sans"]} !important; color: {t["text_secondary"]}; line-height: 1.5; }}
    code, pre {{ font-family: {t["font_mono"]} !important; }}
    hr {{ border: none; border-top: 1px solid {t["border"]}; margin: 0.6rem 0; }}

    /* Buttons: Carbon square corners */
    .stButton > button, .stFormSubmitButton > button {{
        border-radius: 0 !important; font-family: {t["font_sans"]} !important;
        font-weight: 500 !important; font-size: 0.85rem !important;
    }}
    .stButton > button[kind="primary"] {{ background: {t["primary"]} !important; border: none !important; }}
    .stButton > button[kind="primary"]:hover {{ background: {t["primary_hover"]} !important; }}

    /* Inputs: Carbon underline */
    .stTextInput input, .stTextArea textarea {{
        border-radius: 0 !important; background: {t["bg_secondary"]} !important;
        border: none !important; border-bottom: 1px solid {t["text_helper"]} !important;
    }}
    .stTextInput input:focus, .stTextArea textarea:focus {{ border-bottom: 2px solid {t["primary"]} !important; box-shadow: none !important; }}

    section[data-testid="stSidebar"] {{ background-color: {t["bg_secondary"]} !important; border-right: 1px solid {t["border"]}; }}
    section[data-testid="stSidebar"] h2 {{
        font-size: 0.72rem !important; font-weight: 600 !important; color: {t["text_helper"]} !important;
        text-transform: uppercase !important; letter-spacing: 0.1em !important; margin: 0.6rem 0 0.2rem 0;
    }}

    /* ---------- MemPulse widgets ---------- */
    .panel-title {{
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
        color: {t["text_helper"]}; margin: 0.2rem 0 0.4rem 0;
    }}
    .badge {{
        display: inline-block; font-family: {t["font_mono"]}; font-size: 0.72rem; padding: 2px 8px;
        margin: 0 4px 4px 0; background: {t["bg_layer2"]}; color: {t["text_secondary"]};
    }}
    .badge.on {{ background: #0e6027; color: #a7f0ba; }}
    .badge.off {{ background: {t["bg_layer2"]}; color: {t["text_helper"]}; }}

    /* Pipeline strip */
    .pipe {{ display: flex; flex-wrap: wrap; align-items: center; gap: 4px; margin: 4px 0 8px 0; }}
    .pipe .grp {{ font-size: 0.68rem; color: {t["text_helper"]}; text-transform: uppercase; letter-spacing: 0.08em; margin: 0 4px 0 10px; }}
    .pipe .node {{
        font-size: 0.78rem; padding: 5px 10px; background: {t["bg_secondary"]}; color: {t["text_helper"]};
        border: 1px solid {t["border"]}; white-space: nowrap;
    }}
    .pipe .node.done {{ background: #1c2a44; color: {t["text_secondary"]}; border-color: #274a8a; }}
    .pipe .node.active {{
        background: {t["primary"]}; color: #fff; border-color: {t["highlight"]};
        box-shadow: 0 0 0 2px {t["highlight"]}, 0 0 14px {t["highlight"]};
    }}
    .pipe .arrow {{ color: {t["text_placeholder"]}; font-size: 0.75rem; }}

    /* Memory tier cards */
    .tier {{ background: {t["bg_secondary"]}; border-left: 3px solid var(--c); padding: 7px 10px; margin-bottom: 6px; }}
    .tier.pulse {{ animation: pulse 1.4s ease-out 2; }}
    @keyframes pulse {{ 0% {{ box-shadow: 0 0 0 0 rgba(66,190,101,.8); }} 100% {{ box-shadow: 0 0 0 10px rgba(66,190,101,0); }} }}
    .tier .hd {{ display: flex; justify-content: space-between; font-size: 0.78rem; color: {t["text_primary"]}; font-weight: 500; }}
    .tier .hd .n {{ font-family: {t["font_mono"]}; color: var(--c); white-space: nowrap; margin-left: 6px; }}
    .tier .bar {{ height: 4px; background: {t["bg_layer2"]}; margin: 5px 0; }}
    .tier .bar > div {{ height: 4px; background: var(--c); }}
    .tier .items {{ font-size: 0.72rem; color: {t["text_secondary"]}; line-height: 1.35; }}
    .tier .items .new {{ color: {t["success"]}; font-weight: 600; }}
    .tier .items .dim {{ color: {t["text_placeholder"]}; }}

    /* Context-window gauge */
    .gauge {{ display: flex; height: 22px; background: {t["bg_layer2"]}; margin: 4px 0; overflow: hidden; }}
    .gauge > div {{ height: 100%; }}
    .gauge-legend {{ font-size: 0.68rem; color: {t["text_helper"]}; }}
    .gauge-legend span {{ margin-right: 8px; white-space: nowrap; }}
    .gauge-legend i {{ display: inline-block; width: 8px; height: 8px; margin-right: 3px; }}
    .evicted {{ font-size: 0.72rem; color: {t["error"]}; }}

    /* Document ribbon */
    .ribbon {{
        font-family: {t["font_mono"]}; font-size: 0.72rem; line-height: 1.6; white-space: pre-wrap;
        background: {t["bg_secondary"]}; padding: 10px; max-height: 390px; overflow-y: auto; color: {t["text_secondary"]};
    }}
    .ribbon .ck {{ border-radius: 0; }}
    .ribbon .ov {{ background: {t["overlap_bg"]} !important; color: {t["overlap_fg"]}; }}
    .ribbon .needle {{ text-decoration: underline 2px #ffb000; text-underline-offset: 3px; color: #fff; }}
    .ribbon .cut {{ color: {t["error"]}; font-weight: 700; background: #520408; padding: 0 2px; }}
    .ribbon .tag {{ font-size: 0.62rem; color: #fff; background: {t["bg_layer2"]}; padding: 0 3px; margin-right: 2px; }}
    .ribbon .tag.hit {{ background: {t["primary"]}; }}

    /* Prompt blocks (assemble view) */
    .pblock {{ border-left: 3px solid var(--c); background: {t["bg_secondary"]}; padding: 6px 10px; margin-bottom: 4px; font-size: 0.74rem; }}
    .pblock.out {{ opacity: 0.45; border-left-style: dashed; text-decoration: line-through; }}
    .pblock .lbl {{ font-family: {t["font_mono"]}; font-size: 0.68rem; color: var(--c); }}

    /* State diff */
    .diff {{ font-family: {t["font_mono"]}; font-size: 0.72rem; background: {t["bg_secondary"]}; padding: 8px; }}
    .diff .add {{ color: {t["success"]}; }}
    .diff .chg {{ color: {t["warning"]}; }}
    .diff .rem {{ color: {t["error"]}; }}
    .diff .same {{ color: {t["text_placeholder"]}; }}

    /* Health tiles */
    .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(125px, 1fr)); gap: 6px; }}
    .tile {{ background: {t["bg_secondary"]}; padding: 8px 10px; border-top: 3px solid var(--c); }}
    .tile .k {{ font-size: 0.68rem; color: {t["text_helper"]}; text-transform: uppercase; letter-spacing: 0.06em; }}
    .tile .v {{ font-size: 1.25rem; font-weight: 600; color: {t["text_primary"]}; font-family: {t["font_mono"]}; }}
    .tile .s {{ font-size: 0.7rem; color: {t["text_secondary"]}; }}

    .lesson {{ background: #1c2a44; border-left: 3px solid {t["highlight"]}; padding: 8px 12px; font-size: 0.8rem; color: {t["text_primary"]}; margin: 4px 0 8px 0; }}
    .answer {{ background: {t["bg_secondary"]}; border-left: 3px solid {t["success"]}; padding: 10px 12px; font-size: 0.85rem; color: {t["text_primary"]}; white-space: pre-wrap; }}
    </style>
    """
