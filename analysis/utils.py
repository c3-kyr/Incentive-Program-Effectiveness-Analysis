"""
Shared utilities for Incentive Program Effectiveness Analysis
AmEx-inspired styling, data loading, and statistical helpers.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path

# ── Project Paths ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── AmEx-Inspired Color Palette ──────────────────────────────
AMEX_PALETTE = {
    'primary_blue': '#006FCF',
    'dark_blue': '#00175A',
    'light_blue': '#ECEDFF',
    'accent_teal': '#00A5B5',
    'accent_green': '#00A859',
    'warning_amber': '#F5A623',
    'alert_red': '#C8102E',
    'dark_bg': '#1A1A2E',
    'card_bg': '#16213E',
    'text_light': '#E8E8E8',
    'text_muted': '#8892B0',
    'grid': '#2D3A5C',
}

TEAM_COLORS = [
    AMEX_PALETTE['primary_blue'], AMEX_PALETTE['accent_teal'],
    AMEX_PALETTE['accent_green'], AMEX_PALETTE['warning_amber']
]
TIER_COLORS = [
    AMEX_PALETTE['alert_red'], AMEX_PALETTE['warning_amber'],
    AMEX_PALETTE['accent_teal'], AMEX_PALETTE['primary_blue']
]
CLUSTER_COLORS = [
    AMEX_PALETTE['alert_red'], AMEX_PALETTE['warning_amber'],
    AMEX_PALETTE['accent_green'], AMEX_PALETTE['primary_blue']
]

def setup_plot_style():
    """Configure matplotlib for publication-quality AmEx-themed dark plots."""
    plt.style.use('dark_background')
    mpl.rcParams.update({
        'figure.facecolor': AMEX_PALETTE['dark_bg'],
        'axes.facecolor': AMEX_PALETTE['card_bg'],
        'axes.edgecolor': AMEX_PALETTE['grid'],
        'axes.labelcolor': AMEX_PALETTE['text_light'],
        'text.color': AMEX_PALETTE['text_light'],
        'xtick.color': AMEX_PALETTE['text_muted'],
        'ytick.color': AMEX_PALETTE['text_muted'],
        'grid.color': AMEX_PALETTE['grid'],
        'grid.alpha': 0.3,
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'figure.titlesize': 16,
        'figure.titleweight': 'bold',
        'legend.facecolor': AMEX_PALETTE['card_bg'],
        'legend.edgecolor': AMEX_PALETTE['grid'],
    })

def load_data():
    """Load data from CSVs. Generate if not found."""
    agents_path = DATA_DIR / 'agents.csv'
    perf_path = DATA_DIR / 'monthly_performance.csv'
    if not agents_path.exists() or not perf_path.exists():
        print('[INFO] Data files not found. Generating...')
        import subprocess
        gen_script = DATA_DIR / 'generate_data.py'
        subprocess.run([sys.executable, str(gen_script)], check=True)
    agents = pd.read_csv(agents_path)
    performance = pd.read_csv(perf_path)
    df = performance.merge(agents, on='agent_id', how='left')
    return df, agents, performance

def save_figure(fig, filename, dpi=200):
    """Save figure to output directory."""
    filepath = OUTPUT_DIR / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    print(f'[SAVED] {filepath}')
    plt.close(fig)

def format_pvalue(p):
    """Format p-value for display."""
    if p < 0.001:
        return 'p < 0.001'
    elif p < 0.01:
        return f'p = {p:.3f}'
    elif p < 0.05:
        return f'p = {p:.3f}'
    else:
        return f'p = {p:.3f} (n.s.)'

def add_watermark(fig, text='Incentive Program Effectiveness Analysis'):
    """Add subtle watermark to figure."""
    fig.text(0.99, 0.01, text, fontsize=8, color=AMEX_PALETTE['text_muted'],
             ha='right', va='bottom', alpha=0.5, style='italic')
