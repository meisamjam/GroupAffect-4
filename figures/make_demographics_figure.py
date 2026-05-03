"""
Generate paper/figure/dataset_stats/demographics_overview.pdf (and .png)

Two-panel figure:
  Left  — individual-level: age distribution (histogram+KDE) + sex/education/
           English stacked in a tight sub-layout
  Right — group-level composition: per-group age range strips + sex mix bars
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

# ── Data ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(
    os.path.join(os.path.dirname(__file__), '..', '..',
                 'results', 'personality', 'personality_summary.tsv'),
    sep='\t'
)
p = df.copy()

# ── Palette ───────────────────────────────────────────────────────────────────
C = dict(
    female  = '#e07b39',   # orange
    male    = '#457b9d',   # steel blue
    edu_hi  = '#41b3a3',   # teal  (PhD)
    edu_mid = '#6baed6',   # light blue (Master)
    edu_lo  = '#bdbdbd',   # grey  (Bachelor / other)
    dark    = '#222222',
    mid     = '#555555',
    light   = '#999999',
    rule    = '#dddddd',
)

# Education bucketing
edu_map = {'phd': 'PhD', 'master': 'Master',
           'bachelor': 'Bachelor', 'graduate_certificate': 'Other',
           'professional_certificate': 'Other', 'ap': 'Other'}
p['edu_cat'] = p['education'].map(edu_map).fillna('Other')

# English bucketing
eng_map = {'Native speaker': 'Native', 'Fluent': 'Fluent', 'Intermediate': 'Intermediate'}
p['eng_cat'] = p['english_proficiency'].map(eng_map).fillna('Other')

# Group-level summary
grp = p.groupby('group_id').agg(
    age_mean = ('age', 'mean'),
    age_min  = ('age', 'min'),
    age_max  = ('age', 'max'),
    n_female = ('sex', lambda x: (x == 'female').sum()),
    n_male   = ('sex', lambda x: (x == 'male').sum()),
).reset_index().sort_values('age_mean').reset_index(drop=True)
grp['grp_short'] = [f'G{i+1}' for i in range(len(grp))]

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.09, 3.10), dpi=300)
fig.patch.set_facecolor('white')

# Left panel: 0.04–0.46 of figure width, split into upper (age) + lower 3 rows
# Right panel: 0.54–0.98

gs_left = GridSpec(3, 2, left=0.055, right=0.455, top=0.92, bottom=0.10,
                   hspace=0.62, wspace=0.55)
gs_right = GridSpec(1, 1, left=0.530, right=0.980, top=0.92, bottom=0.10)

ax_age   = fig.add_subplot(gs_left[0, :])   # age histogram, full width
ax_sex   = fig.add_subplot(gs_left[1, 0])   # sex pie
ax_edu   = fig.add_subplot(gs_left[1, 1])   # education bars
ax_eng   = fig.add_subplot(gs_left[2, :])   # English proficiency (full width)
ax_grp   = fig.add_subplot(gs_right[0, 0])  # group composition

# Thin vertical divider
fig.add_artist(plt.Line2D([0.490, 0.490], [0.04, 0.96],
                           transform=fig.transFigure,
                           color=C['rule'], lw=0.7, zorder=10))

# Panel labels (placed in figure coords to avoid overlap with ax titles)
fig.text(0.04, 0.97, 'A', fontsize=9, fontweight='bold', va='top', color=C['dark'])
fig.text(0.52, 0.97, 'B', fontsize=9, fontweight='bold', va='top', color=C['dark'])

# ── A1: Age histogram + KDE ───────────────────────────────────────────────────
ages = p['age'].dropna().values
bins = np.arange(22, 62, 4)
ax_age.hist(ages, bins=bins, color='#6baed6', alpha=0.75, edgecolor='white',
            linewidth=0.5, zorder=2)

kde_x = np.linspace(20, 62, 200)
kde = gaussian_kde(ages, bw_method=0.4)
kde_y = kde(kde_x) * len(ages) * 4   # scale to histogram counts
ax_age.plot(kde_x, kde_y, color='#2166ac', lw=1.4, zorder=3)

ax_age.axvline(np.median(ages), color='#c04020', lw=1.0, ls='--', zorder=4)
ax_age.text(np.median(ages) + 0.8, ax_age.get_ylim()[1] * 0.02,
            f'med={np.median(ages):.0f}', fontsize=5.0, color='#c04020', va='bottom')

ax_age.set_xlabel('Age (years)', fontsize=6.5, labelpad=2)
ax_age.set_ylabel('Count', fontsize=6.5, labelpad=2)
ax_age.set_title('Age distribution  (N=40)', fontsize=7.0, pad=3, color=C['dark'])
ax_age.tick_params(labelsize=5.5)
ax_age.spines[['top', 'right']].set_visible(False)
ax_age.set_xlim(20, 62)
ax_age.set_ylim(bottom=0)
# update median line after ylim set
ax_age.lines[1].set_ydata([0, ax_age.get_ylim()[1]])

# ── A2: Sex donut ─────────────────────────────────────────────────────────────
sex_counts = p['sex'].value_counts()
sex_labels = [s.capitalize() for s in sex_counts.index]
sex_colors = [C['female'] if s == 'female' else C['male'] for s in sex_counts.index]
wedges, _ = ax_sex.pie(sex_counts.values, colors=sex_colors,
                        startangle=90, wedgeprops=dict(width=0.55, edgecolor='white', linewidth=0.6))
ax_sex.set_title('Sex', fontsize=6.5, pad=2, color=C['dark'])
short_lbl = ['F', 'M']
for i, (w, slbl, cnt) in enumerate(zip(wedges, short_lbl, sex_counts.values)):
    ang = (w.theta2 + w.theta1) / 2
    x = 0.62 * np.cos(np.deg2rad(ang))
    y = 0.62 * np.sin(np.deg2rad(ang))
    ax_sex.text(x, y, f'{slbl}\n{cnt}', ha='center', va='center',
                fontsize=5.5, color='white', fontweight='bold')

# ── A3: Education bars ────────────────────────────────────────────────────────
edu_order = ['PhD', 'Master', 'Bachelor', 'Other']
edu_colors = [C['edu_hi'], C['edu_mid'], C['edu_lo'], '#dddddd']
edu_counts = p['edu_cat'].value_counts().reindex(edu_order).fillna(0)
bars = ax_edu.barh(edu_order[::-1], edu_counts[edu_order[::-1]].values,
                    color=edu_colors[::-1], edgecolor='white', linewidth=0.4, height=0.6)
for bar, val in zip(bars, edu_counts[edu_order[::-1]].values):
    if val > 0:
        ax_edu.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                    str(int(val)), va='center', fontsize=5.0, color=C['mid'])
ax_edu.set_xlim(0, 30)
ax_edu.set_title('Education', fontsize=6.5, pad=2, color=C['dark'])
ax_edu.tick_params(labelsize=5.0, left=False)
ax_edu.spines[['top', 'right', 'left']].set_visible(False)
ax_edu.set_xlabel('N', fontsize=5.5, labelpad=1)

# ── A4: English proficiency stacked bar ──────────────────────────────────────
eng_order  = ['Native', 'Fluent', 'Intermediate']
eng_colors = ['#2a9d8f', '#74c476', '#fdae6b']
eng_counts = p['eng_cat'].value_counts().reindex(eng_order).fillna(0).astype(int)
total_eng = eng_counts.sum()
left = 0
for lbl, col, cnt in zip(eng_order, eng_colors, eng_counts):
    ax_eng.barh([0], [cnt], left=[left], color=col, edgecolor='white',
                linewidth=0.5, height=0.55)
    short = {'Native': 'Native\n7', 'Fluent': 'Fluent\n30', 'Intermediate': 'Int.\n3'}
    if cnt >= 3:
        ax_eng.text(left + cnt / 2, 0, short.get(lbl, f'{cnt}'), ha='center', va='center',
                    fontsize=5.0, color='white', fontweight='bold')
    left += cnt

ax_eng.set_xlim(0, total_eng)
ax_eng.set_ylim(-0.5, 0.5)
ax_eng.set_title('English proficiency  (N=40)', fontsize=6.5, pad=2, color=C['dark'])
ax_eng.axis('off')

# ── B: Group composition ─────────────────────────────────────────────────────
ax = ax_grp
ax.set_title('Group composition  (10 groups, N=4 each)', fontsize=7.0,
             pad=3, color=C['dark'])

n_grps = len(grp)
y_pos = np.arange(n_grps)

# Age range strip (horizontal line + dot for mean)
for i, row in grp.iterrows():
    ax.plot([row['age_min'], row['age_max']], [i, i],
            color='#9ecae1', lw=2.5, solid_capstyle='round', zorder=2)
    ax.plot(row['age_mean'], i, 'o', ms=4.5, color='#2166ac',
            zorder=3, markeredgecolor='white', markeredgewidth=0.4)

# Sex composition — stacked dots to the right
dot_x_start = 63
dot_spacing = 2.2
for i, row in grp.iterrows():
    x = dot_x_start
    for _ in range(int(row['n_female'])):
        ax.plot(x, i, 'o', ms=4.0, color=C['female'], zorder=3,
                markeredgecolor='white', markeredgewidth=0.3)
        x += dot_spacing
    for _ in range(int(row['n_male'])):
        ax.plot(x, i, 's', ms=3.8, color=C['male'], zorder=3,
                markeredgecolor='white', markeredgewidth=0.3)
        x += dot_spacing

ax.set_yticks(y_pos)
ax.set_yticklabels(grp['grp_short'].values, fontsize=6.0)
ax.set_xlabel('Age (years)', fontsize=6.5, labelpad=2)
ax.tick_params(labelsize=5.5, left=False)
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.set_xlim(18, 75)
ax.set_ylim(-0.7, n_grps - 0.3)
ax.grid(axis='x', lw=0.3, color=C['rule'], zorder=1)

# Secondary x-axis label for sex dots
ax.axvline(61, color=C['rule'], lw=0.5, ls='--')
ax.text(66.5, -0.62, 'Sex mix', fontsize=5.0, ha='center',
        va='top', color=C['mid'])

# Legend
leg_elements = [
    mpatches.Patch(color='#9ecae1', label='Age range'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2166ac',
               markersize=5, label='Age mean'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=C['female'],
               markersize=5, label='Female'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=C['male'],
               markersize=5, label='Male'),
]
ax.legend(handles=leg_elements, fontsize=4.8, loc='lower right',
          framealpha=0.8, edgecolor=C['rule'], borderpad=0.5,
          handlelength=1.2, handletextpad=0.4, labelspacing=0.3)

# Panel titles
fig.text(0.255, 0.985, 'Individual-level demographics', ha='center', va='top',
         fontsize=7.0, fontweight='bold', color=C['dark'])
fig.text(0.755, 0.985, 'Group-level composition', ha='center', va='top',
         fontsize=7.0, fontweight='bold', color=C['dark'])

# ── Save ─────────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), '..', 'figure', 'dataset_stats')
os.makedirs(out_dir, exist_ok=True)
pdf_path = os.path.join(out_dir, 'demographics_overview.pdf')
png_path = os.path.join(out_dir, 'demographics_overview.png')

fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved {pdf_path}")
print(f"Saved {png_path}")
