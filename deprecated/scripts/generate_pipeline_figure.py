import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

os.makedirs('figures', exist_ok=True)

fig, ax = plt.subplots(figsize=(20, 10), dpi=600, facecolor='white')
ax.set_xlim(0, 24)
ax.set_ylim(0, 12)
ax.axis('off')

# Font styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Helvetica Neue', 'Helvetica', 'Arial', 'sans-serif']

c_text = "#212529"
c_med = "#495057"
c_light = "#adb5bd"

c_gran = "#d35400" # Orange
c_enc  = "#2980b9" # Blue
c_sem  = "#8e44ad" # Purple
c_bg_mut = "#f8f9fa"
c_bord_mut = "#dee2e6"

def draw_round_box(x, y, w, h, text, facecolor='white', edgecolor='black', tc='black', fs=10, fw='normal', ls=1.4, alpha=1.0):
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1", facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5, zorder=3, alpha=alpha
    ))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs, fontweight=fw, color=tc, zorder=5, linespacing=ls)

def draw_arrow(start, end, color="#adb5bd", lw=1.5, head_width=0.2, head_length=0.3):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle=f"-|>,head_width={head_width},head_length={head_length}", 
                                color=color, lw=lw, shrinkA=0, shrinkB=0), zorder=2)

# --- Stage 1 (Muted Left side) ---
ax.text(2.25, 11, "Stage 1\nCohort & Formatting", ha='center', va='top', fontsize=12, fontweight='bold', color=c_med)
draw_round_box(0.5, 8.0, 3.5, 1.5, "MIMIC-IV v3.1\nRaw EHR Extracts", c_bg_mut, c_bord_mut, c_med, 10)
draw_arrow((2.25, 8.0), (2.25, 7.0), c_bord_mut)
draw_round_box(0.5, 5.5, 3.5, 1.5, "MEDS Pipeline\n(Leakage Controls &\nClinical Thresholds)", c_bg_mut, c_bord_mut, c_med, 10)
draw_arrow((2.25, 5.5), (2.25, 4.5), c_bord_mut)
draw_round_box(0.5, 3.0, 3.5, 1.5, "Study Cohorts\nExp 1-2: Hosp. >24h\nExp 3: ICU Only", c_bg_mut, c_bord_mut, c_med, 10)

# Connect Stage 1 to Stage 2
draw_arrow((4.0, 8.8), (4.8, 8.8), c_bord_mut, lw=3, head_width=0.3, head_length=0.4)

# --- Stage 2 (Centerpiece: Tokenization Factory) ---
ax.add_patch(patches.Rectangle((4.8, 0.5), 14.2, 11.2, facecolor='#fdfefe', edgecolor='#d5d8dc', linewidth=2, linestyle='--', zorder=1))
ax.text(11.9, 11.2, "Stage 2: Representation Mechanics & Tokenization Factory", ha='center', va='center', fontsize=14, fontweight='bold', color=c_text, bbox=dict(facecolor='white', edgecolor='none', pad=2))

# Event enters
draw_round_box(5.5, 9.5, 12.8, 1.0, "Raw Clinical Event  $\\rightarrow$  [ Time: 14:30 | Code: LAB//50971 | Value: 4.2 mmol/L ]", "#ffffff", "#bdc3c7", c_text, 12, fw='bold')

# Axis 3: Semantics (Left)
ax.add_patch(patches.FancyBboxPatch((5.5, 3.5), 5.5, 5.2, boxstyle="round,pad=0.1", facecolor=c_sem, alpha=0.04, edgecolor=c_sem, linewidth=1.5, zorder=2))
ax.text(8.25, 8.2, "Axis 3: Vocabulary Semantics", ha='center', va='center', fontsize=11, fontweight='bold', color=c_sem)
draw_round_box(6.0, 6.7, 4.5, 1.0, "Native (Institution Specific)\n$\\rightarrow$ LAB//50971", "#ffffff", c_sem, c_text, 10)
draw_round_box(6.0, 5.2, 4.5, 1.0, "Standardized (CLIF)\n$\\rightarrow$ LAB//potassium", "#ffffff", c_sem, c_text, 10)
draw_round_box(6.0, 3.7, 4.5, 1.0, "Negative Controls\n(Random / Freq. Matched)", "#ffffff", c_sem, c_text, 10)

# Axis 1: Granularity (Right Top)
ax.add_patch(patches.FancyBboxPatch((11.5, 6.2), 6.8, 2.8, boxstyle="round,pad=0.1", facecolor=c_gran, alpha=0.04, edgecolor=c_gran, linewidth=1.5, zorder=2))
ax.text(14.9, 8.5, "Axis 1: Quantization Granularity", ha='center', va='center', fontsize=11, fontweight='bold', color=c_gran)
# Bell curve
x_curve = np.linspace(12.0, 15.5, 100)
y_curve = np.exp(-(x_curve - 13.75)**2 / 1.0) * 1.2 + 6.5
ax.plot(x_curve, y_curve, color=c_gran, lw=1.5, zorder=4)
ax.fill_between(x_curve, 6.5, y_curve, color=c_gran, alpha=0.15, zorder=3)
# Bins
for x_line in [12.6, 13.1, 13.6, 14.1, 14.6, 15.1]:
    ax.plot([x_line, x_line], [6.5, np.exp(-(x_line - 13.75)**2 / 1.0) * 1.2 + 6.5], color=c_gran, lw=1, ls='--', zorder=4)
ax.scatter([13.9], [6.5], color='red', s=60, zorder=6)
ax.text(13.9, 6.2, "4.2", ha='center', va='top', fontsize=9, color='red', fontweight='bold')
ax.plot([12.6, 12.6], [6.5, 7.5], color=c_text, lw=1.5, zorder=5) # Lc
ax.plot([15.1, 15.1], [6.5, 7.5], color=c_text, lw=1.5, zorder=5) # Uc
ax.text(12.6, 7.6, "$L_c$", ha='center', va='bottom', fontsize=9, color=c_text)
ax.text(15.1, 7.6, "$U_c$", ha='center', va='bottom', fontsize=9, color=c_text)
ax.text(16.0, 7.4, "Population Quantiles\n(Deciles $\\rightarrow$ Centiles)\nvs.\nClinical Anchoring", ha='left', va='center', fontsize=9.5, color=c_text)

# Axis 2: Encoding (Right Bottom)
ax.add_patch(patches.FancyBboxPatch((11.5, 3.5), 6.8, 2.5, boxstyle="round,pad=0.1", facecolor=c_enc, alpha=0.04, edgecolor=c_enc, linewidth=1.5, zorder=2))
ax.text(14.9, 5.5, "Axis 2: Value Encoding Mechanics", ha='center', va='center', fontsize=11, fontweight='bold', color=c_enc)
draw_round_box(12.0, 3.8, 1.6, 1.2, "Discrete\n(Hard Bin)\n$\\rightarrow$ [BIN_6]", "#ffffff", c_enc, c_text, 9)
draw_round_box(14.0, 3.8, 2.3, 1.2, "Soft (ConSE)\n(Interpolation)\n$\\rightarrow$ 0.4[B_6] + 0.6[B_7]", "#ffffff", c_enc, c_text, 9)
draw_round_box(16.7, 3.8, 1.3, 1.2, "Continuous\n(xVal)\n$\\rightarrow$ [NUM]*$z$", "#ffffff", c_enc, c_text, 9)

# Plumb it all
draw_arrow((8.25, 9.5), (8.25, 8.7), c_sem, lw=2)
draw_arrow((14.9, 9.5), (14.9, 9.0), c_gran, lw=2)
draw_arrow((14.9, 6.2), (14.9, 6.0), c_enc, lw=2) # into enc

# Final timeline
draw_round_box(5.5, 1.5, 12.8, 1.0, "Fused / Unfused Event Tokenization  $\\rightarrow$  [BOS] ... [TIME]  [Code Token]  [Value Token(s)]  ... [EOS]", "#ffffff", "#bdc3c7", c_text, 12, fw='bold')
draw_arrow((8.25, 3.5), (8.25, 2.5), c_sem, lw=2)
draw_arrow((14.9, 3.5), (14.9, 2.5), c_enc, lw=2)

# --- Stage 3 & 4 (Muted Right side) ---
ax.text(21.75, 11, "Stages 3 & 4\nModel & Evaluation", ha='center', va='top', fontsize=12, fontweight='bold', color=c_med)
draw_round_box(20.0, 6.8, 3.5, 2.0, "Stage 3: Causal LM\nPretraining\n$L=4096$, No Early Stop\n87M Llama 3.2\n(Time-Aware RoPE)", c_bg_mut, c_bord_mut, c_med, 10, ls=1.6)
draw_arrow((21.75, 6.8), (21.75, 6.0), c_bord_mut)
draw_round_box(20.0, 2.5, 3.5, 3.5, "Stage 4: Probe Eval\n$\\ell_2$-regularized Logit/MLP\n\nTargets:\n- Mortality (Same Adm)\n- Long LOS (>7d)\n- ICU Admission/Stay\n- Mech. Ventilation", c_bg_mut, c_bord_mut, c_med, 10, ls=1.6)

# The connection from Stage 2 Tokenization jumps up to Stage 3
ax.plot([18.3, 19.3, 19.3], [2.0, 2.0, 7.7], color=c_bord_mut, lw=3, zorder=1)
draw_arrow((19.3, 7.7), (20.0, 7.7), c_bord_mut, lw=3, head_width=0.3, head_length=0.4)


plt.tight_layout()
plt.savefig('figures/pipeline_overview.png', dpi=600, bbox_inches='tight', transparent=False)
plt.close()
print("Redesigned pipeline_overview.png successfully generated.")
