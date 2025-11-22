import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.lines import Line2D

from matplotlib.colors import LinearSegmentedColormap
def plotALCM(logs:list, pos:list, cidx:int, plotversion=0, figname='f'):
  if figname is 'f':
    figname += plotversion
  # logs: list of records [original.record[i], hpt.record[i]]
  sideA = logs[0].records[pos[0]]
  sideB = logs[1].records[pos[1]]

  # Color palette
  def lighten_color(color, amount=0.5):
    r, g, b = color
    return (
      r + (1 - r) * amount,
      g + (1 - g) * amount,
      b + (1 - b) * amount
    )

  tab10 = sns.color_palette("tab10")
  tab10_light = [lighten_color(c, amount=0.3) for c in tab10]
  # Create single-color gradient colormaps based on model color
  def create_colormap(color):
    return LinearSegmentedColormap.from_list("custom_cmap", [(1, 1, 1), color])

  # Colormaps for confusion matrices
  cmap_sideA = create_colormap(tab10[cidx])
  cmap_sideB = create_colormap(tab10_light[cidx])
  # for reference
  # feature_order = ['raw', 'l2', 'l2a', 'rmo', 'fft', 'mnf', 'zcr', 'tap']
  # idx = [7, 0, 1, 2, 5, 4, 3, 6]

  # Set up the subplot grid
  fig = plt.figure(figsize=(10, 6))
  gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

  # Legend handles for line style (metrics)
  train_line = Line2D([], [], color='gray', linestyle='-.', label='Training')
  val_line   = Line2D([], [], color='gray', linestyle='-', label='Validation')
  stop_line  = Line2D([], [], color='gray', linestyle='--', label='Early Stop')

  # Legend handles for model type (color + marker)
  og_marker  = Line2D([], [], color=tab10[cidx], linestyle='-', label='Original')
  hpt_marker = Line2D([], [], color=tab10_light[cidx], linestyle='-', label='HPT')

  # Accuracy plot
  ax1 = fig.add_subplot(gs[0, 0])
  ax1.plot(sideA['acc'], color=tab10[cidx], linestyle='-.')
  ax1.plot(sideA['vac'], color=tab10[cidx], linestyle='-')
  ax1.axvline(sideA['epc'], color=tab10[cidx], linestyle='--')
  ax1.plot(sideB['acc'], color=tab10_light[cidx], linestyle='-.')
  ax1.plot(sideB['vac'], color=tab10_light[cidx], linestyle='-')
  ax1.axvline(sideB['epc'], color=tab10_light[cidx], linestyle='--')
  ax1.set_ylim((0,1))
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Accuracy')
  ax1.grid(True)

  # Loss plot
  ax2 = fig.add_subplot(gs[0,1])
  ax2.plot(sideA['los'], color=tab10[cidx], linestyle='-.')
  ax2.plot(sideA['vlo'], color=tab10[cidx], linestyle='-')
  ax2.axvline(sideA['epc'], color=tab10[cidx],linestyle='--')
  ax2.plot(sideB['los'], color=tab10_light[cidx], linestyle='-.')
  ax2.plot(sideB['vlo'], color=tab10_light[cidx], linestyle='-')
  ax2.axvline(sideB['epc'], color=tab10_light[cidx], linestyle='--')
  ax2.legend(
    handles=[train_line, val_line, stop_line, og_marker, hpt_marker],
    loc='upper right',
    bbox_to_anchor=(1.3, 1.02),
    ncol=1, # Stack vertically
    borderaxespad=0,
    frameon=True, # Optional: add/remove box frame
    title='Legend',
    fontsize=9,
    title_fontsize=10
  )
  # ax2.set_title('TAP Loss')
  ax2.set_xlabel('Epoch')
  ax2.set_yscale('log')
  ax2.set_ylabel('Loss')
  ax2.grid(True)

  # Confusion matrix plot (right, spanning both rows)
  cm = confusion_matrix(sideA['obs'], sideA['prd'])
  ax3 = fig.add_subplot(gs[1, 0])
  sns.heatmap(cm, annot=False, fmt='d', cmap=cmap_sideA, cbar=False, ax=ax3, square=False)
  # ax3.set_title('Original')
  ax3.set_xlabel('Original Predicted')
  ax3.set_ylabel('Actual')

  cm = confusion_matrix(sideB['obs'], sideB['prd'])
  ax4 = fig.add_subplot(gs[1, 1])
  sns.heatmap(cm, annot=False, fmt='d', cmap=cmap_sideB, cbar=False, ax=ax4, square=False)
  # ax4.set_title('Tuned Hyperparameters')
  ax4.set_xlabel('HPT Predicted')
  ax4.set_ylabel('Actual')

  fig.subplots_adjust(wspace=0.3, hspace=0.4)
  plt.tight_layout()
  # plt.show()
  # if not figname:
    # figname =f'{plotversion}.{sideA['DID'][3:]}.{sideA['seg']}.{sideA['FEM']}'
  plt.savefig(f'../Figs/PDF/{figname}.pdf', format='pdf', bbox_inches='tight')
  plt.savefig(f'../Figs/EPS/{figname}.eps', format='eps', bbox_inches='tight')
  plt.savefig(f'../Figs/png/{figname}.png', format='png', bbox_inches='tight') 

from matplotlib.ticker import MaxNLocator
def plotALCM_RGB(
  logs:list, pos:list, colors:list, cmaps=["Blues", "Blues"], 
  plotversion=0, multiLegend=False, figname='f'
):
  if figname is 'f':
    figname += plotversion
    
  hplabel = 'HpO'
  # ALCM: Accuracy, Loss, Confusion Matrix
  # logs: list of records [original.record[i], hpt.record[i]]
  sideA = logs[0].records[pos[0]]
  sideB = logs[1].records[pos[1]]

  # Set up the subplot grid
  fig = plt.figure(figsize=(10, 6))
  gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

  # Legend handles for line style (metrics)
  train_line = Line2D([], [], color='gray', linestyle='-.', label='Training')
  val_line   = Line2D([], [], color='gray', linestyle='-', label='Validation')
  estop_line = Line2D([], [], color='gray', linestyle='--', label='Early Stop')
  



  # Accuracy plot
  ax1 = fig.add_subplot(gs[0, 0])
  ax1.plot(sideA['acc'], color=colors[0], linestyle='-.')
  ax1.plot(sideA['vac'], color=colors[0], linestyle='-')
  ax1.axvline(sideA['epc'], color=colors[1], linestyle='--')
  ax1.plot(sideB['acc'], color=colors[2], linestyle='-.')
  ax1.plot(sideB['vac'], color=colors[2], linestyle='-')
  ax1.axvline(sideB['epc'], color=colors[3], linestyle='--')
  ax1.set_ylim((0,1))
  ax1.set_xlabel('Epoch')
  ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax1.set_ylabel('Accuracy')
  ax1.grid(True)

  if multiLegend:
    # Legend handles for model type (color + marker)
    og_marker  = Line2D([], [], color=colors[0], linestyle='-', label='Original')
    # stop_lineA  = Line2D([], [], color=colors[1], linestyle='--', label='Early Stop')
    hpt_marker = Line2D([], [], color=colors[2], linestyle='-', label=hplabel)
    # stop_lineB  = Line2D([], [], color=colors[3], linestyle='--', label='Early Stop')
    ax1.legend(
      handles=[train_line, val_line, estop_line, og_marker, hpt_marker],
      # handles=[train_line, val_line, og_marker, stop_lineA, hpt_marker, stop_lineB],
      loc='upper right',
      bbox_to_anchor=(1.3, 1.02),
      ncol=1, # Stack vertically
      borderaxespad=0,
      frameon=True, # Optional: add/remove box frame
      title='Legend',
      fontsize=9,
      title_fontsize=10
    )

  # Loss plot
  # Legend handles for model type (color + marker)
  og_marker  = Line2D([], [], color=colors[4], linestyle='-', label='Original')
  # stop_lineA  = Line2D([], [], color=colors[5], linestyle='--', label='Early Stop')
  hpt_marker = Line2D([], [], color=colors[6], linestyle='-', label=hplabel)
  # stop_lineB  = Line2D([], [], color=colors[7], linestyle='--', label='Early Stop')

  ax2 = fig.add_subplot(gs[0,1])
  ax2.plot(sideA['los'], color=colors[4], linestyle='-.')
  ax2.plot(sideA['vlo'], color=colors[4], linestyle='-')
  ax2.axvline(sideA['epc'], color=colors[5],linestyle='--')
  ax2.plot(sideB['los'], color=colors[6], linestyle='-.')
  ax2.plot(sideB['vlo'], color=colors[6], linestyle='-')
  ax2.axvline(sideB['epc'], color=colors[7], linestyle='--')
  ax2.legend(
    handles=[train_line, val_line, estop_line, og_marker, hpt_marker],
    # handles=[train_line, val_line, og_marker, stop_lineA, hpt_marker, stop_lineB],
    loc='upper right',
    bbox_to_anchor=(1.3, 1.02),
    ncol=1, # Stack vertically
    borderaxespad=0,
    frameon=True, # Optional: add/remove box frame
    title='Legend',
    fontsize=9,
    title_fontsize=10
  )
  # ax2.set_title('TAP Loss')
  ax2.set_xlabel('Epoch')
  ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax2.set_ylabel('Loss')
  # ax2.set_yscale('log')
  ax2.grid(True)

  # Confusion matrix plot (right, spanning both rows)
  cm = confusion_matrix(sideA['obs'], sideA['prd'])
  ax3 = fig.add_subplot(gs[1, 0])
  sns.heatmap(cm, annot=False, fmt='d', cmap=cmaps[0], cbar=False, ax=ax3, square=False)
  # ax3.set_title('Original')
  ax3.set_xlabel('Original Predicted')
  ax3.set_ylabel('Actual')

  cm = confusion_matrix(sideB['obs'], sideB['prd'])
  ax4 = fig.add_subplot(gs[1, 1])
  sns.heatmap(cm, annot=False, fmt='d', cmap=cmaps[1], cbar=False, ax=ax4, square=False)
  # ax4.set_title('Tuned Hyperparameters')
  ax4.set_xlabel(f'{hplabel} Predicted')
  ax4.set_ylabel('Actual')

  fig.subplots_adjust(wspace=0.3, hspace=0.4)
  plt.tight_layout()
  # plt.show()
  
  # figname =f'{plotversion}.{sideA['DID'][3:]}.{sideA['seg']}.{sideA['FEM']}'
  plt.savefig(f'../Figs/PDF/{figname}.pdf', format='pdf', bbox_inches='tight')
  plt.savefig(f'../Figs/EPS/{figname}.eps', format='eps', bbox_inches='tight')
  plt.savefig(f'../Figs/png/{figname}.png', format='png', bbox_inches='tight') 

