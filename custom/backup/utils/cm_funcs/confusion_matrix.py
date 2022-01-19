from .confusion_matrix_pretty_print import plot_confusion_matrix_from_data
def plot_cm(gt_labels,pred_labels):

    columns = []
    annot = True;
    cmap = 'Oranges';
    fmt = '.2f'
    lw = 0.5
    cbar = False
    show_null_values = 2
    pred_val_axis = 'y'
    # size::
    fz = 12;
    figsize = [9, 9];
    if (len(gt_labels) > 10):
        fz = 9;
        figsize = [14, 14];
    plot_confusion_matrix_from_data(gt_labels,pred_labels, columns,
                                    annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)
#