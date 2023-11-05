import numpy
import torch
from matplotlib import pyplot


def plot_stroke(stroke, save_name=None):
    # Plot a single example.
    f, ax = pyplot.subplots()

    x = numpy.cumsum(stroke[:, 1])
    y = numpy.cumsum(stroke[:, 2])

    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    f.set_size_inches(5. * size_x / size_y, 5.)

    cuts = numpy.where(stroke[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value],
                'k-', linewidth=3)
        start = cut_value + 1
    ax.axis('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if save_name is None:
        pyplot.show()
    else:
        try:
            pyplot.savefig(
                save_name,
                bbox_inches='tight',
                pad_inches=0.5)
        except Exception:
            print("Error building image!: " + save_name)

    pyplot.close()


def plot_stroke_with_ending(stroke, save_name=None):
    # Plot a single example.
    f, ax = pyplot.subplots()

    x = numpy.cumsum(stroke[:, 1])
    y = numpy.cumsum(stroke[:, 2])

    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    f.set_size_inches(5. * size_x / size_y, 5.)

    cuts = numpy.where(stroke[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value],
                'k-', linewidth=3)

        # Adding a scatter plot for the end of the stroke
        ax.scatter(x[cut_value-1], y[cut_value-1], color='red',
                   s=50)  # s is the size of the marker

        start = cut_value + 1
    ax.axis('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if save_name is None:
        pyplot.show()
    else:
        try:
            pyplot.savefig(
                save_name,
                bbox_inches='tight',
                pad_inches=0.5)
        except Exception:
            print("Error building image!: " + save_name)

    pyplot.close()


def check_nan_inf(tensor, layer_name="", check_neg=True, upper_bound=1e10, lower_bound=-1e10):
    num_nan = torch.isnan(tensor).sum().item()
    num_inf = torch.isinf(tensor).sum().item()
    out_of_upper_bound = tensor > upper_bound
    out_of_lower_bound = tensor < lower_bound if not check_neg else tensor < 0

    if num_nan > 0:
        total_elements = tensor.numel()
        print(f"NaN detected after {layer_name}: {num_nan}/{total_elements}")
        print("NaN values:", tensor[torch.isnan(tensor)])
        return False

    if num_inf > 0:
        total_elements = tensor.numel()
        print(f"Inf detected after {layer_name}: {num_inf}/{total_elements}")
        print("Inf values:", tensor[torch.isinf(tensor)])
        return False

    if out_of_upper_bound.sum().item() > 0:
        total_elements = tensor.numel()
        print(
            f"Out of upper bound detected after {layer_name}: {out_of_upper_bound.sum().item()}/{total_elements}")
        print("Out of upper bound values:",
              tensor[out_of_upper_bound])
        return False

    if out_of_lower_bound.sum().item() > 0:
        total_elements = tensor.numel()
        print(
            f"Out of lower bound detected after {layer_name}: {out_of_lower_bound.sum().item()}/{total_elements}")
        print("Out of lower bound values:",
              tensor[out_of_lower_bound])
        return False
    return True
