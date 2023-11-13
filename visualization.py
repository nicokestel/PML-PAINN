import numpy as np
import matplotlib.pyplot as plt

def show(y, x=None, outfile=None, title=None, xlabel=None, ylabel=None, **kwargs):
    """
    Visualizes a list of y-values, optionally with x-values.

    Args:
        y: A list of y-values.
        x: A list of x-values (optional).
        outfile: The filename to save the visualization to, or None to display it inline.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        **kwargs: Additional keyword arguments to pass to the visualization function.
    """

    # If x-values are not provided, generate them automatically.
    if x is None:
        x = np.arange(len(y))

    # Check if the data is 1D or 2D.
    if len(x.shape) == 1 and len(y.shape) == 1:
        # 1D data.
        plt.plot(x, y, **kwargs)
    elif len(x.shape) == 2 and len(y.shape) == 2 and x.shape[1] == y.shape[1]:
        # 2D data.
        plt.contourf(x, y, **kwargs)
    else:
        raise ValueError('Data must be 1D or 2D.')

    # Set the title and labels.
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    plt.grid(True)

    # Save the figure to a file if specified.
    if outfile is not None:
        plt.savefig(outfile)
    else:
        plt.show()


# Example data
x_values = np.linspace(0, 10, 100)
y_values = np.sin(x_values)

# Example usage of the show function
show(y=y_values, x=x_values, outfile="sin_wave_plot.png", color='blue', linestyle='--', linewidth=2)