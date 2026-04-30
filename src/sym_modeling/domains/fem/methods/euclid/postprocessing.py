from sym_modeling.domains.fem.imports import *
from sym_modeling.domains.fem.methods.euclid.weak_form import *

import matplotlib.tri as mtri
import re


PIOLA_COMPONENT_MAP = {
    "Pxx": 0,
    "Pxy": 1,
    "Pyx": 2,
    "Pyy": 3,
}
DEFAULT_PLOT_QUANTITIES = ("Pxx", "Pxy", "Pyx", "Pyy", "Pnorm")


def loadThetaFromResults(resultfile):
    """
    Load the last saved theta vector from a EUCLID result file.
    
    _Input Arguments_
    
    - `resultfile` - path to results file
    
    _Output Arguments_
    
    - `theta` - material parameters
    
    ---
    
    """
    with open(resultfile, "r", encoding="utf-8") as handle:
        contents = handle.read()
    matches = re.findall(
        r"Theta \(Lp-Norm Penalty \+ Threshold\):\s*\n(\[[^\]]*\])",
        contents,
        flags=re.DOTALL,
    )
    if not matches:
        raise ValueError("Could not find theta in results file: %s" % resultfile)
    theta_text = matches[-1].replace("\n", " ").strip()[1:-1]
    theta = np.fromstring(theta_text, sep=" ")
    if theta.size == 0:
        raise ValueError("Failed to parse theta from results file: %s" % resultfile)
    return theta


def getTriangulation(data):
    """
    Build a matplotlib triangulation for the element mesh.
    
    _Input Arguments_
    
    - `data`
    
    _Output Arguments_
    
    - `triangulation`
    
    ---
    
    """
    triangles = np.column_stack(data.connectivity).astype(int)
    return mtri.Triangulation(data.x_nodes[:,0], data.x_nodes[:,1], triangles=triangles)


def getFieldValues(P, quantity):
    """
    Extract a scalar field from the first Piola-Kirchhoff stress tensor.
    
    _Input Arguments_
    
    - `P` - first Piola-Kirchhoff stress tensor at quadrature points

    - `quantity` - component or derived field
    
    _Output Arguments_
    
    - `values` - scalar field
    
    ---
    
    """
    if quantity in PIOLA_COMPONENT_MAP:
        return P[:, PIOLA_COMPONENT_MAP[quantity]]
    if quantity == "Pnorm":
        return np.linalg.norm(P, axis=1)
    raise ValueError("Unsupported plot quantity: %s" % quantity)


def getFieldLabel(quantity):
    """
    Create a human-readable label for a plot quantity.
    
    _Input Arguments_
    
    - `quantity`
    
    _Output Arguments_
    
    - `label`
    
    ---
    
    """
    if quantity == "Pnorm":
        return "||P||_F"
    return quantity


def plotFieldComparison(data, baseline_values, modeled_values, quantity, output_path, title=None):
    """
    Plot baseline, modeled, and error fields for a single scalar quantity.
    
    _Input Arguments_
    
    - `data`

    - `baseline_values`

    - `modeled_values`

    - `quantity`

    - `output_path`

    - `title`
    
    _Output Arguments_
    
    - _none_
    
    ---
    
    """
    triangulation = getTriangulation(data)
    error_values = modeled_values - baseline_values

    field_min = np.min(np.concatenate((baseline_values, modeled_values)))
    field_max = np.max(np.concatenate((baseline_values, modeled_values)))
    if np.isclose(field_min, field_max):
        margin = max(1.0, abs(field_min)) * 1e-12
        field_min -= margin
        field_max += margin

    error_abs_max = np.max(np.abs(error_values))
    if np.isclose(error_abs_max, 0.0):
        error_abs_max = 1e-12

    rmse = np.sqrt(np.mean(np.square(error_values)))
    quantity_label = getFieldLabel(quantity)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    baseline_plot = axes[0].tripcolor(
        triangulation,
        facecolors=baseline_values,
        shading="flat",
        cmap="viridis",
        vmin=field_min,
        vmax=field_max,
    )
    modeled_plot = axes[1].tripcolor(
        triangulation,
        facecolors=modeled_values,
        shading="flat",
        cmap="viridis",
        vmin=field_min,
        vmax=field_max,
    )
    error_plot = axes[2].tripcolor(
        triangulation,
        facecolors=error_values,
        shading="flat",
        cmap="coolwarm",
        vmin=-error_abs_max,
        vmax=error_abs_max,
    )

    axes[0].set_title("Baseline")
    axes[1].set_title("Modeled")
    axes[2].set_title("Error (RMSE = %.3e)" % rmse)

    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])

    axes[0].set_ylabel(quantity_label)
    if title is not None:
        fig.suptitle(title)

    fig.colorbar(baseline_plot, ax=axes[:2], shrink=0.8)
    fig.colorbar(error_plot, ax=axes[2], shrink=0.8)

    output_path = str(output_path)
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plotPiolaFieldComparison(data, theta, output_dir, prefix, quantities=DEFAULT_PLOT_QUANTITIES, title_prefix=None):
    """
    Generate baseline-versus-modeled Piola field plots for a dataset.
    
    _Input Arguments_
    
    - `data`

    - `theta` - material parameters

    - `output_dir`

    - `prefix`

    - `quantities`

    - `title_prefix`
    
    _Output Arguments_
    
    - `plot_paths` - paths to generated plots
    
    ---
    
    """
    if data.P is None:
        raise ValueError("Reference Piola field is not available in this dataset.")

    baseline_P = data.P
    modeled_P = computeFirstPiolaTheta(data, theta)
    plot_paths = []
    for quantity in quantities:
        baseline_values = getFieldValues(baseline_P, quantity)
        modeled_values = getFieldValues(modeled_P, quantity)
        title = None
        if title_prefix is not None:
            title = "%s | %s" % (title_prefix, getFieldLabel(quantity))
        filename = "%s_%s.png" % (prefix, quantity)
        output_path = os.path.join(output_dir, filename)
        plotFieldComparison(
            data,
            baseline_values,
            modeled_values,
            quantity,
            output_path,
            title=title,
        )
        plot_paths.append(output_path)
    return plot_paths
