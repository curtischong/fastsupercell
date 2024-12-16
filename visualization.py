import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_3d_coords(coords: np.ndarray, title="3D Coordinates Plot"):
    """
    Plots 3D coordinates.

    Parameters:
        coords (list or np.ndarray): A list or array of 3D coordinates. Each coordinate should be a tuple (x, y, z).
        title (str): Title of the plot.
    """
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)

    if coords.shape[1] != 3:
        raise ValueError(
            "Input coordinates must have three columns representing x, y, and z."
        )

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x, y, z, c="blue", marker="o", label="Points")

    ax.set_title(title)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_aspect(
        "equal", adjustable="box"
    )  # by default the scales of the axis are NOT the equal (z-axis is shorter)
    ax.legend()
    plt.show()

def visualize_lattice(lattice: np.ndarray, translation=None):
    # Create a Plotly figure
    fig = go.Figure()
    points = plot_with_parallelepiped(fig, lattice, translation)

    # Set the layout for the 3D plot
    fig.update_layout(
        title="Crystal Structure",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )
    return fig

def plot_edges(fig, edges, color):
    for edge in edges:
        fig.add_trace(
            go.Scatter3d(
                x=[edge[0][0], edge[1][0]],
                y=[edge[0][1], edge[1][1]],
                z=[edge[0][2], edge[1][2]],
                mode="lines",
                line=dict(color=color, width=5),
                showlegend=False,  # Do not add to the legend
            )
        )


def plot_with_parallelepiped(fig, L, translation=None, color="#0d5d85"):
    v1 = L[0]
    v2 = L[1]
    v3 = L[2]
    # Create the parallelepiped by combining the basis vectors
    points = np.array([[0, 0, 0], v1, v1 + v2, v2, v3, v1 + v3, v1 + v2 + v3, v2 + v3])
    if translation is not None:
        points = points + translation

    # Create the edges of the parallelepiped as tuples of Cartesian coordinates
    edges = [
        (tuple(points[0]), tuple(points[1])),
        (tuple(points[1]), tuple(points[2])),
        (tuple(points[2]), tuple(points[3])),
        (tuple(points[3]), tuple(points[0])),
        (tuple(points[4]), tuple(points[5])),
        (tuple(points[5]), tuple(points[6])),
        (tuple(points[6]), tuple(points[7])),
        (tuple(points[7]), tuple(points[4])),
        (tuple(points[0]), tuple(points[4])),
        (tuple(points[1]), tuple(points[5])),
        (tuple(points[2]), tuple(points[6])),
        (tuple(points[3]), tuple(points[7])),
    ]
    # Plot the edges using the helper function
    plot_edges(fig, edges, color)

    return points

def plot_points(fig, coords, color="#0d5d85"):
    for coord in coords:
        fig.add_trace(
            go.Scatter3d(
                x=[coord[0]],
                y=[coord[1]],
                z=[coord[2]],
                mode="markers",
                marker=dict(color=color, size=5),
                showlegend=False,  # Do not add to the legend
            )
        )