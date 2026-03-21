import networkx as nx
import numpy as np
import matplotlib.pyplot as plt



def calculate_nodes_projections(g0: nx.Graph, g1: nx.Graph) -> dict[int, list[int]]:
    nodes_idx_to_ids_g0: dict[int, int] = nx.get_node_attributes(g0, "cell_id")
    nodes_idx_to_ids_g1: dict[int, int] = nx.get_node_attributes(g1, "cell_id")

    nodes_to_neigbours_in_future: dict[int, list[int]] = {}

    for node in g0.nodes:
        mapping_list: list[int] = []

        if nodes_idx_to_ids_g0[node] in nodes_idx_to_ids_g1.values():
            mapping_list.append(nodes_idx_to_ids_g0[node])

        for neigh in g0.neighbors(node):
            if nodes_idx_to_ids_g0[neigh] in nodes_idx_to_ids_g1.values():
                mapping_list.append(nodes_idx_to_ids_g0[neigh])

        nodes_to_neigbours_in_future[nodes_idx_to_ids_g0[node]] = mapping_list

    return nodes_to_neigbours_in_future


def calculate_vector_stats(
        g0: nx.Graph, 
        g1: nx.Graph, 
        tracked_stats_names: list[str] = ['cell_size', 'ERKKTR_ratio', 'FoxO3A_ratio']
    ) -> dict[int, np.ndarray]:
    nodes_projections_t0_t1: dict[int, list[int]] = calculate_nodes_projections(g0, g1)

    ids_to_idx_g0: dict[int, int] = {val: key for key, val in nx.get_node_attributes(g0, "cell_id").items()}
    ids_to_idx_g1: dict[int, int] = {val: key for key, val in nx.get_node_attributes(g1, "cell_id").items()}

    
    subgraph_stats: dict[int, np.ndarray] = {}

    for node_id in nodes_projections_t0_t1:
        stats_vector: list[float] = []

        starting_idx: float = 0
        if node_id == nodes_projections_t0_t1[node_id][0]:
            starting_idx = 1

        stats_vector.append(starting_idx)

        for stat_name in tracked_stats_names:
            stats_vector.append(
                g0.nodes[ids_to_idx_g0[node_id]][stat_name]
            )


        stats_dict: dict[str, list[float]] = {stat_name: [] for stat_name in tracked_stats_names}

        for node_neigh_id in nodes_projections_t0_t1[node_id][starting_idx:]:
            for stat_name in tracked_stats_names:
                stats_dict[stat_name].append(
                    g1.nodes[ids_to_idx_g1[node_neigh_id]][stat_name]
                )

        for stat_name in tracked_stats_names:
            stats_vector.append(np.mean(stats_dict[stat_name]).astype(float))
            stats_vector.append(np.std(stats_dict[stat_name], mean=stats_vector[-1]).astype(float))

        stats_vector.append(len(nodes_projections_t0_t1[node_id]) - starting_idx)

        subgraph_stats[node_id] = np.array(stats_vector)
        
    return subgraph_stats

def save_plot_graph_with_labels(g: nx.Graph, labels: np.ndarray, path: str) -> None:
    nx.draw(
        g,
        node_size=[np.pi * (value+0.5) ** 2.5 for value in nx.get_node_attributes(g, "ERKKTR_ratio").values()], 
        width = 0.5, 
        with_labels=False, 
        pos=nx.get_node_attributes(g, "cell_pos"), 
        node_color = [f"C{lab}" if lab != -1 else "black" for lab in labels],
    )
    plt.savefig(path)
    plt.close()