import pandas as pd
import networkx as nx
import numpy as np
from scipy.spatial import Voronoi
from collections.abc import Callable



class MakeGraphFromSingleTime:
    """
    Makes Graph of Voronoi Diagram neigbours with following data on:
    - Nodes:
        - cell_size_col_name (values)
        - tracked_ratios_cols_names (values)
        - xy_dimensions_cols_names (values)
    - Edges:
        - distance (based on xy_dimensions_cols_names)

    [TODO]
    """
    def __init__(
            self,
            df: pd.DataFrame,
            cell_size_col_name: str = 'Nuclear_size',
            tracked_ratios_cols_names: list[str] = ["ERKKTR_ratio", "FoxO3A_ratio"],
            xy_dimensions_cols_names: tuple[str, str] = ("objNuclei_Location_Center_X", "objNuclei_Location_Center_Y"),
            cell_size_based_on_area_func: None|Callable[[float], tuple[float, float]] = None,
            verbose: bool = False
        ) -> None:

        self._calculate_max_cell_xy: Callable[[float], tuple[float, float]]

        if cell_size_based_on_area_func is None:
            self._calculate_max_cell_xy = lambda max_area: (np.sqrt(max_area / np.pi) * 2, np.sqrt(max_area / np.pi) * 2)
        else:
            self._calculate_max_cell_xy = cell_size_based_on_area_func


        vor_diag: Voronoi = Voronoi(self._extract_points_from_df(df, dims_cols=xy_dimensions_cols_names))
        correct_bbox: tuple[float, float, float, float] = self._estimate_correct_bbox(
            self._extract_max_area_from_df(df, cell_size_col=cell_size_col_name),
            self._extract_bbox_from_df(df, dims_cols=xy_dimensions_cols_names)
        )

        self._vor_graph: nx.Graph = nx.Graph(
            self._calculate_correct_edges_for_vor_graph(
                vor_diag,
                correct_bbox,
            )
        )

        self._add_attributes_to_graph(
            self._extract_track_ids_from_df(df),
            self._extract_cell_sizes_from_df(df, cell_size_col_name),
            self._extract_cell_ratios_from_df(df, tracked_ratios_cols_names),
            self._extract_points_from_df(df, xy_dimensions_cols_names)
        )


    @staticmethod
    def _extract_cell_ratios_from_df(df: pd.DataFrame, ratios_cols: list[str]) -> dict[str, np.ndarray]:
        return {ratio_name: df[ratio_name].to_numpy() for ratio_name in ratios_cols}

    @staticmethod
    def _extract_points_from_df(df: pd.DataFrame, dims_cols: tuple[str, str]) -> np.ndarray:
        return df[dims_cols].to_numpy()
    
    @staticmethod
    def _extract_track_ids_from_df(df: pd.DataFrame) -> np.ndarray:
        return df.index.to_numpy()
    
    @staticmethod
    def _extract_cell_sizes_from_df(df: pd.DataFrame, cell_size_col: str) -> np.ndarray:
        return df[cell_size_col].to_numpy()
    
    @staticmethod
    def _extract_bbox_from_df(df: pd.DataFrame, dims_cols: tuple[str, str]) -> tuple[float, float, float, float]:
        return (
            df[dims_cols[0]].min(), 
            df[dims_cols[0]].max(),
            df[dims_cols[1]].min(), 
            df[dims_cols[1]].max()
        )
    
    @staticmethod
    def _extract_max_area_from_df(df: pd.DataFrame, cell_size_col: str) -> float:
        return df[cell_size_col].max()
        
    
    def _estimate_correct_bbox(self, max_cell_area: float, raw_bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        max_cell_xy = self._calculate_max_cell_xy(max_cell_area)

        bbox_x_ext, bbox_y_ext = max_cell_xy[0]/2, max_cell_xy[1]/2

        return (
            raw_bbox[0] - bbox_x_ext, 
            raw_bbox[1] + bbox_x_ext, 
            raw_bbox[2] - bbox_y_ext, 
            raw_bbox[3] + bbox_y_ext
        )
    
    def _calculate_correct_edges_for_vor_graph(
        self,
        voronoi_diag: Voronoi,
        bbox: tuple[float, float, float, float],
    ) -> list[tuple[int, int,]]:
        xmin, xmax, ymin, ymax = bbox

        def check_if_vert_in_bbox(vert_id: int) -> bool:
            vert = voronoi_diag.vertices[vert_id]

            return xmin < vert[0] < xmax and ymin < vert[1] < ymax
            

        ridge_points_output: list[tuple[int, int]] = []

        for ridge_idx, ridge_verts_ids in enumerate(voronoi_diag.ridge_vertices):

            if ridge_verts_ids[0] == -1:
                if not check_if_vert_in_bbox(ridge_verts_ids[1]):
                    continue

            if ridge_verts_ids[1] == -1:
                if not check_if_vert_in_bbox(ridge_verts_ids[0]):
                    continue

            if check_if_vert_in_bbox(ridge_verts_ids[0]) or check_if_vert_in_bbox(ridge_verts_ids[1]):
                ridge_points_output.append((
                    voronoi_diag.ridge_points[ridge_idx][0], voronoi_diag.ridge_points[ridge_idx][1]
                ))
            

        return ridge_points_output
    
    def _add_attributes_to_graph(
            self,
            cells_ids: np.ndarray,
            cells_sizes: np.ndarray,
            cells_ratios: dict[str, np.ndarray],
            cells_points: np.ndarray,
        ) -> None:

        nx.set_node_attributes(
            self._vor_graph,
            {cell_idx: cell_id for cell_idx, cell_id in enumerate(cells_ids)},
            "cell_id"
        )

        nx.set_node_attributes(
            self._vor_graph,
            {cell_idx: cell_size for cell_idx, cell_size in enumerate(cells_sizes)},
            "cell_size"
        )

        for ratio_key in cells_ratios:
            nx.set_node_attributes(
                self._vor_graph,
                {cell_idx: ratio for cell_idx, ratio in enumerate(cells_ratios[ratio_key])},
                "ratio_key"
            )

        nodes_pos: dict[int, np.ndarray] = {cell_idx: cell_pos for cell_idx, cell_pos in enumerate(cells_points)}

        nx.set_node_attributes(
            self._vor_graph,
            nodes_pos,
            "cell_pos"
        )

        edges_attrs_dict: dict[tuple[int, int], np.float32] = {}
        for node in self._vor_graph.nodes:
            for node_neigh in self._vor_graph.neighbors(node):
                if (node, node_neigh) not in edges_attrs_dict.keys():
                    edges_attrs_dict[(node, node_neigh)] = np.linalg.norm(nodes_pos[node] - nodes_pos[node_neigh])

        nx.set_edge_attributes(self._vor_graph, edges_attrs_dict, name = 'dist')


        pass