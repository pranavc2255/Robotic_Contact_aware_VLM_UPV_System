from upv_vlm_orient.anchor_selection.axis_parameterization import (
    build_axis_search_domain,
    map_axis_scalar_to_point,
)
from upv_vlm_orient.anchor_selection.clean_edge_pair_patch_presenter import (
    build_boxed_candidate_tile,
    build_boxed_candidate_tile_grid,
    build_clean_edge_pair_grid,
    build_clean_edge_pair_presentation,
)
from upv_vlm_orient.anchor_selection.contact_hybrid_selector import (
    build_hybrid_decision_trace,
    compute_anchor_contact_features,
    hard_veto,
)
from upv_vlm_orient.anchor_selection.cross_section_sampler import (
    compute_anchor_cross_section_hits,
    sample_cross_section_boundary_hits,
)
from upv_vlm_orient.anchor_selection.edge_neighborhood import (
    compute_candidate_local_edge_supports,
    extract_local_edge_support,
    extract_ordered_mask_contour,
)
from upv_vlm_orient.anchor_selection.edge_pair_crop_builder import (
    build_edge_pair_crop,
    build_edge_pair_crop_grid,
    map_candidate_points_to_rotated_crop_frame,
)
from upv_vlm_orient.anchor_selection.true_edge_pair_patch_builder import (
    build_true_edge_pair_grid,
    build_true_edge_pair_patch,
)
from upv_vlm_orient.anchor_selection.pair_observation_builder import (
    build_pair_observation,
    build_pair_observation_crop,
)
from upv_vlm_orient.anchor_selection.rotated_object_view_builder import (
    build_rotated_object_view,
    crop_rotated_object_view,
    render_rotated_object_overlay,
    rotate_image_and_mask_to_canonical,
)
from upv_vlm_orient.anchor_selection.result_models import (
    AxisSearchDomain,
    BoundaryHit,
    CleanEdgePairPatch,
    EdgePairCrop,
    LocalEdgeSupport,
    PairObservation,
    RotatedObjectView,
    TrueEdgePairPatch,
)
from upv_vlm_orient.anchor_selection.semantic_contact_verifier import (
    build_semantic_advisory,
    build_semantic_summary,
    run_semantic_heuristic_v1,
)

__all__ = [
    "AxisSearchDomain",
    "BoundaryHit",
    "CleanEdgePairPatch",
    "EdgePairCrop",
    "LocalEdgeSupport",
    "PairObservation",
    "RotatedObjectView",
    "TrueEdgePairPatch",
    "build_axis_search_domain",
    "build_boxed_candidate_tile",
    "build_boxed_candidate_tile_grid",
    "build_clean_edge_pair_grid",
    "build_clean_edge_pair_presentation",
    "build_hybrid_decision_trace",
    "build_semantic_advisory",
    "build_semantic_summary",
    "build_edge_pair_crop",
    "build_edge_pair_crop_grid",
    "build_true_edge_pair_grid",
    "build_true_edge_pair_patch",
    "build_pair_observation",
    "build_pair_observation_crop",
    "build_rotated_object_view",
    "compute_anchor_cross_section_hits",
    "compute_anchor_contact_features",
    "compute_candidate_local_edge_supports",
    "crop_rotated_object_view",
    "extract_local_edge_support",
    "extract_ordered_mask_contour",
    "hard_veto",
    "map_candidate_points_to_rotated_crop_frame",
    "map_axis_scalar_to_point",
    "render_rotated_object_overlay",
    "rotate_image_and_mask_to_canonical",
    "run_semantic_heuristic_v1",
    "sample_cross_section_boundary_hits",
]
