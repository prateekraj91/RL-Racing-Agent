"""Unit tests for Buddh Track Importer and Coordinate Projection."""

import numpy as np
import pytest

from buddh.importer import BuddhTrackImporter, centripetal_catmull_rom_closed, latlon_to_utm43n


def test_utm43n_projection_accuracy():
    """Verify UTM Zone 43N projection produces known reference values."""
    # Reference coordinates near Buddh International Circuit
    lat = np.array([28.349])
    lon = np.array([77.537])
    easting, northing = latlon_to_utm43n(lat, lon)

    # In UTM 43N, Buddh easting is ~748.68 km, northing is ~3138.48 km
    assert abs(easting[0] - 748680.89) < 1.0, f"Unexpected Easting: {easting[0]}"
    assert abs(northing[0] - 3138479.29) < 1.0, f"Unexpected Northing: {northing[0]}"


def test_centripetal_catmull_rom_properties():
    """Verify closed Centripetal Catmull-Rom spline closes and passes through control points."""
    P = np.array([
        [0.0, 0.0],
        [100.0, 0.0],
        [100.0, 100.0],
        [0.0, 100.0],
    ])
    dense = centripetal_catmull_rom_closed(P, samples_per_seg=20, alpha=0.5)
    assert len(dense) == 80
    # First point of first segment should equal P[0]
    assert np.allclose(dense[0], P[0], atol=1e-5)
    # First point of second segment should equal P[1]
    assert np.allclose(dense[20], P[1], atol=1e-5)


def test_importer_execution_and_geometry():
    """Verify BuddhTrackImporter produces valid manifest, closed loop, and boundaries."""
    importer = BuddhTrackImporter(resample_interval_m=2.0)
    res = importer.process_geometry()

    manifest = res["manifest"]
    assert manifest["direction_of_travel"] == "clockwise"
    assert manifest["num_turns_detected"] == 16
    assert abs(manifest["resampled_length_m"] - 5138.0) < 15.0

    centerline = res["centerline_sim"]
    left_b = res["left_boundary_sim"]
    right_b = res["right_boundary_sim"]

    # Boundary widths must equal 70 px everywhere
    widths = np.linalg.norm(left_b - right_b, axis=1)
    assert np.allclose(widths, 70.0, atol=1e-3)
