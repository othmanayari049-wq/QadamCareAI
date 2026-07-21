"""QadamCare professional v6 entry point.

Adds safer STANDUP reporting:
- explicit same-participant/session/orientation confirmation,
- optional left/right plantar-foot ROI masks,
- dataset-similarity wording instead of disease probability,
- descriptive ROI-aware thermal intensity analysis,
- no automatic HIGH/MEDIUM/LOW clinical monitoring label.

Run with:
    python -m streamlit run app_qadamcare_pro_v6.py
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
V3_APP = ROOT / "app_qadamcare_pro_v3.py"
source = V3_APP.read_text(encoding="utf-8")

# Preserve the base application's Markdown report function (v5 correction).
old_pattern = r'''r'def build_clinical_inputs\(\):.*?def show_validation\(validation, title\):' '''.strip()
new_pattern = r'''r'def build_clinical_inputs\(\):.*?def build_markdown_report\(data\):' '''.strip()
old_marker = '''def show_validation(validation, title):\'\'\''''
new_marker = '''def build_markdown_report(data):\'\'\''''
if old_pattern not in source or old_marker not in source:
    raise RuntimeError("Expected v3 helper markers were not found. Pull the latest branch and retry.")
source = source.replace(old_pattern, new_pattern, 1)
source = source.replace(old_marker, new_marker, 1)

# Inject additional transformations into the v3 wrapper before it executes app_unified.py.
injection_point = 'namespace = {"__file__": str(BASE_APP), "__name__": "qadamcare_professional_v3_runtime"}'
if injection_point not in source:
    raise RuntimeError("Expected v3 execution marker was not found.")

patch_code = r'''
# Install ROI-aware STANDUP thermal analysis.
source = source.replace(
    "from advanced_wound_measurements import calculate_advanced_wound_measurements",
    "from advanced_wound_measurements import calculate_advanced_wound_measurements\n"
    "from standup_roi_thermal_analysis import analyze_relative_thermal_with_rois, save_relative_thermal_overlay",
    1,
)

standup_upload_pattern = re.compile(
    r'elif workflow == WORKFLOW_STANDUP:\n.*?\nelse:\n    st\.markdown\("### Required input"\)',
    re.DOTALL,
)
standup_upload_replacement = '''elif workflow == WORKFLOW_STANDUP:
    st.markdown("### Required paired inputs")
    st.info(
        "Upload a matched plantar RGB image and grayscale thermal image. "
        "The fusion model will run only after explicit pair confirmation."
    )
    c1, c2 = st.columns(2)
    with c1:
        uploaded["standup_rgb"] = st.file_uploader(
            "STANDUP plantar RGB image", type=["png", "jpg", "jpeg"], key="standup_rgb"
        )
    with c2:
        uploaded["standup_thermal"] = st.file_uploader(
            "STANDUP grayscale thermal image", type=["png", "jpg", "jpeg"], key="standup_thermal"
        )

    st.markdown("#### Required pair confirmation")
    same_participant = st.selectbox(
        "Both images are from the same participant", ["Not confirmed", "Yes", "No"], index=0
    )
    same_session = st.selectbox(
        "Both images are from the same capture session", ["Not confirmed", "Yes", "No"], index=0
    )
    orientation_verified = st.selectbox(
        "Left/right orientation has been verified", ["Not confirmed", "Yes", "No"], index=0
    )

    st.markdown("#### Optional foot-region masks")
    st.caption(
        "Upload binary masks only when available: white = the selected plantar foot, black = background. "
        "Without both masks, the app reports whole-frame intensity statistics and does not assign anatomical thermal zones."
    )
    m1, m2 = st.columns(2)
    with m1:
        uploaded["left_foot_roi"] = st.file_uploader(
            "Left plantar-foot ROI mask", type=["png", "jpg", "jpeg"], key="left_foot_roi"
        )
    with m2:
        uploaded["right_foot_roi"] = st.file_uploader(
            "Right plantar-foot ROI mask", type=["png", "jpg", "jpeg"], key="right_foot_roi"
        )
else:
    st.markdown("### Required input")'''
source, count = standup_upload_pattern.subn(standup_upload_replacement, source, count=1)
if count != 1:
    raise RuntimeError("Could not install the safer STANDUP upload interface.")

standup_execution_pattern = re.compile(
    r'        elif workflow == WORKFLOW_STANDUP:\n.*?\n        else:\n            file = uploaded\.get\("pseudo_thermal"\)',
    re.DOTALL,
)
standup_execution_replacement = '''        elif workflow == WORKFLOW_STANDUP:
            rgb_file = uploaded.get("standup_rgb")
            thermal_file = uploaded.get("standup_thermal")
            if rgb_file is None or thermal_file is None:
                raise ValueError("Upload both matching STANDUP RGB and grayscale thermal images.")

            rgb_path = save_uploaded_file(rgb_file, "standup_rgb")
            thermal_path = save_uploaded_file(thermal_file, "standup_thermal")
            primary_image_path = rgb_path
            validation = validate_standup_pair(rgb_path, thermal_path)

            confirmations_ok = (
                same_participant == "Yes"
                and same_session == "Yes"
                and orientation_verified == "Yes"
            )
            confirmation_lines = [
                f"Same participant confirmed: {same_participant}",
                f"Same capture session confirmed: {same_session}",
                f"Left/right orientation verified: {orientation_verified}",
            ]
            sections.append(
                {
                    "heading": "Paired input validation and confirmation",
                    "lines": [
                        f"Automated validation status: {validation['status']}",
                        validation["message"],
                    ]
                    + confirmation_lines
                    + validation.get("warnings", [])
                    + [validation["safety_note"]],
                }
            )
            image_items.extend(
                [
                    {"path": str(rgb_path), "caption": "Submitted STANDUP plantar RGB image"},
                    {"path": str(thermal_path), "caption": "Submitted STANDUP grayscale thermal image"},
                ]
            )

            if not validation["is_valid"] or not confirmations_ok:
                execution_status = "BLOCKED — VERIFIED MATCHED PAIR REQUIRED"
                sections.append(
                    {
                        "heading": "Model execution",
                        "lines": [
                            "The RGB–thermal fusion model was not run.",
                            "Automated image checks and all three user confirmations must pass.",
                            "No dataset-pattern similarity score or thermal anatomical comparison was generated.",
                        ],
                    }
                )
            else:
                dm = analyze_dm_control_pattern(rgb_path, thermal_path)
                dm_score = dm.get("diabetic_foot_pattern_probability")
                control_score = dm.get("healthy_control_pattern_probability")
                dm_score_text = f"{float(dm_score):.1%}" if dm_score is not None else "Not available"
                control_score_text = f"{float(control_score):.1%}" if control_score is not None else "Not available"
                sections.append(
                    {
                        "heading": "STANDUP RGB–thermal dataset-pattern similarity",
                        "lines": [
                            f"Model available: {dm.get('model_available')}",
                            f"Closest dataset-defined pattern: {dm.get('predicted_pattern')}",
                            f"Model similarity score for the dataset-defined diabetic-foot group: {dm_score_text}",
                            f"Model similarity score for the dataset-defined healthy/control group: {control_score_text}",
                            "These complementary scores come from a binary classifier and sum to approximately 100%.",
                            "They are not disease probabilities, diagnostic confidence, severity scores, blood-glucose estimates, or treatment recommendations.",
                            dm.get("safety_note"),
                        ],
                    }
                )

                left_roi_file = uploaded.get("left_foot_roi")
                right_roi_file = uploaded.get("right_foot_roi")
                if (left_roi_file is None) != (right_roi_file is None):
                    raise ValueError("Upload both left and right foot ROI masks, or leave both empty.")

                left_roi_path = None
                right_roi_path = None
                if left_roi_file is not None and right_roi_file is not None:
                    left_roi_path = save_uploaded_file(left_roi_file, "left_foot_roi")
                    right_roi_path = save_uploaded_file(right_roi_file, "right_foot_roi")
                    image_items.extend(
                        [
                            {"path": str(left_roi_path), "caption": "User-supplied left plantar-foot ROI mask"},
                            {"path": str(right_roi_path), "caption": "User-supplied right plantar-foot ROI mask"},
                        ]
                    )

                thermal_result = analyze_relative_thermal_with_rois(
                    thermal_path,
                    left_mask_path=left_roi_path,
                    right_mask_path=right_roi_path,
                )
                thermal_overlay_path = save_relative_thermal_overlay(thermal_result, APP_OUTPUTS)
                if thermal_overlay_path:
                    overlay_path = thermal_overlay_path
                    image_items.append(
                        {"path": thermal_overlay_path, "caption": "ROI-aware relative thermal intensity overlay"}
                    )

                thermal_lines = [
                    f"Foot ROI status: {thermal_result['roi_status']}",
                    f"Analysis scope: {thermal_result['analysis_scope']}",
                    f"Relative upper-decile intensity coverage: {thermal_result['upper_decile_coverage_percent']}%",
                    f"Relative middle-intensity coverage: {thermal_result['middle_intensity_coverage_percent']}%",
                    f"Clinical monitoring level: {thermal_result['clinical_monitoring_level']}",
                    f"Temperature measurement: {thermal_result['temperature_measurement']}",
                ]
                if thermal_result.get("whole_frame_left_right_difference") is not None:
                    thermal_lines.append(
                        f"Whole-frame left–right relative intensity difference: {thermal_result['whole_frame_left_right_difference']}"
                    )
                if thermal_result.get("anatomical_asymmetry_available"):
                    zone_values = thermal_result.get("zone_asymmetry") or {}
                    thermal_lines.extend(
                        [
                            f"Approximate toes/forefoot relative asymmetry: {zone_values.get('toes/forefoot')}",
                            f"Approximate midfoot relative asymmetry: {zone_values.get('midfoot')}",
                            f"Approximate heel relative asymmetry: {zone_values.get('heel')}",
                        ]
                    )
                else:
                    thermal_lines.append(
                        "Anatomical thermal asymmetry: Not available because validated left/right foot ROI masks were not supplied."
                    )
                thermal_lines.extend(thermal_result.get("notes", []))
                thermal_lines.append(thermal_result.get("safety_note"))
                sections.append(
                    {
                        "heading": "Relative grayscale thermal-intensity description",
                        "lines": thermal_lines,
                    }
                )

        else:
            file = uploaded.get("pseudo_thermal")'''
source, count = standup_execution_pattern.subn(standup_execution_replacement, source, count=1)
if count != 1:
    raise RuntimeError("Could not install the safer STANDUP execution workflow.")
'''

source = source.replace(injection_point, patch_code + "\n" + injection_point, 1)

compiled = compile(source, str(V3_APP), "exec")
namespace = {"__file__": str(V3_APP), "__name__": "__main__"}
exec(compiled, namespace)
