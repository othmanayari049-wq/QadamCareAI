"""Evidence context used by the local documentation model.

This module does not diagnose or prescribe. It supplies a bounded clinical-reasoning
framework so the local LLM can produce a useful clinician review note without
inventing facts or merely repeating the structured summary.
"""

ADA_REFERENCE = (
    "American Diabetes Association Professional Practice Committee for Diabetes. "
    "Retinopathy, Neuropathy, and Foot Care: Standards of Care in Diabetes—2026. "
    "Diabetes Care. 2026;49(Suppl 1):S261-S276. doi:10.2337/dc26-S012."
)

IWGDF_IDSA_REFERENCE = (
    "Senneville E, et al. IWGDF/IDSA Guidelines on the Diagnosis and Treatment of "
    "Diabetes-Related Foot Infections. Clinical Infectious Diseases. 2023;ciad527. "
    "doi:10.1093/cid/ciad527."
)

EVIDENCE_FRAMEWORK = """
Use the following evidence-informed principles only as a reasoning framework. Do not
claim that any condition is present unless it was explicitly entered by the user or
established by a qualified clinician.

1. A diabetes-related foot infection is a clinical diagnosis based on local or
   systemic inflammatory signs. An image, temperature map, segmentation mask, or AI
   score cannot confirm infection by itself.
2. Redness, warmth, swelling, discharge/odour, pain escalation, and fever/systemic
   symptoms are review signals. Combinations increase concern, but remain unconfirmed
   until clinical examination.
3. Fever or systemic illness together with local foot findings can indicate a more
   serious pathway and warrants prompt clinician assessment. Absence of fever does not
   exclude a foot infection.
4. Neuropathy can reduce protective sensation and pain perception. Therefore, a low
   pain score does not necessarily indicate a low-risk foot when neuropathy is reported.
5. Peripheral arterial/vascular disease may impair perfusion and healing and can
   increase the seriousness of an ulcer or infection concern. Image colour or thermal
   appearance cannot establish ischemia.
6. Probe-to-bone is an important clinician-entered warning finding for possible bone
   involvement, but it is not diagnostic on its own.
7. Open ulceration, unexplained swelling, erythema, or increased skin temperature in a
   person with diabetes merits timely foot-specialist or interprofessional review.
8. Increasing wound-like area across comparable visits can support concern for delayed
   healing or progression, but pixel measurements are comparable only when capture
   conditions are similar.
9. Thermal information is supportive only. Relative thermal intensity and asymmetry do
   not diagnose infection, ischemia, or future ulceration, and pseudo-colour values are
   not temperatures unless calibrated raw thermal data are available.
10. The FUSeg output identifies visible ulcer-like pixels only. It cannot determine
    depth, infection, perfusion, exposed structures, osteomyelitis, or tissue viability.
11. Age, sex/gender, and diabetes type provide context but must not be used to invent a
    diagnosis. Older age may increase the importance of comorbidity, mobility, vision,
    self-care, renal, cardiovascular, and medication review. Type 1 and type 2 diabetes
    both can be associated with neuropathy and foot complications; duration and control
    are required for meaningful individualized interpretation.
12. A strong clinical synthesis should distinguish:
    - facts explicitly supplied by the user,
    - valid AI/image-derived outputs,
    - plausible clinical pathways to consider,
    - dangerous combinations/red flags,
    - information still missing,
    - what a clinician would verify next.
13. Never prescribe antibiotics, medication, surgery, dressings, debridement, imaging,
    or laboratory tests. You may state that a clinician would determine whether such
    evaluation is indicated.
14. Never calculate a formal IWGDF infection grade, WIfI stage, Wagner grade, SINBAD
    score, PEDIS grade, or amputation probability unless every required variable is
    explicitly available and the software has a validated implementation. This
    prototype does not currently have such validated implementations.
"""

REFERENCES_TEXT = f"""
Evidence framework references:
- {ADA_REFERENCE}
- {IWGDF_IDSA_REFERENCE}
"""
