@'

\# QadamCare AI



AI-assisted diabetic-foot screening, monitoring, secondary complication review support, and clinician documentation prototype.



QadamCare AI is an educational engineering prototype that supports visual screening and documentation for diabetic-foot follow-up. It combines RGB wound-region segmentation, image-quality assessment, clinician-entered findings, previous-visit comparison, complication pathway review support, optional thermography research extension, and clinician-facing report export.



> Important: QadamCare AI is not a diagnostic system. It does not confirm infection, ischemia, osteomyelitis, ulcer depth, Wagner grade, amputation risk, or treatment decisions. All outputs require review by a qualified healthcare professional.



\## Main Features



\- RGB diabetic-foot wound/ulcer-like region segmentation

\- AI mask and overlay generation

\- image-quality assessment

\- wound-like area estimation in pixels

\- previous-visit area comparison

\- clinician-entered finding review

\- secondary complication pathway support

\- optional thermography research extension

\- local LLM/VLM documentation copilot

\- professional PDF and Markdown report export



\## System Workflow



```text

Foot image upload

&#x20;       |

&#x20;       v

Image quality assessment

&#x20;       |

&#x20;       v

RGB segmentation model

&#x20;       |

&#x20;       v

Mask, overlay, area, confidence

&#x20;       |

&#x20;       v

Clinical inputs + previous visit data

&#x20;       |

&#x20;       v

Advanced review-support logic

&#x20;       |

&#x20;       v

Complication pathway support

&#x20;       |

&#x20;       v

Clinician-facing report export

