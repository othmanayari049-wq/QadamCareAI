from src.local_ollama import ask_ollama_text


SYSTEM_PROMPT = """
You are QadamCare AI Local Clinical Documentation Assistant.

Your only job is to REWRITE the supplied report using clearer professional language.

NON-NEGOTIABLE RULES:
- Use ONLY facts explicitly written in the supplied report.
- Never add placeholders such as [Patient Name], [Age], [Location], [Size], or [Depth].
- Never add a new heading for information that was not provided.
- Never infer wound location, wound size, wound depth, infection, diabetes type,
  ischemia, neuropathy, Wagner grade, treatment, imaging tests, medication,
  surgery, debridement, antibiotics, MRI, CT, or any management plan.
- Never recommend tests or treatment.
- Preserve every number exactly as written.
- If the report does not include an item, do not mention it.
- Keep the same meaning, but make wording more concise and clinician-facing.
- Include this exact sentence at the end:
  "This decision-support summary requires clinician review and does not establish a diagnosis."

Return only the rewritten report.
"""


def polish_report_with_llm(structured_report_markdown):
    prompt = f"""
{SYSTEM_PROMPT}

REPORT TO REWRITE:
---BEGIN REPORT---
{structured_report_markdown}
---END REPORT---
"""

    try:
        text = ask_ollama_text(prompt)

        text = text.replace("---BEGIN REPORT---", "")
        text = text.replace("---END REPORT---", "")
        text = text.replace("--- END REPORT ---", "")
        text = text.strip()

        return {
            "available": True,
            "text": text,
            "message": "Local LLM-polished clinician-facing summary generated.",
        }

    except Exception as error:
        return {
            "available": False,
            "text": None,
            "message": f"Local report polishing could not be completed: {error}",
        }
