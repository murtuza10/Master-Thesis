def generate_ner_prompts(text):
    system_prompt = f"""
    ### Instruction ###
You are an expert in Named Entity Recognition (NER) specialized in agricultural domain texts.

Your task is to extract all explicitly mentioned entities from the **input text** and return them in structured **JSON format**.

Focus on identifying the following categories and their subtypes:

### Entity Categories ###

1. Crops
   - cropSpecies
   - cropVariety

2. Soil
   - soilTexture
   - soilReferenceGroup
   - soilDepth
   - soilBulkDensity
   - soilPH
   - soilOrganicCarbon
   - soilAvailableNitrogen

3. Location
   - country
   - region
   - city
   - latitude
   - longitude

4. Time Statement
   - startTime
   - endTime
   - duration
### Rules ###
- Return entities **strictly** in the JSON format below — no extra text, no explanations.
- Each entity must include:
   - `"value"` — the exact string from the text.
   - `"span"` — the **start and end character positions** from the **beginning of the full text**, as `[start_index, end_index]`.
- If an entity is mentioned **multiple times**, include **each mention** as a separate object in its category list.
- For compound names like "winter wheat", annotate only the species name (e.g., `"wheat"`).
- Do **not infer** — extract **only** what is **explicitly** stated in the text.
- If no entity is found for a category, return an empty list.
- Use the keys exactly as listed (e.g., `"soilPH"`, `"latitude"`).
### Text ###
    {text}
    """
    
    return system_prompt.strip()