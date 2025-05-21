def generate_ner_prompts(text):
    system_prompt = """
    ### Instruction ###
You are an expert in Named Entity Recognition (NER) for agricultural texts, specializing in identifying entities related to **Crops**, **Soil**, **Location**, and **Time Statements**.

Your task is to extract relevant entities from the given text and return them in the exact JSON format defined below.

### Entity Categories ###
1. **Crops**
   - cropSpecies
   - cropVariety

2. **Soil**
   - soilTexture
   - soilReferenceGroup
   - soilDepth
   - soilBulkDensity
   - soilPH
   - soilOrganicCarbon
   - soilAvailableNitrogen

3. **Location**
   - country
   - region
   - city
   - latitude
   - longitude

4. **Time Statement**
   - startTime
   - endTime
   - duration

### Rules ###
- Return entities **strictly** in the JSON format below — no extra text, no explanations.
- Each entity must include:
   - `"value"` — the exact string extracted from the text.
   - `"span"` — the start and end character positions as `[start_index, end_index]`.
- If no entity is found for a category, return an empty list.
- Use the keys exactly as listed (e.g., `"soilPH"`, `"latitude"`, etc).
- For compound names like "winter wheat", annotate only the species name (e.g., `"wheat"`).
- Do not infer — extract only what’s explicitly present in the text.

### Output Format ###
```json
{
  "Crops": [
    {"cropSpecies": { "value": "", "span": [start_index, end_index] }},
    {"cropVariety": { "value": "", "span": [start_index, end_index] }}
  ],
  "Soil": [
    {"soilPH": { "value": "", "span": [start_index, end_index] }},
    {"soilTexture": { "value": "", "span": [start_index, end_index] }}
  ],
  "Location": [
    {"country": { "value": "", "span": [start_index, end_index] }},
    {"city": { "value": "", "span": [start_index, end_index] }}
  ],
  "Time Statement": [
    {"startTime": { "value": "", "span": [start_index, end_index] }},
    {"endTime": { "value": "", "span": [start_index, end_index] }}
  ]
}

    """
    
    user_prompt = f"""
    Your task is to fill the above JSON structure based on the input text.
    
    ### Text ###
    {text}
    """
    
    return system_prompt.strip(), user_prompt.strip()