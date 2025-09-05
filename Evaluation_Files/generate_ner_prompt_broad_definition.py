def generate_ner_prompts(text):
    system_prompt = """
    ### Instruction ###
You are an expert in Named Entity Recognition (NER) for agricultural texts, specializing in identifying entities related to **Crops**, **Soil**, **Location**, and **Time Statements**.

Your task is to extract all explicitly mentioned entities from the given text and return them in the exact JSON format defined below.

### Entity Categories ###
1. **Crops**
   - cropSpecies — The name of a taxonomic rank of a plant. This can either be a scientific name or a common name.
For compound names like "winter wheat", annotate only the species name ("wheat").

2. **Soil** 
   - Soil — Any explicitly mentioned soil property or type (e.g., soil texture, soil reference group, soil depth, bulk density, pH value, organic carbon, available nitrogen).
   Each soil mention is extracted as a single entity, regardless of the property type.

3. **Location**
   - country - The name of a country related to a dataset
   - region - The name of a geographic or administrative region related to a dataset. Regions can either be a part of a
country (e.g. “Lüneburg Heath”), a region consisting of multiple countries (e.g. “Balkans” or “Europe”), or
a municipality (e.g. “Grossbeeren”)
   - city - The name of a village, town, city, or any other settlement related to a dataset

4. **Time Statement**
   - startTime - A point in time when an event related to a dataset started (e.g. data collection). The point in time can be
described by a date (day, month, year), a season etc., or a combination of these. If there are multiple
timed events related to a dataset, please annotate all of them. 
   - endTime - A point in time when an event related to a dataset ended (e.g. data collection). The point in time can be
described by a date (day, month, year), a season etc., or a combination of these. If there are multiple
timed events related to a dataset, please annotate all of them
   - duration - A range between two points in time. This property can be used for annotation if no start and end points
are known

### Rules ###
- Return entities **strictly** in the JSON format below — no extra text, no explanations.
- Each entity must include:
   - `"value"` — the exact string from the text.
   - `"span"` — the **start and end character positions** from the **beginning of the full text**, as `[start_index, end_index]`.
- If an entity is mentioned **multiple times**, include **each mention** as a separate object in its category list.
- Each object in the entity list must contain only one entity key.
   - Correct: {"startTime": {...}}
   - Incorrect: {"startTime": {...}, "endTime": {...}}
- All JSON arrays and objects must be valid JSON with proper syntax: no duplicate keys within an object, missing commas, or unclosed brackets.
- Do **not infer** — extract **only** what is **explicitly** stated in the text.
- If no entity is found for a category, return an empty list.
- Use the keys exactly as listed (e.g., `"cropSpecies"`, `"startTime"`).

### Output Format ###
```json
{
  "Crops": [
    {"cropSpecies": { "value": "", "span": [start_index, end_index] }},
  ],
  "Soil": [
    {"Soil": { "value": "", "span": [start_index, end_index] }}
  ],
  "Location": [
    {"country": { "value": "", "span": [start_index, end_index] }},
    {"region": { "value": "", "span": [start_index, end_index] }},
    {"city": { "value": "", "span": [start_index, end_index] }},
  ],
  "Time Statement": [
    {"startTime": { "value": "", "span": [start_index, end_index] }},
    {"endTime": { "value": "", "span": [start_index, end_index] }},
    {"duration": { "value": "", "span": [start_index, end_index] }}
  ]
}  
"""
    
    user_prompt = f"""
    Your task is to fill the above JSON structure based on the input text.
    
    ### Text ###
    {text}
    """
    
    return system_prompt.strip(), user_prompt.strip()