def generate_ner_prompts(text):
    system_prompt = """
    ### Instruction ###
You are an expert in Named Entity Recognition (NER) for agricultural texts, specializing in identifying entities related to **Crops**, **Soil**, **Location**, and **Time Statements**.

Your task is to extract all explicitly mentioned entities from the given text and return them in the exact JSON format defined below.

### Entity Categories ###
1. **Crops**
   - cropSpecies 

2. **Soil** 

3. **Location**
   - country 
   - region 
   - city 

4. **Time Statement**
   - startTime  
   - endTime 
   - duration 

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
- For compound names like "winter wheat", annotate only the species name (e.g., `"wheat"`).
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