def generate_ner_prompts(text):
    system_prompt = """
  You are an expert in Named Entity Recognition (NER) for agricultural texts.  
Your task is to extract entities **only** from the explicitly allowed subcategories below and return them in JSON.  

---

### Allowed Entity Subcategories ###

1. Crop  
   - cropSpecies — Scientific or common species name of a crop (e.g., "wheat", "Zea mays").  
   - cropVariety — Specific cultivar/variety name (e.g., "Golden Delicious").  

2. Location  
   - country — Country name.  
   - region — Geographic/administrative region.  
   - city — Settlement name (village, town, city).  
   - latitude — Latitude coordinate.  
   - longitude — Longitude coordinate.  

3. TimeStatement  
   - startTime — Explicit start date/season/year.  
   - endTime — Explicit end date/season/year.  
   - duration — Explicit range (when start and end are not separately given).  

---

### Rules ###

1. Extract **only** entities that belong to the above subcategories.  
2. **If an entity does not belong to one of the allowed subcategories, you must completely ignore it and do not output it at all.**  
3. Each mention of an entity must be returned separately.  
4. If no entities exist for a category, return an empty list.  
5. Do **not infer or generalize** entities (e.g., do not map “field” to a location).  
6. The entity key must always be exactly `"Crop"`, `"Location"`, or `"TimeStatement"`.  
7. Return JSON exactly in this structure:  

```json
{
  "Crop": [
    {"Crop": { "value": "", "span": [start_index, end_index] }}
  ],
  "Location": [
    {"Location": { "value": "", "span": [start_index, end_index] }}
  ],
  "TimeStatement": [
    {"TimeStatement": { "value": "", "span": [start_index, end_index] }}
  ]
}
"""
    
    user_prompt = f"""
    Your task is to fill the above JSON structure based on the input text.
    
    ### Text ###
    {text}
    """
    
    return system_prompt.strip(), user_prompt.strip()