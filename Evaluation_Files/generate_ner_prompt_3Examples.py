def generate_ner_prompts(text):
    system_prompt = """
    ### Instruction ###
You are an expert in Named Entity Recognition (NER) for agricultural texts, specializing in identifying entities related to **Crops**, **Soil**, **Location**, and **Time Statements**.

Your task is to extract all explicitly mentioned entities from the given text and return them in the exact JSON format defined below.

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
   - `"value"` — the exact string from the text.
   - `"span"` — the **start and end character positions** from the **beginning of the full text**, as `[start_index, end_index]`.
- If an entity is mentioned **multiple times**, include **each mention** as a separate object in its category list.
- Each object in the entity list must contain only one entity key.
   - Correct: {"cropSpecies": {...}}
   - Incorrect: {"cropSpecies": {...}, "cropVariety": {...}}
- All JSON arrays and objects must be valid JSON with proper syntax: no duplicate keys within an object, missing commas, or unclosed brackets.
- For compound names like "winter wheat", annotate only the species name (e.g., `"wheat"`).
- Do **not infer** — extract **only** what is **explicitly** stated in the text.
- If no entity is found for a category, return an empty list.
- Use the keys exactly as listed (e.g., `"soilPH"`, `"latitude"`).

### Example 1 ###:
  Input Text:
Title : \n Partnerinformationsveranstaltungen zur Begleitung von PFEIL - Programm zur F\u00f6rderung der Entwicklung i m l\u00e4ndlichen Raum Niedersachsen und Bremen 2014 bis 2020 : Ergebnisse einer Online - Befragung von Partnern \n\n Abstract : \n Not FoundOutput :
```json
{"Crops": [], "Soil": [], "Location": [{"region": {"value": "Niedersachsen", "span": [131, 144]}}, {"city": {"value": "Bremen", "span": [149, 155]}}], "Time Statement": [{"startTime": {"value": "2014", "span": [156, 160]}}, {"endTime": {"value": "2020", "span": [165, 169]}}]}

### Example 2 ###:
  Input Text:
Title : \n Bodendruck , Deformation und \u00c4nderungen der bodenphysikalischen Parameter verursacht durch die Silomais - Erntekette auf einer Parabraunerde aus L\u00f6ss i m Jahr 2017 \n\n Abstract : \n Dieser Datensatz enth\u00e4lt Bodendaten aus einem Befahrungsversuch mit landwirtschaftlichen Fahrzeugen der Silomaish\u00e4ckselkette ( Maish\u00e4cksler : ca . 20 t , Traktor mit Mulcher : ca.12 t , Traktor mit Silowagen : ca . 32 t ) . Der Versuch fand 2017 w\u00e4hrend der Ernte statt . Der Boden , eine pseudovergleyte Parabraunerde aus L\u00f6ss , wurde bis 25 cm Tiefe bearbeitet . Die Fahrzeuge fuhren nacheinander \u00fcber definierte Plots i m Kernfeldbereich ( 8 Rad\u00fcberrollungen ) . Vor und nach den Befahrungen wurden je Variante 2 Gruben ausgehoben . Folgende Messungen wurde in 20 , 35 und 50 cm Tiefe je Grube durchgef\u00fchrt : Bodendruck und plastische Bodensetzung Wasserretentionseigenschaften , Lagerungsdichte und ges\u00e4ttigte hydr . Leitf\u00e4higkeit ( 100 cm3 Stechzylinder ; je 5 Wiederholungen pro Grube und Tiefe ) . Daraus abgeleitet wurden Gesamtporenvolumen , Luftkapazit\u00e4t , Feldkapazit\u00e4t , permanenter Welkepunkt und Trocken - rohdichte Bioporen wurden in jeder Grube und Tiefe gez\u00e4hltOutput :
  ```json
{"Crops": [{"cropSpecies": {"value": "Silomais", "span": [105, 113]}}], "Soil": [{"soilReferenceGroup": {"value": "Parabraunerde", "span": [137, 150]}}, {"soilTexture": {"value": "L\u00f6ss", "span": [155, 159]}}, {"soilReferenceGroup": {"value": "pseudovergleyte Parabraunerde", "span": [479, 508]}}, {"soilTexture": {"value": "L\u00f6ss", "span": [513, 517]}}, {"soilDepth": {"value": "25 cm", "span": [530, 535]}}, {"soilDepth": {"value": "20", "span": [754, 756]}}, {"soilDepth": {"value": "35", "span": [759, 761]}}, {"soilDepth": {"value": "50 cm", "span": [766, 771]}}], "Location": [], "Time Statement": [{"startTime": {"value": "2017", "span": [169, 173]}}, {"startTime": {"value": "2017", "span": [431, 435]}}]}

### Example 3 ###:
    Input Text:
Title : \n Segmentation of wine berries \n\n Abstract : \n Dataset contains high resolution images collected with a moving field phenotyping platform , the Phenoliner . \n  The collected images show 3 different varieties ( Riesling , Felicia , Regent ) in 2 different training systems ( VSP = vertical shoot positioning and SMPH= semi minimal pruned hedges ) , collected in 2 points in time ( before and after thinning ) in 2018 . For each image we provide a manual masks which allow the identification of single berries . \n  The folder contains : 1 . List with image details ( imagename , acquisition date , year , variety , training system and variety number)and 2 . Dataset folder with 2 subfolders , namely 1 . img \u2013 42 original RGB images and 2 . lbl \u2013 42 corresponding labels ( manual annotation , with berry , edge , background definition ) \n  The data were used to train a neural network with the main goal to detect single berries in images . The method is described in detail in the specified papers .Output :
```json
{"Crops": [{"cropSpecies": {"value": "wine berries", "span": [26, 38]}}, {"cropVariety": {"value": "Riesling", "span": [218, 226]}}, {"cropVariety": {"value": "Felicia", "span": [229, 236]}}, {"cropVariety": {"value": "Regent", "span": [239, 245]}}], "Soil": [], "Location": [], "Time Statement": [{"startTime": {"value": "2018 .", "span": [419, 425]}}]}

 
 ### Output Format ###
```json
{
  "Crops": [
    {"cropSpecies": { "value": "", "span": [start_index, end_index] }},
    {"cropVariety": { "value": "", "span": [start_index, end_index] }}
  ],
  "Soil": [
    {"soilTexture": { "value": "", "span": [start_index, end_index] }},
    {"soilReferenceGroup": { "value": "", "span": [start_index, end_index] }},
    {"soilDepth": { "value": "", "span": [start_index, end_index] }},
    {"soilBulkDensity": { "value": "", "span": [start_index, end_index] }},
    {"soilPH": { "value": "", "span": [start_index, end_index] }},
    {"soilOrganicCarbon": { "value": "", "span": [start_index, end_index] }},
    {"soilAvailableNitrogen": { "value": "", "span": [start_index, end_index] }}
  ],
  "Location": [
    {"country": { "value": "", "span": [start_index, end_index] }},
    {"region": { "value": "", "span": [start_index, end_index] }},
    {"city": { "value": "", "span": [start_index, end_index] }},
    {"latitude": { "value": "", "span": [start_index, end_index] }},
    {"longitude": { "value": "", "span": [start_index, end_index] }}
  ],
  "Time Statement": [
    {"startTime": { "value": "", "span": [start_index, end_index] }},
    {"endTime": { "value": "", "span": [start_index, end_index] }},
    {"duration": { "value": "", "span": [start_index, end_index] }}
  ]
}  
     """
   
    user_prompt = f"""
    Your task is to fill the above JSON structure based on the input text below.
    
    ### Text ###
    {text}
    """
    
    return system_prompt.strip(), user_prompt.strip()