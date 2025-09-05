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

### Example 1 ###:
    Input Text:
 Title : \n  50 years box plot experiment in Grossbeeren ( 1972 - 2022 ) - Plots \n Abstract_text_1 : \n  50 years box plot experiment in Grossbeeren ( 1972 - 2022 ) - Plots . \n Abstract_text_2 : \n  The Box Plot Experiment in Grossbeeren was set up in 1972 to investigate the effect of different fertilization strategies within an irrigated vegetable crop rotation system for three different soils . Therefore , this vegetable long - term fertilization experiment can be used to investigate different plant - soil - systems under the same climatic conditions . The experimented was halted in 2022 . The experimental site ( 52 \u00b0 21\u201901.30 \u2019\u2019 E , 13 \u00b0 19\u201905.47 \u2019\u2019 N , 50 m a.s.l . ) is located in the transition zone between the more maritime - affected Northern German Plain and the continental climate of the European mainland . Weather data were collected in an agrometeorological station close to the experimental area . The long - term means ( 1991 - 2020 ) for air temperature and annual precipitation are 9.7 \u00b0 C and 492 mm . The single plots are quadratic concrete boxes with walls of 10 cm thickness , a surface area of 4 m2 and a depth of 75 cm . The upper 50 cm are filled with the tested soils ; the lower 25 cm comprises a coarse - sandy drainage layer . The three soil types are Arenic Luvisol ( weak loamy sand ) , Gleyic Fluvisol ( heavy sandy loam ) and Luvic - Phaeozem ( medium clayey silt ) according to the World Reference Base \u2013 WRB ( and the Bodenkundliche Kartieranleitung \u2013 KA4 ) . Within 10 rotations , the vegetable species white cabbage ( Brassica oleracea L. var . capitata f. alba ) , carrot ( Daucus carota L. ) , cucumber ( Cucumis sativus L. ) , leek ( Allium porrum L. ) and celery ( Apium graveolens L. var . rapaceum Mill . ) were cultivated . No celery was cultivated during the first rotation . The experiment consists of 12 fertilization treatments in different combinations of mineral N fertilization and organic amendments and as quadruplicate for each of the three soils . The experimental set - up scheme can be found in the supplementary material . Mineral N fertilizer was applied as calcium ammonium nitrate . Mineral P and K fertilization was uniform for all treatments . Total N and total C in soil , plant and organic amendments were determined using a CNS analyser VARIO El ( Elemental Hanau ) since 1995 and before by wet combustion with K2Cr2O7 und H2SO4 . C and N in the soil samples and N in the plant samples were analysed annually . The C contents of the crop residues ( leaf + stalk + root ) of the five vegetable species were investigated irregularly . In autumn , the soil was annually dug up to 20 cm by using a spade . Weeds were removed by a combination of mechanical ( cultivator , rake or hoe ) and chemical measures . Insect protection nets , insecticides or fungicides were used where necessary . Approximately 150 mm per year was additionally irrigated with a sprinkler system . More details about the experiment \u2019s description can be found in the supplementary material . Description of table 1 Related datasets are listed in the metadata element ' Related Identifier ' . Dataset version 1.0 \n Keywords : \n  horticulture , long - term experiments , vegetable crops , fertilization , fertilizers , soil types , soil fertility , soil organic carbon , soil organic matter , field crops , crop management , crop production , crop rotation , crop residues , crop residue management , crop yield , nutrient balance , nutrient management , nutrient uptake , nutrient use efficiency , nutrient utilization , nitrogen , nitrogen balance , nitrogen content , nitrogen fertilizers , nitrogen - use efficiency , potassium , phosphorus , magnesium , cucumbers , Cucumis , Cucumis sativus , carrots , Daucus carota , cabbages , Brassica oleracea var . capitata , leeks , Allium ampeloprasum , celery , Apium graveolens , Apium graveolens var . rapaceum , farmyard manure , organic amendments , organic fertilizers , slurry , bark mulches , resource management , Luvisols , Fluvisols , Phaeozems , opendata , , Boden , agricultural management , horticulture , crop production , crop rotation , crop waste , cultivation , cultivation system , cultivation method , food production ( agriculture ) , irrigation farming , manure , mineral fertiliser , nitrogenous fertiliser , organic fertiliser , soil fertilisation , soil fertility , vegetable , vegetable cultivation , vegetable waste , yield ( agricultural ) , resource utilisation , organic matter , phosphate
Output :
```json
 {"Crops": [{"cropSpecies": {"value": "Brassica oleracea L.", "span": [1560, 1580]}}, {"cropSpecies": {"value": "Daucus carota L.", "span": [1617, 1633]}}, {"cropSpecies": {"value": "Cucumis sativus L.", "span": [1649, 1667]}}, {"cropSpecies": {"value": "Allium", "span": [1679, 1685]}}, {"cropSpecies": {"value": "celery", "span": [1702, 1708]}}, {"cropSpecies": {"value": "Apium graveolens L.", "span": [1711, 1730]}}, {"cropSpecies": {"value": "celery", "span": [1776, 1782]}}, {"cropSpecies": {"value": "cucumbers", "span": [3698, 3707]}}, {"cropSpecies": {"value": "Cucumis", "span": [3710, 3717]}}, {"cropSpecies": {"value": "Cucumis sativus", "span": [3720, 3735]}}, {"cropSpecies": {"value": "carrots", "span": [3738, 3745]}}, {"cropSpecies": {"value": "Daucus carota", "span": [3748, 3761]}}, {"cropSpecies": {"value": "cabbages", "span": [3764, 3772]}}, {"cropSpecies": {"value": "leeks", "span": [3810, 3815]}}, {"cropSpecies": {"value": "Allium ampeloprasum", "span": [3818, 3837]}}, {"cropSpecies": {"value": "celery", "span": [3840, 3846]}}, {"cropSpecies": {"value": "Apium graveolens", "span": [3849, 3865]}}], "Soil": [{"Soil": {"value": "depth", "span": [1133, 1138]}}, {"Soil": {"value": "75 cm", "span": [1142, 1147]}}, {"Soil": {"value": "Arenic Luvisol", "span": [1286, 1300]}}, {"Soil": {"value": "weak loamy sand", "span": [1303, 1318]}}, {"Soil": {"value": "Gleyic Fluvisol", "span": [1323, 1338]}}, {"Soil": {"value": "heavy sandy loam", "span": [1341, 1357]}}, {"Soil": {"value": "Luvic - Phaeozem", "span": [1364, 1380]}}, {"Soil": {"value": "medium clayey silt", "span": [1383, 1401]}}, {"Soil": {"value": "soil organic carbon", "span": [3288, 3307]}}, {"Soil": {"value": ", Luvisols", "span": [4007, 4017]}}, {"Soil": {"value": ", Fluvisols", "span": [4018, 4029]}}, {"Soil": {"value": ", Phaeozems", "span": [4030, 4041]}}], "Location": [{"city": {"value": "Grossbeeren", "span": [43, 54]}}, {"city": {"value": "Grossbeeren", "span": [134, 145]}}, {"city": {"value": "Grossbeeren", "span": [222, 233]}}, {"region": {"value": "Northern German Plain", "span": [747, 768]}}, {"city": {"value": "Hanau", "span": [2329, 2334]}}], "Time Statement": [{"duration": {"value": "50 years", "span": [11, 19]}}, {"startTime": {"value": "1972", "span": [57, 61]}}, {"endTime": {"value": "2022", "span": [64, 68]}}, {"duration": {"value": "50 years", "span": [102, 110]}}, {"startTime": {"value": "1972", "span": [148, 152]}}, {"endTime": {"value": "2022", "span": [155, 159]}}, {"startTime": {"value": "1972", "span": [248, 252]}}, {"endTime": {"value": "2022", "span": [588, 592]}}, {"startTime": {"value": "1991", "span": [942, 946]}}, {"endTime": {"value": "2020", "span": [949, 953]}}, {"startTime": {"value": "1995", "span": [2343, 2347]}}]}

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
    Your task is to fill the above JSON structure based on the input text below.
    
    ### Text ###
    {text}
    """
    
    return system_prompt.strip(), user_prompt.strip()