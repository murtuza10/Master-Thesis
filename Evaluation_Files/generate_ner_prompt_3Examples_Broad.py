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
Title : \n  Long - term field trial on tillage and fertilization in crop rotation - Westerfeld \n Abstract_text_1 : \n  Long - term field trial on tillage and fertilization in crop rotation - Westerfeld . \n Abstract_text_2 : \n  The long - term field trial started 1992 in Bernburg , Saxony Anhalt , Germany ( 51 \u00b0 82 ' N , 11 \u00b0 70 ' , 138 m above sea level ) . The soil is a loess chernozem over limestone with an effective rooting depth of 100 cm , containing 22 % clay , 70 % silt and 8 % sand in the ploughed upper ( Ap ) horizon . It has a neutral pH ( 7.0 - 7.4 ) and an appropriate P and K supply ( 45 - 70 mg kg-1 and 130 - 185 mg kg-1 in calcium - acetate - lactate extract , respectively ) . The 1980 - 2010 annual average temperature was 9.7 \u00b0 C , the average annual precipitation 511 mm . On five large parcels in strip split plot design ( 1.2 ha each , main plots ) the individual crops - grain maize ( Zea mays ) - winter wheat ( Triticum aestivum ) - winter barley ( Hordeum vulgare ) - winter rapeseed ( Brassica napus ssp . napus ) - winter wheat - are rotated . All crop residues remain on the fields . Conservation tillage cultivator ( CT , 10 cm flat non - inversion soil loosening ) is compared to sub - plots with conventional tillage ( MP ; mould - board plough , carrier board with combined alternating ploughshares , ploughing depth 30 cm , incl . soil inversion ) . The differentially managed soils are either intensively ( Int ) operated according to usual practice regarding N supply and pesticide application or extensively managed ( Ext ; reduced N supply , without fungicides and growth regulators ) . \n Keywords : \n  Soil , fertilizers , tillage , agricultural land management , crop rotation , opendata , Boden.
Output:
```json
{"Crops": [{"cropSpecies": {"value": "maize", "span": [904, 909]}}, {"cropSpecies": {"value": "Zea mays", "span": [912, 920]}}, {"cropSpecies": {"value": "wheat", "span": [932, 937]}}, {"cropSpecies": {"value": "Triticum aestivum", "span": [940, 957]}}, {"cropSpecies": {"value": "barley", "span": [969, 975]}}, {"cropSpecies": {"value": "Hordeum vulgare", "span": [978, 993]}}, {"cropSpecies": {"value": "rapeseed", "span": [1005, 1013]}}, {"cropSpecies": {"value": "Brassica napus", "span": [1016, 1030]}}, {"cropSpecies": {"value": "wheat", "span": [1054, 1059]}}], "Soil": [{"Soil": {"value": "chernozem", "span": [378, 387]}}, {"Soil": {"value": "depth", "span": [429, 434]}}, {"Soil": {"value": "100 cm", "span": [438, 444]}}, {"Soil": {"value": "clay", "span": [463, 467]}}, {"Soil": {"value": "silt", "span": [475, 479]}}, {"Soil": {"value": "sand", "span": [488, 492]}}, {"Soil": {"value": "pH", "span": [549, 551]}}, {"Soil": {"value": "7.0", "span": [554, 557]}}, {"Soil": {"value": "7.4", "span": [560, 563]}}, {"Soil": {"value": "depth", "span": [1348, 1353]}}, {"Soil": {"value": "30 cm", "span": [1354, 1359]}}], "Location": [{"city": {"value": "Westerfeld", "span": [83, 93]}}, {"city": {"value": "Westerfeld", "span": [189, 199]}}, {"city": {"value": "Bernburg", "span": [269, 277]}}, {"country": {"value": "Germany", "span": [296, 303]}}], "Time Statement": [{"startTime": {"value": "1992", "span": [261, 265]}}, {"startTime": {"value": "1980", "span": [702, 706]}}, {"endTime": {"value": "2010", "span": [709, 713]}}]}

### Example 2 ###:
  Input Text:
Title : \n Evidence of hybridization betweenn genetically distinct Baltic cod stocks during peak population abundance - historical \n\n Abstract : \n Range expansions can lead to increased contact of divergent populations , thus increasing the potential of hybridization events . Whether viable hybrids are produced will most likely depend on the level of genomic divergence and associated genomic incompatibilities between the different entities as well as environmental conditions . By taking advantage of historical Baltic cod ( Gadus morhua ) otolith samples combined with genotyping and whole genome sequencing , we here investigate the genetic impact of the increased spawning stock biomass of the eastern Baltic cod stock in the mid 1980s . The eastern Baltic cod is genetically highly differentiated from the adjacent western Baltic cod , and locally adapted to the brackish environmental conditions in the deeper Eastern basins of the Baltic Sea unsuitable for its marine counterparts . Our genotyping results show an increased proportion of eastern Baltic cod in western Baltic areas ( Mecklenburg Bay and Arkona Basin ) \u2013 indicative of a range expansion westwards \u2013 during the peak population abundance in the 1980s . Additionally , we detect high frequencies of potential hybrids ( including F1 , F2 and backcrosses ) , verified by whole genome sequencing data for a subset of individuals . Analysis of mitochondrial genomes further indicates directional gene flow from eastern Baltic cod males to western Baltic cod females . Our findings unravel that increased overlap in distribution can promote hybridization between highly divergent populations , and that the hybrids can be viable and survive under specific and favourable environmental conditions . However , the observed hybridization had seemingly no long - lasting impact on the continuous separation and genetic differentiation between the unique Baltic cod stocks.
Output:
```json
{"Crops": [], "Soil": [], "Location": [{"region": {"value": "Baltic Sea", "span": [940, 950]}}, {"region": {"value": "Baltic", "span": [1077, 1083]}}, {"region": {"value": "Mecklenburg Bay", "span": [1092, 1107]}}, {"region": {"value": "Arkona Basin", "span": [1112, 1124]}}], "Time Statement": [{"startTime": {"value": "1980s", "span": [736, 741]}}, {"startTime": {"value": "1980s", "span": [1217, 1222]}}]}

### Example 3 ###:
    Input Text:
Title : \n Fatty acid analyses reveal differences in feeding ecology of North Sea squids that overlap in time and space \n\n Abstract : \n Climate - induced changes in marine ecosystems have been documented worldwide . As one of the main consequences , a shift in the distribution of species is observable in many marine areas , resulting in the formation of new species communities and new interactions . In the North Sea , the squid community has changed considerably over the last 100 years . Some species have disapeared while new species have established and are now living in coexistence in a new community . Although squids are considered to be predators that feed rather non - selectively , we aimed to answer the question of whether their diet differs nevertheless . Therefore , we analysed the fatty acids of three squid species whose distribution substantially overlaps . We were able to recognise a dependence between the size of the squid and the composition of fatty acids and are able to demonstrate the already known ontogenetic shift in food composition on the basis of fatty acid composition . Furthermore , we illustrate that the fatty acid composition differs significantly between squid species , which points to different prey of the analysed squid species and which may be one reason for their successful coexistence.
Output:
```json
{"Crops": [], "Soil": [], "Location": [{"region": {"value": "North Sea", "span": [71, 80]}}, {"region": {"value": "North Sea", "span": [409, 418]}}], "Time Statement": [{"duration": {"value": "100 years", "span": [480, 489]}}]}

 
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