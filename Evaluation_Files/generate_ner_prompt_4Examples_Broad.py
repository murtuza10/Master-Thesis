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
Title : \n  Long - term field trial on tillage and fertilization in crop rotation - Westerfeld \n Abstract_text_1 : \n  Long - term field trial on tillage and fertilization in crop rotation - Westerfeld . \n Abstract_text_2 : \n  The long - term field trial started 1992 in Bernburg , Saxony Anhalt , Germany ( 51 \u00b0 82 ' N , 11 \u00b0 70 ' , 138 m above sea level ) . The soil is a loess chernozem over limestone with an effective rooting depth of 100 cm , containing 22 % clay , 70 % silt and 8 % sand in the ploughed upper ( Ap ) horizon . It has a neutral pH ( 7.0 - 7.4 ) and an appropriate P and K supply ( 45 - 70 mg kg-1 and 130 - 185 mg kg-1 in calcium - acetate - lactate extract , respectively ) . The 1980 - 2010 annual average temperature was 9.7 \u00b0 C , the average annual precipitation 511 mm . On five large parcels in strip split plot design ( 1.2 ha each , main plots ) the individual crops - grain maize ( Zea mays ) - winter wheat ( Triticum aestivum ) - winter barley ( Hordeum vulgare ) - winter rapeseed ( Brassica napus ssp . napus ) - winter wheat - are rotated . All crop residues remain on the fields . Conservation tillage cultivator ( CT , 10 cm flat non - inversion soil loosening ) is compared to sub - plots with conventional tillage ( MP ; mould - board plough , carrier board with combined alternating ploughshares , ploughing depth 30 cm , incl . soil inversion ) . The differentially managed soils are either intensively ( Int ) operated according to usual practice regarding N supply and pesticide application or extensively managed ( Ext ; reduced N supply , without fungicides and growth regulators ) . \n Keywords : \n  Soil , fertilizers , tillage , agricultural land management , crop rotation , opendata , Boden". 
Output:
```json
 {"Crops": [{"cropSpecies": {"value": "maize", "span": [904, 909]}}, {"cropSpecies": {"value": "Zea mays", "span": [912, 920]}}, {"cropSpecies": {"value": "wheat", "span": [932, 937]}}, {"cropSpecies": {"value": "Triticum aestivum", "span": [940, 957]}}, {"cropSpecies": {"value": "barley", "span": [969, 975]}}, {"cropSpecies": {"value": "Hordeum vulgare", "span": [978, 993]}}, {"cropSpecies": {"value": "rapeseed", "span": [1005, 1013]}}, {"cropSpecies": {"value": "Brassica napus", "span": [1016, 1030]}}, {"cropSpecies": {"value": "wheat", "span": [1054, 1059]}}], "Soil": [{"Soil": {"value": "chernozem", "span": [378, 387]}}, {"Soil": {"value": "depth", "span": [429, 434]}}, {"Soil": {"value": "100 cm", "span": [438, 444]}}, {"Soil": {"value": "clay", "span": [463, 467]}}, {"Soil": {"value": "silt", "span": [475, 479]}}, {"Soil": {"value": "sand", "span": [488, 492]}}, {"Soil": {"value": "pH", "span": [549, 551]}}, {"Soil": {"value": "7.0", "span": [554, 557]}}, {"Soil": {"value": "7.4", "span": [560, 563]}}, {"Soil": {"value": "depth", "span": [1348, 1353]}}, {"Soil": {"value": "30 cm", "span": [1354, 1359]}}], "Location": [{"city": {"value": "Westerfeld", "span": [83, 93]}}, {"city": {"value": "Westerfeld", "span": [189, 199]}}, {"city": {"value": "Bernburg", "span": [269, 277]}}, {"country": {"value": "Germany", "span": [296, 303]}}], "Time Statement": [{"startTime": {"value": "1992", "span": [261, 265]}}, {"startTime": {"value": "1980", "span": [702, 706]}}, {"endTime": {"value": "2010", "span": [709, 713]}}]}

### Example 2 ###:
  Input Text:
Input Text:
Title : \n Assessing SNP - markers to study population mixing and ecological adaptation in Baltic cod \n\n Abstract : \n This project was partly funded by the European Maritime and Fisheries Fund ( EMFF ) of the European Union ( EU ) under the Data Collection Framework ( DCF , Regulation 2017/1004 of the European Parliament and of the Council ) . Cod individuals from the Bornholm Basin were collected during RV ALKOR cruises . CP and JD were supported by the BONUS BIO - C3 project , which has received funding from BONUS ( Art 185 ) , funded jointly by the EU and from national funding institutions including the German BMBF under grant No . 03F0682 . Computational analyses were performed on the Abel Cluster owned by the UiO and the Norwegian metacenter for High Performance Computing ( NOTUR ) and operated by the UiO Department for Research Computing . The whole genome sequencing of cod samples and SNP identification for the minimal panel were funded by \u201c The Aqua Genome Project \u201d ( 221734 / O30 ) through the Research Council of Norway .".
Output:
```json
{"Crops": [], "Soil": [], "Location": [{"region": {"value": "Bornholm Basin", "span": [370, 384]}}], "Time Statement": []}

### Example 3 ###:
    Input Text:
Title : \n Nachruf f\u00fcr Prof. Dr. Edgar Mai\u00df \n\n Abstract : \n Not Found".
Output:
```json
{"Crops": [], "Soil": [], "Location": [], "Time Statement": []}

### Example 4 ###:
    Input Text:
Title : \n Th\u00fcnen - Baseline 2024 - 2034 : Agrar\u00f6konomische Projektionen f\u00fcr Deutschland \n\n Abstract : \n Dieser Bericht stellt ausgew\u00e4hlte Ergebnisse der Th\u00fcnen - Baseline 2024 - 2034 sowie die zugrunde liegenden Annahmen dar . Die Th\u00fcnen - Baseline ist ein Basisszenario und beschreibt die zuk\u00fcnftige Entwicklung der Agrarm\u00e4rkte unter definierten politischen und wirtschaftlichen Rahmenbedingungen . Zentrale Annahmen sind die Beibehaltung der derzeitigen Agrarpolitik und Umsetzung bereits beschlossener Politik - \u00e4nderungen sowie die Fortschreibung exogener Einflussfaktoren auf Basis historischer Trends . Die Berechnungen beruhen auf Daten und Informationen , die bis zum Fr\u00fchjahr 2024 vorlagen . Dargestellt werden Projektionsergebnisse f\u00fcr Agrarhandel , Preise , Nachfrage , Produktion , Einkommen und Umweltwirkungen . Die Darstellung der Ergebnisse konzentriert sich haupts\u00e4chlich auf die Entwicklungen des deutschen Agrarsektors bis zum Jahr 2034 i m Vergleich zum Durchschnitt der Basisperiode 2020 2022 . Die Ergebnisse zeigen , dass die EU ihre Position i m weltweiten Agrarhandel bis zum Jahr 2034 behaupten kann . Die Preise f\u00fcr Agrarprodukte sinken zu Beginn der Projektionsperiode vom hohen Niveau des Basisjahres , k\u00f6nnen sich bis zum Jahr 2034 jedoch wieder erholen . In Deutschland entwickelt sich der Anbau von Getreide r\u00fcckl\u00e4ufig , was auf ver\u00e4nderte Preiserelationen sowie einen R\u00fcckgang der landwirtschaftlich genutzten Fl\u00e4che zur\u00fcckzuf\u00fchren ist . I m Tiersektor setzt sich der in den letzten Jahren beobachtete Abbau der Tierbest\u00e4nde und R\u00fcckgang der Fleischerzeugung fort , insbesondere in der Schweinehaltung , wohingegen die Gefl\u00fcgelfleischerzeugung bis zum Jahr 2034 noch leicht w\u00e4chst . Eine positive Preisentwicklung am Milchmarkt in Verbindung mit einer weiteren Steigerung der Milchleistung f\u00fchren au\u00dferdem zu einem moderaten Anstieg der Milchanlieferungen . Das durchschnittliche reale Einkommen landwirtschaftlicher Betriebe geht \u00fcber die Projektionsperiode um 17 Prozent zur\u00fcck und liegt damit i m Jahr 2034 wieder auf dem mittleren Niveau der letzten zehn Jahre .",
Output:
```json
{"Crops": [], "Soil": [], "Location": [{"country": {"value": "Deutschland", "span": [76, 87]}}, {"country": {"value": "Deutschland", "span": [1289, 1300]}}], "Time Statement": [{"startTime": {"value": "2024", "span": [28, 32]}}, {"endTime": {"value": "2034", "span": [35, 39]}}, {"startTime": {"value": "2024", "span": [171, 175]}}, {"endTime": {"value": "2034", "span": [178, 182]}}, {"endTime": {"value": "Fr\u00fchjahr 2024", "span": [676, 689]}}, {"endTime": {"value": "2034", "span": [951, 955]}}, {"startTime": {"value": "2020", "span": [1004, 1008]}}, {"endTime": {"value": "2022", "span": [1009, 1013]}}, {"endTime": {"value": "2034", "span": [1106, 1110]}}, {"endTime": {"value": "2034", "span": [1257, 1261]}}, {"endTime": {"value": "2034", "span": [1690, 1694]}}, {"endTime": {"value": "2034", "span": [2038, 2042]}}, {"duration": {"value": "zehn Jahre", "span": [2087, 2097]}}]}

 
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