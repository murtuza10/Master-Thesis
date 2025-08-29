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
Title : \n 2020_Helfrich_AGEE \n\n Abstract : \n Datenmeldung nach dem EGovG. Der Datensatz kann auf Anfrage zur Verf\u00fcgung gestellt werden . Eine Datennutzungsvereinbarung wird abgeschlossen . Grassland conversion to cropland bears a risk of increased nitrate ( NO3\u2212 ) leaching and nitrous oxide ( N2O ) emission due to enhanced nitrogen ( N ) mineralization . This study investigates the dynamics of mineral N and N2O emissions following chemical and mechanical conversion from permanent grassland to cropland ( maize ) at two sites with different texture ( clayey loam and sandy loam ) and fertilization regime ( with and without mineral N - fertilization ) over a two - year period . Soil mineral N levels increased shortly after conversion and remained elevated in converted plots compared to permanent grassland or long - term cropland in the second year of investigation . Fluxes of N2O were higher from converted plots than permanent grassland or cropland . However , soil mineral N contents and cumulative N2O emissions did not differ between conversion types . Only the distribution of N2O losses over the two years differed : while losses were of similar magnitude in both years in mechanically converted plots , the major part of N2O loss in chemically converted plots occurred in the first year after conversion while emissions approximated grassland level in the second year . N2O fluxes were mainly controlled by water - filled pore space and soil NO3\u2212 levels . Despite differences in N levels at the two sites , these key findings are similar on both study sites . They indicate strongly accelerated mineralization after conversion , an effect that still lasted in the converted plots at the end of the two - year investigation irrespective of the conversion type used .
Output :
```json
{"Crops": [{"cropSpecies": {"value": "maize", "span": [509, 514]}}], "Soil": [{"soilTexture": {"value": "clayey loam", "span": [555, 566]}}, {"soilTexture": {"value": "sandy loam", "span": [571, 581]}}], "Location": [], "Time Statement": [{"duration": {"value": "two - year period", "span": [663, 680]}}]}

### Example 2 ###:
  Input Text:
Title : \n Th\u00fcnen - Baseline 2024 - 2034 : Agrar\u00f6konomische Projektionen f\u00fcr Deutschland \n\n Abstract : \n Dieser Bericht stellt ausgew\u00e4hlte Ergebnisse der Th\u00fcnen - Baseline 2024 - 2034 sowie die zugrunde liegenden Annahmen dar . Die Th\u00fcnen - Baseline ist ein Basisszenario und beschreibt die zuk\u00fcnftige Entwicklung der Agrarm\u00e4rkte unter definierten politischen und wirtschaftlichen Rahmenbedingungen . Zentrale Annahmen sind die Beibehaltung der derzeitigen Agrarpolitik und Umsetzung bereits beschlossener Politik - \u00e4nderungen sowie die Fortschreibung exogener Einflussfaktoren auf Basis historischer Trends . Die Berechnungen beruhen auf Daten und Informationen , die bis zum Fr\u00fchjahr 2024 vorlagen . Dargestellt werden Projektionsergebnisse f\u00fcr Agrarhandel , Preise , Nachfrage , Produktion , Einkommen und Umweltwirkungen . Die Darstellung der Ergebnisse konzentriert sich haupts\u00e4chlich auf die Entwicklungen des deutschen Agrarsektors bis zum Jahr 2034 i m Vergleich zum Durchschnitt der Basisperiode 2020 2022 . Die Ergebnisse zeigen , dass die EU ihre Position i m weltweiten Agrarhandel bis zum Jahr 2034 behaupten kann . Die Preise f\u00fcr Agrarprodukte sinken zu Beginn der Projektionsperiode vom hohen Niveau des Basisjahres , k\u00f6nnen sich bis zum Jahr 2034 jedoch wieder erholen . In Deutschland entwickelt sich der Anbau von Getreide r\u00fcckl\u00e4ufig , was auf ver\u00e4nderte Preiserelationen sowie einen R\u00fcckgang der landwirtschaftlich genutzten Fl\u00e4che zur\u00fcckzuf\u00fchren ist . I m Tiersektor setzt sich der in den letzten Jahren beobachtete Abbau der Tierbest\u00e4nde und R\u00fcckgang der Fleischerzeugung fort , insbesondere in der Schweinehaltung , wohingegen die Gefl\u00fcgelfleischerzeugung bis zum Jahr 2034 noch leicht w\u00e4chst . Eine positive Preisentwicklung am Milchmarkt in Verbindung mit einer weiteren Steigerung der Milchleistung f\u00fchren au\u00dferdem zu einem moderaten Anstieg der Milchanlieferungen . Das durchschnittliche reale Einkommen landwirtschaftlicher Betriebe geht \u00fcber die Projektionsperiode um 17 Prozent zur\u00fcck und liegt damit i m Jahr 2034 wieder auf dem mittleren Niveau der letzten zehn Jahre .
Output :
  ```json
{"Crops": [], "Soil": [], "Location": [{"country": {"value": "Deutschland", "span": [76, 87]}}, {"country": {"value": "Deutschland", "span": [1289, 1300]}}], "Time Statement": [{"startTime": {"value": "2024", "span": [28, 32]}}, {"endTime": {"value": "2034", "span": [35, 39]}}, {"startTime": {"value": "2024", "span": [171, 175]}}, {"endTime": {"value": "2034", "span": [178, 182]}}, {"endTime": {"value": "Frühjahr 2024", "span": [676, 689]}}, {"endTime": {"value": "2034", "span": [951, 955]}}, {"startTime": {"value": "2020", "span": [1004, 1008]}}, {"endTime": {"value": "2022", "span": [1009, 1013]}}, {"endTime": {"value": "2034", "span": [1106, 1110]}}, {"endTime": {"value": "2034", "span": [1257, 1261]}}, {"endTime": {"value": "2034", "span": [1690, 1694]}}, {"endTime": {"value": "2034", "span": [2038, 2042]}}, {"duration": {"value": "zehn Jahre", "span": [2087, 2097]}}]}

### Example 3 ###:
    Input Text:
 Title : \n Microsatellite marker data of Patellifolia patellaris , P. procumbens and P. webbiana \n\n Abstract : \n Microsatellite primers were developed to promote studies on the patterns of genetic diversity within Patellifolia patellaris and the relationship between the three species of the genus Patellifolia . The genomic sequence from Patellifolia procumbens was screened for SSRs and 3648 SSRs were identified . A subset of 53 SSR markers was validated of which 25 proved to be polymorphic in the three species except for the P. webbiana - specific marker JKIPat16 . A detailed description of the marker including GenBank accession numbers was published by Nachtigall et al . ( 2016 ) Applications in Plant Sciences 4(8):1600040 ( DOI : 10.3732 / apps.1600040 ) . The SSR markers were applied to study the genetic differentiation between P. patellaris as well as P. procumbens / P. webbiana occurrences sampled on the Iberian Peninsula , Madeira , the Canary Islands and the Cape Verde Islands ( Frese et al . , 2017 , Euphytica 213:187 ( DOI : 10.1007 / s10681 - 017 - 1942 - 0 ) , Frese et al . , 2018 accepted ) . \n  The marker set was used to study genetic diversity and genetic differentiation within the species . SSR data presented in excel file were generated by JKI in Quedlinburg , Germany . P. patellaris plants were sampled at 26 localities in 2015 , analysed using 24 SSR markers , the raw data were binned and statistically analysed in 2016 - 2017 . The first data sheet contains information on 581 plants in total . P. procumbens , P. webbiana were sampled in the same year at 7 locations , analysed using 22 SSR markers , the raw data were binned and statistically analysed in 2016 - 2017 . The second data sheet contains information on 172 plants in total . Both dataset include null alleles which are coded as \u201c 999 \u201d .
Output :
```json
{"Crops": [{"cropSpecies": {"value": "Patellifolia patellaris", "span": [40, 63]}}, {"cropSpecies": {"value": "P. procumbens", "span": [66, 79]}}, {"cropSpecies": {"value": "P. webbiana", "span": [84, 95]}}, {"cropSpecies": {"value": "Patellifolia patellaris", "span": [213, 236]}}, {"cropSpecies": {"value": "Patellifolia", "span": [297, 309]}}, {"cropSpecies": {"value": "Patellifolia procumbens", "span": [338, 361]}}, {"cropSpecies": {"value": "P. patellaris", "span": [842, 855]}}, {"cropSpecies": {"value": "P. procumbens", "span": [867, 880]}}, {"cropSpecies": {"value": "P. webbiana", "span": [883, 894]}}, {"cropSpecies": {"value": "P. patellaris", "span": [1306, 1319]}}, {"cropSpecies": {"value": "P. procumbens", "span": [1535, 1548]}}, {"cropSpecies": {"value": "P. webbiana", "span": [1551, 1562]}}], "Soil": [], "Location": [{"region": {"value": "Iberian Peninsula", "span": [922, 939]}}, {"region": {"value": "Madeira", "span": [942, 949]}}, {"region": {"value": "Canary Islands", "span": [956, 970]}}, {"region": {"value": "Cape Verde Islands", "span": [979, 997]}}, {"city": {"value": "Quedlinburg", "span": [1282, 1293]}}, {"country": {"value": "Germany", "span": [1296, 1303]}}], "Time Statement": [{"startTime": {"value": "2015", "span": [1360, 1364]}}, {"startTime": {"value": "2016", "span": [1454, 1458]}}, {"endTime": {"value": "2017", "span": [1461, 1465]}}, {"startTime": {"value": "2016", "span": [1697, 1701]}}, {"endTime": {"value": "2017", "span": [1704, 1708]}}]}

### Example 4 ###:
    Input Text:
 Title : \n Data from : N2 and N2O mitigation potential of replacing maize with the perennial biomass crop Silphium perfoliatum \u2013 An incubation study \n\n Abstract : \n Sustainability of biogas production is strongly dependent on soil - borne greenhouse gas ( GHG ) emissions during feedstock cultivation . Maize ( Zea mays ) is the most common feedstock for biogas production in Europe . Since it is an annual crop requiring high fertilizer input , maize cropping can cause high GHG emissions on sites that , due to their hydrology , have high N2O emission potential . On such sites , cultivation of cup plant ( Silphium perfoliatum ) as a perennial crop could be a more environmentally - friendly alternative offering versatile ecosystem services . To evaluate the possible benefits of perennial cup - plant cropping on GHG emissions and nitrogen losses , an incubation study was conducted with intact soil cores from a maize field and a cup plant field . The 15N gas flux method was used to quantify N source - specific N2 and N2O fluxes . Cumulated N2O emissions and N2+N2O emissions did not differ significantly between maize and cup plant soils , but tended to be higher in maize soil . Soils from both systems exhibited relatively high and similar N2O/(N2+N2O ) ratios ( N2Oi ) . N2O emissions originating from sources other than the 15N - labelled NO3 pool were low , but were the only fluxes exhibiting a significant difference between the maize and cup plant soils . Missing differences in fluxes derived from the 15N - pool indicate that under the experimental conditions with high moisture and NO3- level , and without plants , the cropping system had little effect on N fluxes related to denitrification . Lower soil pH and higher bulk density in the cup plant soil are likely to have reduced the mitigation potential of perennial biomass cropping .
Output :
```json
{"Crops": [{"cropSpecies": {"value": "maize", "span": [67, 72]}}, {"cropSpecies": {"value": "Silphium perfoliatum", "span": [105, 125]}}, {"cropSpecies": {"value": "Maize", "span": [302, 307]}}, {"cropSpecies": {"value": "Zea mays", "span": [310, 318]}}, {"cropSpecies": {"value": "maize", "span": [445, 450]}}, {"cropSpecies": {"value": "cup plant", "span": [596, 605]}}, {"cropSpecies": {"value": "Silphium perfoliatum", "span": [608, 628]}}, {"cropSpecies": {"value": "maize", "span": [917, 922]}}, {"cropSpecies": {"value": "cup plant", "span": [935, 944]}}, {"cropSpecies": {"value": "maize", "span": [1120, 1125]}}, {"cropSpecies": {"value": "cup plant", "span": [1130, 1139]}}, {"cropSpecies": {"value": "maize", "span": [1175, 1180]}}, {"cropSpecies": {"value": "maize", "span": [1444, 1449]}}, {"cropSpecies": {"value": "cup plant", "span": [1454, 1463]}}, {"cropSpecies": {"value": "cup plant", "span": [1759, 1768]}}], "Soil": [{"soilPH": {"value": "pH", "span": [1725, 1727]}}, {"soilBulkDensity": {"value": "bulk density", "span": [1739, 1751]}}], "Location": [], "Time Statement": []}

### Example 5 ###:
    Input Text:
 Title : \n Dataset : Temporal dynamics in the compositional relationships of cropland microbiomes \n\n Abstract : \n Die hier hinterlegten Daten beziehen sich auf die Publikation von Forschungsergebnissen i m Zusammenhang mit dem MonViA Projekt ( Monitoring von biologischer Vielfalt in Agrarlandschaften ) . Auf drei benachbarten Feldern eines landwirtschaftlichen Betriebs in der N\u00e4he von Hildesheim wurden in einem 14 - t\u00e4gigen Zeitraster die Abundanz und Vielfalt der bodenmikrobiologischen Lebensgemeinschaft \u00fcber einen Zeitraum von 2 Jahren erfasst . Auf den Feldern , die sich in Bodentextur ( Lehm und Ton ) und Bodenbearbeitung ( konservierend und konventionell ) unterschieden , wurden landwirtschaftliche Ma\u00dfnahmen nach herk\u00f6mmlicher Praxis betrieben . Um Einflussfaktoren zu untersuchen , wurden parallel meteorlogische Daten und physikochemische Bodeneigenschaften erfasst ( pH Wert , Gehalte an organischen Kohlenstoff und Gesamt - Stickstoff ) . Zur mikrobiologischen Charakterisierung wurde Boden - DNA mit PCR Verfahren untersucht . Die Abundanz wurde mit quantitativer PCR ( qPCR ) bestimmt , die Vielfalt mit der Sequenzierung von PCR Produkten mit Hilfe der IlluminaMiSeq Technologie .
 Output :
```json
{"Crops": [], "Soil": [{"soilTexture": {"value": "Lehm", "span": [597, 601]}}, {"soilTexture": {"value": "Ton", "span": [606, 609]}}, {"soilOrganicCarbon": {"value": "organischen Kohlenstoff", "span": [905, 928]}}], "Location": [{"city": {"value": "Hildesheim", "span": [387, 397]}}], "Time Statement": [{"duration": {"value": "14 - t\u00e4gigen", "span": [414, 426]}}, {"duration": {"value": "2 Jahren", "span": [534, 542]}}]}
 
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