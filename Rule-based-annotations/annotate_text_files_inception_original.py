import spacy
import json
import os
import sys
from datetime import datetime
from spacy.matcher import Matcher
from spacy import displacy
from cassis import *
from cassis.typesystem import TYPE_NAME_FS_ARRAY, TYPE_NAME_ANNOTATION
from geotext import GeoText
from collections import Counter
import re

sys.path.append(r"/home/s27mhusa_hpc/Master-Thesis/FineTune19SeptBroad")

from gliner import GLiNER
from extra_rules import make_soil_patterns, make_soil_referencegroup
from power_monitor import power_monitor, estimate_carbon_footprint

POWER_MONITOR_INTERVAL = 0.5
model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")



def load_concept_list(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
def initialize_nlp_with_entity_ruler():
    model_name = "en_core_web_sm"
    
    species_file = r"/home/s27mhusa_hpc/Master-Thesis/ConceptsList/species_list_modified.json"
    soilTexture_file = r"/home/s27mhusa_hpc/Master-Thesis/ConceptsList/soilTexture_list.json"
    bulkDensity_file = r"/home/s27mhusa_hpc/Master-Thesis/ConceptsList/bulkDensity_list.json"
    organicCarbon_file = r"/home/s27mhusa_hpc/Master-Thesis/ConceptsList/organicCarbon_list.json"
    soilReferenceGroup_file = r"/home/s27mhusa_hpc/Master-Thesis/ConceptsList/soilReferenceGroup.json"
    germanCities_file = r"/home/s27mhusa_hpc/Master-Thesis/ConceptsList/de_cities_list.json"
    varieties_file = r"/home/s27mhusa_hpc/Master-Thesis/ConceptsList/varieties_list.json"
    nlp = spacy.load(model_name)
    # nlp.disable_pipes("ner")

    species_list = load_concept_list(species_file)
    soilTexture_list = load_concept_list(soilTexture_file)
    bulkDensity_list = load_concept_list(bulkDensity_file)
    organicCarbon_list = load_concept_list(organicCarbon_file)
    soilReferenceGroup_list = load_concept_list(soilReferenceGroup_file)
    germanCities_list = load_concept_list(germanCities_file)
    varieties_list = load_concept_list(varieties_file)

    matcher = Matcher(nlp.vocab)

    depth_before_number = [
    {"LEMMA": "depth"},  # Ensures "depth" is present
    {"IS_PUNCT": True, "OP": "*"},  # Optional punctuation
    {"POS": "ADP", "OP": "*"},  # Optional prepositions (e.g., "of", "in")
    {"POS": "ADV", "OP": "*"},  # Optional adverbs (e.g., "approximately")
    {"POS": "VERB", "OP": "*"},  # Optional verbs (e.g., "is", "reaches")
    {"POS": "DET", "OP": "*"},  # Optional determiners (e.g., "the")
    {"IS_PUNCT": True, "OP": "*"},  # Optional punctuation
    {"LIKE_NUM": True},  # Matches the number (e.g., 100, 50) - Optional now
    {"IS_PUNCT": True, "OP": "*"},  # Optional punctuation
    {
        "LOWER": {
            "IN": [
                "cm", "centimeter", "centimeters",
                "m", "meter", "meters",
                "in", "inch", "inches",
                "ft", "feet", "foot"
            ]
        },
        "OP": "*"  # Make unit optional
    }
]
    
    number_before_depth = [
    {"LIKE_NUM": True},  # Matches numbers (e.g., 100, 50) - Optional now
    {"IS_PUNCT": True, "OP": "*"},  # Optional punctuation
    {
        "LOWER": {
            "IN": [
                "cm", "centimeter", "centimeters",
                "m", "meter", "meters",
                "in", "inch", "inches",
                "ft", "feet", "foot"
            ]
        },
        "OP": "*"  # Make unit optional
    },
    {"POS": "ADP", "OP": "*"},  # e.g., "of", "in"
    {"POS": "VERB", "OP": "*"},  # e.g., "has", "is"
    {"POS": "DET", "OP": "*"},  # e.g., "the"
    {"IS_PUNCT": True, "OP": "*"},  # Optional punctuation
    {"LEMMA": "depth"}  # Ensures "depth" is present
]

    ph_before_number = [
        {"LEMMA": "ph"},  # Match "ph" as a separate entity
        {"IS_PUNCT": True, "OP": "*"},  # Optional punctuation before the number (e.g., "(")
        {"POS": "ADP", "OP": "*"},  # Optional prepositions (e.g., "of", "in")
        {"POS": "ADV", "OP": "*"},  # Optional adverbs (e.g., "approximately")
        {"POS": "VERB", "OP": "*"},  # Optional verbs (e.g., "is", "reaches")
        {"POS": "DET", "OP": "*"},  # Optional determiners (e.g., "the")
        {"IS_PUNCT": True, "OP": "*"},  # Optional punctuation (e.g., "-")
        {"LIKE_NUM": True, "TEXT": {"REGEX": r"^(?:[0-9]|1[0-4])(\.\d+)?$"}},  # Match single pH values like "7.0" or "7.4"
        {"IS_PUNCT": True, "OP": "*"},  # Optional punctuation (e.g., "-")
        {"LIKE_NUM": True, "TEXT": {"REGEX": r"^(?:[0-9]|1[0-4])(\.\d+)?$"}},  # Match the second pH value for ranges (e.g., "7.0" or "7.4")
        {"IS_PUNCT": True},  # Optional punctuation after the number (e.g., ")")
    ]

    ph_before_number_1 = [
    {"LEMMA": "ph"},  # Match "ph" as a separate entity
    {"POS": "ADP", "OP": "*"},  # Optional prepositions (e.g., "of", "in")
    {"POS": "ADV", "OP": "*"},  # Optional adverbs (e.g., "approximately")
    {"POS": "VERB", "OP": "*"},  # Optional verbs (e.g., "is", "reaches")
    {"POS": "DET", "OP": "*"},  # Optional determiners (e.g., "the")
    {"LIKE_NUM": True, "TEXT": {"REGEX": r"^\d+(\.\d+)?$"}},  # Match whole numbers and decimals (e.g., "6", "6.0", "7.4")
]
    ph_comma_list = [
    {"LEMMA": "ph"},
    {"IS_SPACE": True, "OP": "*"},
    {"LIKE_NUM": True, "TEXT": {"REGEX": r"^(?:[0-9]|1[0-4])(\.\d+)?$"}},
    {"IS_PUNCT": True, "TEXT": ",", "OP": "?"},
    {"LIKE_NUM": True, "TEXT": {"REGEX": r"^(?:[0-9]|1[0-4])(\.\d+)?$"}, "OP": "?"},
    {"LOWER": "and", "OP": "?"},
    {"LIKE_NUM": True, "TEXT": {"REGEX": r"^(?:[0-9]|1[0-4])(\.\d+)?$"}, "OP": "?"}
]
    ph_range = [
    {"LEMMA": "ph"},
    {"IS_SPACE": True, "OP": "*"},
    {"LIKE_NUM": True, "TEXT": {"REGEX": r"^(?:[0-9]|1[0-4])(\.\d+)?$"}},  # First number in the range
    {"LOWER": "to"},  # "to" indicating a range
    {"LIKE_NUM": True, "TEXT": {"REGEX": r"^(?:[0-9]|1[0-4])(\.\d+)?$"}},  # Second number in the range
]
    number_before_ph = [
        {"LIKE_NUM": True, "TEXT": {"REGEX": r"^(?:[0-9]|1[0-4])(\.\d+)?$"}},  # Match single pH values like "7.0" or "7.4"
        {"IS_PUNCT": True, "OP": "*"},  # Optional punctuation
        {"LIKE_NUM": True, "TEXT": {"REGEX": r"^(?:[0-9]|1[0-4])(\.\d+)?$"}, "OP": "*"},  # Match single pH values like "7.0" or "7.4"
        {"POS": "NOUN", "OP": "*"},  # Optional nouns (e.g., "cm", "meters")
        {"POS": "ADP", "OP": "*"},  # Optional prepositions (e.g., "is", "has")
        {"POS": "VERB", "OP": "*"},  # Optional verbs (e.g., "is", "has been")
        {"POS": "DET", "OP": "*"},  # Optional determiners (e.g., "the")
        {"IS_PUNCT": True, "OP": "*"},  # Optional punctuation
        {"LEMMA": "ph"}  # Match "pH" as a separate entity
    ]

    depth_variants = [
        [{"LOWER": "soil"}, {"LOWER": "depth"}],  # Matches "soil depth"
        [{"LOWER": "depth"}]  # Matches "depth"
    ]

    soil_available_nitrogen_pattern = [
    # Matches phrases like "plant nitrogen (N) availability"
    [
        {"LOWER": "plant", "OP": "?"},  # Optional "plant"
        {"LOWER": "nitrogen"},  # Mandatory "nitrogen"
        {"TEXT": "(", "OP": "?"},  # Optional opening bracket
        {"LOWER": "n", "OP": "?"},  # Optional chemical symbol (N)
        {"TEXT": ")", "OP": "?"},  # Optional closing bracket
        {"LOWER": "availability"}  # Mandatory "availability"
    ],
    # Matches phrases like "available nitrogen" or "nitrogen availability"
    [
        {"LOWER": "available", "OP": "?"},  # Optional "available"
        {"LOWER": "nitrogen"},  # Mandatory "nitrogen"
        {"LOWER": "availability", "OP": "?"}  # Optional "availability"
    ]
]
    latitude_pattern = [
    {"IS_DIGIT": True, "LENGTH": 2},  # Matches the degrees part (e.g., "51")
    {"TEXT": "°"},                    # Matches the degree symbol
    {"IS_DIGIT": True, "LENGTH": 2},  # Matches the minutes part (e.g., "82")
    {"TEXT": "'"},                    # Matches the minutes symbol
    {"LOWER": {"IN": ["n", "s"]}}     # Matches the direction (N or S, case-insensitive)
]

    longitude_pattern = [
    {"IS_DIGIT": True, "LENGTH": {"IN": [2, 3]}},  # Matches 2 or 3 digits (degrees part)    
    {"TEXT": "°"},                    # Matches the degree symbol
    {"IS_DIGIT": True, "LENGTH": 2},  # Matches the minutes part (e.g., "82")
    {"TEXT": "'"},                    # Matches the minutes symbol
    {"LOWER": {"IN": ["e", "w"]}, "OP": "?"}     # Matches the direction (N or S, case-insensitive)
]
    
    longitude_pattern_1 = [
    {"IS_DIGIT": True, "LENGTH": {"IN": [1, 2, 3]}},  # Degrees (1 to 3 digits)
    {"TEXT": "°"},                                     # Degree symbol
    {"TEXT": {"REGEX": r"^\d{2}’\d{2}(?:\.\d+)?$"}},  # Minutes, seconds, and optional fractional seconds
    {"TEXT": "’’"},                                   # Seconds symbol
    {"LOWER": {"IN": ["e", "w"]}}                     # Direction (E or W, case-insensitive)
]
    latitude_pattern_1 = [
    {"IS_DIGIT": True, "LENGTH": {"IN": [1, 2, 3]}},  # Degrees (1 to 3 digits)
    {"TEXT": "°"},                                     # Degree symbol
    {"TEXT": {"REGEX": r"^\d{2}’\d{2}(?:\.\d+)?$"}},  # Minutes, seconds, and optional fractional seconds
    {"TEXT": "’’"},                                   # Seconds symbol
    {"LOWER": {"IN": ["n", "s"]}}                     # Direction (E or W, case-insensitive)
]
    start_end_date = [
    {"SHAPE": "dddd"},  # 4-digit year (start year)
    {"TEXT": "-"},      # Hyphen
    {"SHAPE": "dddd"}   # 4-digit year (end year)
    ]   
    start_end_date_to = [
    {"SHAPE": "dddd"},  # 4-digit year (start year)
    {"TEXT": "to"},      # Hyphen
    {"SHAPE": "dddd"}   # 4-digit year (end year)
    ]
    # Define the list of trigger words for start date
    start_triggers = ["started", "since", "began", "initiated", "commenced", "launched", 
                      "from", "set up", "established", "started", "inaugurated", "opened", 
                      "created", "formed", "constituted", "triggered"]
    months_seasons = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",  # months
        "spring", "summer", "fall", "autumn", "winter"]
    start_date = [
    {"LOWER": {"IN": start_triggers}},  # trigger word
    {"LOWER": {"IN": ["in", "at", "on", "from"]}, "OP": "?"},  # optional preposition
    {"LOWER": {"IN": months_seasons}},  # Month/season (first part)
    {"SHAPE": "dddd"},  # 4-digit year (first part)
]

    matcher.add("soilDepth", [number_before_depth, depth_before_number])
    matcher.add("soilPH", [number_before_ph, ph_before_number_1, ph_before_number,ph_comma_list,ph_range])
    matcher.add("soilAvailableNitrogen", soil_available_nitrogen_pattern)
    matcher.add("latitude", [latitude_pattern, latitude_pattern_1])
    matcher.add("longitude", [longitude_pattern, longitude_pattern_1])
    matcher.add("startEndDate", [start_end_date,start_end_date_to])
    matcher.add("startDate", [start_date])



    ruler = nlp.add_pipe("entity_ruler", before="parser")
    patterns = (
        sorted([
            {"label": "cropSpecies", "pattern": species}
            for species in species_list
        ], key=lambda x: -len(x["pattern"])) +
            make_soil_patterns(soilTexture_list) +
        sorted([
            {"label": "soilBulkDensity", "pattern": bulkDensity}
            for bulkDensity in bulkDensity_list
        ], key=lambda x: -len(x["pattern"])) +
        sorted([
            {"label": "soilOrganicCarbon", "pattern": organicCarbon}
            for organicCarbon in organicCarbon_list
        ], key=lambda x: -len(x["pattern"])) +
        make_soil_referencegroup(soilReferenceGroup_list) +
        sorted([
            {"label": "city", "pattern": city}
            for city in germanCities_list
        ], key=lambda x: -len(x["pattern"]))+
        sorted([
            {"label": "cropVariety", "pattern": variety}
            for variety in varieties_list
        ], key=lambda x: -len(x["pattern"]))
    )
    ruler.add_patterns(patterns)

    return nlp, matcher

def annotate_text_inception(input_file_path, output_file_path, nlp, matcher):
    germanCities_file = r"/home/s27mhusa_hpc/Master-Thesis/ConceptsList/de_cities_list.json"
    germanCities_list = load_concept_list(germanCities_file)
    try:
        with open(input_file_path, "r", encoding="utf-8") as file:
            text = file.read()
        orig_doc = nlp(text)
        doc = nlp(text.lower())
        matches = matcher(doc)
        places = GeoText(text)

        SENTENCE_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
        TOKEN_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
        NER_TYPE_CROPS = "webanno.custom.Crops"
        NER_TYPE_SOIL = "webanno.custom.Soil"
        NER_TYPE_LOCATION = "webanno.custom.Location"
        NER_TYPE_TIME = "webanno.custom.Timestatement"

        with open(r"/home/s27mhusa_hpc/Master-Thesis/Evaluation_Files/full-typesystem.xml", "rb") as f:
            ts = load_typesystem(f)

        cas = Cas(typesystem=ts)
        cas.sofa_string = text

        Sentence = ts.get_type(SENTENCE_TYPE)
        Token = ts.get_type(TOKEN_TYPE)
        CropsEntity = ts.get_type(NER_TYPE_CROPS)
        SoilEntity = ts.get_type(NER_TYPE_SOIL)
        LocationEntity = ts.get_type(NER_TYPE_LOCATION)
        TimeEntity = ts.get_type(NER_TYPE_TIME)
        # Track annotated spans to prevent overlaps
        annotated_spans = []
        # Define units for depth and other measurements
        units = {"cm", "centimeter", "centimeters", "m", "meter", "meters",
         "in", "inch", "inches", "ft", "feet", "foot"}
        # Helper function to check for overlaps
        def is_overlap(new_span, existing_spans):
            for existing_span in existing_spans:
                if not (new_span[1] <= existing_span[0] or new_span[0] >= existing_span[1]):
                    return True  # Overlap detected
            return False  # No overlap

        # Add sentences and tokens to the CAS
        for sent in doc.sents:
            cas_sentence = Sentence(begin=sent.start_char, end=sent.end_char)
            cas.add(cas_sentence)

            for token in sent:
                cas_token = Token(begin=token.idx, end=token.idx + len(token.text))
                cas.add(cas_token)

        # Add entities to the CAS
        

        for ent in doc.ents:
            new_span = (ent.start_char, ent.end_char)
            if not is_overlap(new_span, annotated_spans):
                if ent.label_ == "cropSpecies":
                    cas_named_entity = CropsEntity(begin=ent.start_char, end=ent.end_char, crops="cropSpecies")
                elif ent.label_ == "cropVariety":
                    cas_named_entity = CropsEntity(begin=ent.start_char, end=ent.end_char, crops="cropVariety")
                elif ent.label_ in ["soilTexture", "soilBulkDensity", "soilOrganicCarbon", "soilReferenceGroup"]:
                    cas_named_entity = SoilEntity(begin=ent.start_char, end=ent.end_char, Soil=ent.label_)
                elif ent.label_ == "city":
                    if orig_doc[ent.start:ent.end].text.istitle():
                        cas_named_entity = LocationEntity(begin=ent.start_char, end=ent.end_char, Location="locationName")
                    else:
                        continue
                else:
                    continue
                cas.add(cas_named_entity)
                annotated_spans.append(new_span)  # Track the annotated span

        # Add soil depth matches to the CAS
        filtered_matches = {}
        for match_id, start, end in matches:
            if start not in filtered_matches or end > filtered_matches[start][1]:
                filtered_matches[start] = (match_id, start, end)
        for match_id, start, end in filtered_matches.values():
            span = doc[start:end]
            new_span = (span.start_char, span.end_char)
            if not is_overlap(new_span, annotated_spans):
                depth_added = False
                ph_added = False
                nitrogen_added = False
                latitude_added = False
                longitude_added = False

                for i, token in enumerate(span):
                    if token.lemma_ in ["depth", "ph", "availability"] and not (
                        depth_added if token.lemma_ == "depth" else ph_added if token.lemma_ == "ph" else nitrogen_added
                    ):
                        soil_type = (
                            "soilDepth" if token.lemma_ == "depth" else
                            "soilPH" if token.lemma_ == "ph" else
                            "soilAvailableNitrogen"
                        )
                        cas_named_entity = SoilEntity(
                            begin=token.idx,
                            end=token.idx + len(token.text),
                            Soil=soil_type
                        )
                        cas.add(cas_named_entity)
                        annotated_spans.append((token.idx, token.idx + len(token.text)))  # Track the annotated span

                        if token.lemma_ == "depth":
                            depth_added = True
                        elif token.lemma_ == "ph":
                            ph_added = True
                        else:
                            nitrogen_added = True

                    elif token.shape_ == "dddd":
                        seasons_months = ["january", "february", "march", "april", "may", "june", 
                                          "july", "august", "september", "october", "november", 
                                          "december", "spring", "summer", "fall", "winter"]
                        if nlp.vocab.strings[match_id] == "startDate":
                            try:
                                if span[i-1].text in seasons_months:
                                    #print()
                                    cas_named_entity = TimeEntity(
                                        begin = span[i-1].idx,
                                        end = span[i-1].idx + len(span[i-1].text)+ len(token.text)+1,
                                        Timestatement="startTime"
                                    )
                                    cas.add(cas_named_entity)
                                    annotated_spans.append((span[i-1].idx, span[i-1].idx + len(span[i-1].text)+ len(token.text)+1))
                            except:
                                cas_named_entity = TimeEntity(
                                    begin=token.idx,
                                    end=token.idx + len(token.text),
                                    Timestatement="startTime"
                                )
                                #print(f'data details: {token.text}, {token.idx}, {token.idx + len(token.text)}')
                                cas.add(cas_named_entity)
                                annotated_spans.append((token.idx, token.idx + len(token.text)))
                        elif nlp.vocab.strings[match_id] == "startEndDate":
                            start_year = span[0].text
                            end_year = span[2].text
                            if token.text == start_year:
                                cas_named_entity = TimeEntity(
                                    begin=token.idx,
                                    end=token.idx + len(token.text),
                                    Timestatement="startTime"
                                )
                            elif token.text == end_year:
                                cas_named_entity = TimeEntity(
                                begin=token.idx,
                                end=token.idx + len(token.text),
                                Timestatement="endTime"
                            )
                            else:
                                continue
                            cas.add(cas_named_entity)
                            annotated_spans.append((token.idx, token.idx + len(token.text)))
                    elif token.like_num:
                        if any(t.lemma_ == "depth" for t in span):
                            soil_type = "soilDepth"

                            # Default to using just the number
                            begin = token.idx
                            end = token.idx + len(token.text)

                            # Extend the span if the next token is a unit
                            if i + 1 < len(span):
                                next_token = span[i + 1]
                                if next_token.text.lower() in units:
                                    end = next_token.idx + len(next_token.text)

                            # Annotate number + optional unit as a single span
                            if not is_overlap((begin, end), annotated_spans):
                                cas_named_entity = SoilEntity(
                                    begin=begin,
                                    end=end,
                                    Soil=soil_type
                                )
                                cas.add(cas_named_entity)
                                annotated_spans.append((begin, end))
                        elif any(t.lemma_ in ["ph", "availability"] for t in span):
                            soil_type = (
                                "soilPH" if "ph" in [t.lemma_ for t in span] else
                                "soilAvailableNitrogen"
                            )
                            cas_named_entity = SoilEntity(
                                begin=token.idx,
                                end=token.idx + len(token.text),
                                Soil=soil_type
                            )
                            cas.add(cas_named_entity)
                            annotated_spans.append((token.idx, token.idx + len(token.text)))  # Track the annotated span

                # Latitude check
                full_text = "".join([token.text for token in span]).strip()
                if (re.match(r"^\d{1,2}°\d{1,2}' ?[NnSs]$", full_text) or re.match(r"^\d{1,3}°\d{2}’\d{2}(?:\.\d+)?’’[NnSs]$", full_text))  and not latitude_added:
                    cas_named_entity = LocationEntity(
                        begin=span.start_char,
                        end=span.end_char,
                        Location="latitude"
                    )
                    cas.add(cas_named_entity)
                    annotated_spans.append((span.start_char, span.end_char))  # Track the annotated span
                    latitude_added = True

                # Longitude check
                elif (re.match(r"^\d{1,3}°\d{1,2}' ?[EeWw]?$", full_text) or re.match(r"^\d{1,3}°\d{2}’\d{2}(?:\.\d+)?’’[EeWw]$", full_text)) and not longitude_added:
                    cas_named_entity = LocationEntity(
                        begin=span.start_char,
                        end=span.end_char,
                        Location="longitude"
                    )
                    cas.add(cas_named_entity)
                    annotated_spans.append((span.start_char, span.end_char))  # Track the annotated span
                    longitude_added = True

        # Add city and country annotations
        cities = places.cities
        countries = places.countries
        cities_counts = Counter(cities)
        countries_counts = Counter(countries)

        for city, count in cities_counts.items():
            if not city == "Boden":
                if count > 0:
                    start_city = 0
                    for i in range(count):
                        start_city = text.find(city, start_city + 1)
                        end_city = start_city + len(city)
                        new_span = (start_city, end_city)
                        if not is_overlap(new_span, annotated_spans):
                            cas_named_entity = LocationEntity(begin=start_city, end=end_city, Location="locationName")
                            cas.add(cas_named_entity)
                            annotated_spans.append(new_span)  # Track the annotated span

        for country, count in countries_counts.items():
            if count > 0:
                start_country = 0
                for i in range(count):
                    start_country = text.find(country, start_country + 1)
                    end_country = start_country + len(country)
                    new_span = (start_country, end_country)
                    if not is_overlap(new_span, annotated_spans):
                        cas_named_entity = LocationEntity(begin=start_country, end=end_country, Location="locationName")
                        cas.add(cas_named_entity)
                        annotated_spans.append(new_span)  # Track the annotated span
        #Add date_related entities
        labels = ["startDate","endDate", "date","duration", "geographicRegion"]
        location_dic = {"startDate": "startTime", "endDate": "endTime", "duration": "duration"}
        for large, sent in zip(orig_doc.sents, doc.sents):
            entities = model.predict_entities(sent.text, labels)
            for entity in entities:
                start = entity["start"]+sent.start_char
                end = entity["end"]+sent.start_char
                new_span = (start, end)
                if not is_overlap(new_span, annotated_spans):
                    if entity["label"] == "geographicRegion":
                        location_name = entity["text"].lower()
                        # Check if the string is part of any list element
                        matches = [item for item in germanCities_list if location_name.lower() == item.lower()]
                        if matches:
                            cas_named_entity = LocationEntity(begin=start, end=end, Location="locationName")
                        else:
                            normal_text = large.text[entity["start"]:entity["end"]]
                            #print(large.text.lower()==sent.text)
                            #print(sent)
                            if normal_text.istitle():
                                #print(normal_text)
                                cas_named_entity = LocationEntity(begin=start, end=end, Location="locationName")
                            else:
                                continue
                        cas.add(cas_named_entity)
                        annotated_spans.append(new_span)
                    else:
                        if entity["label"] == "startDate" or entity["label"] == "endDate":
                            if entity["score"] > 0.8:
                                cas_named_entity = TimeEntity(begin=start, end=end, Timestatement=location_dic[entity["label"]])
                                cas.add(cas_named_entity)
                                annotated_spans.append(new_span)
                            else:
                                cas_named_entity = TimeEntity(begin=start, end=end)
                                cas.add(cas_named_entity)
                                annotated_spans.append(new_span)
                        elif entity["label"] == "date":
                            cas_named_entity = TimeEntity(begin=start, end=end)
                            cas.add(cas_named_entity)
                            annotated_spans.append(new_span)
                        else:
                            cas_named_entity = TimeEntity(begin=start, end=end, Timestatement=location_dic[entity["label"]])
                            cas.add(cas_named_entity)
                            annotated_spans.append(new_span)
        
        matcher = Matcher(nlp.vocab)
        seasons_months = ["january", "february", "march", "april", "may", "june", 
                                          "july", "august", "september", "october", "november", 
                                          "december", "spring", "summer", "fall", "winter"]
        # Basic year (4-digit number)
        matcher.add("YEAR", [[{"SHAPE": "dddd"}]])

        # Month + Year (e.g. March 2021)
        matcher.add("MONTH_YEAR", [
            [{"LOWER": {"IN": [m.lower() for m in seasons_months]}}, {"SHAPE": "dddd"}]
        ])

        # Full date format (e.g. 10 April 2023)
        matcher.add("FULL_DATE", [[
            {"LIKE_NUM": True}, {"LOWER": {"IN": [m.lower() for m in seasons_months]}}, {"SHAPE": "dddd"}
        ]])
        doc = nlp(text.lower())
        matches = matcher(doc)

        filtered_matches = {}
        for match_id, start, end in matches:
            if start not in filtered_matches or end > filtered_matches[start][1]:
                filtered_matches[start] = (match_id, start, end)
        for match_id, start, end in matches:
            span = doc[start:end]
            new_span = (span.start_char, span.end_char)

            if not is_overlap(new_span, annotated_spans):
                label = nlp.vocab.strings[match_id]
                if label == "YEAR":
                    if len(span) > 4 or int(span.text) > 2060:
                        continue
                #print(f"Matched {label}: {span.text}")
                cas_named_entity = TimeEntity(
                    begin=span.start_char,
                    end=span.end_char
                )
                cas.add(cas_named_entity)
                annotated_spans.append(new_span)
        # Add region annotations
        # for ent in doc.ents:
        #     if ent.label_ == "GPE":
        #         if ent.text.capitalize() not in cities and ent.text.capitalize() not in countries and not ent.text.capitalize() == "Düngung":
        #             new_span = (ent.start_char, ent.end_char)
        #             if not is_overlap(new_span, annotated_spans):
        #                 cas_named_entity = LocationEntity(begin=ent.start_char, end=ent.end_char, Location="locationName")
        #                 cas.add(cas_named_entity)
        #                 annotated_spans.append(new_span)  # Track the annotated span

        if len(doc.ents) > 0:
            cas.to_xmi(output_file_path)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error processing {input_file_path}: {e}")


def save_power_monitoring_results(power_results, output_directory, run_metadata=None):
    if not power_results:
        print("⚠️  No power monitoring data collected.")
        return None

    os.makedirs(output_directory, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    carbon_footprint = estimate_carbon_footprint(power_results.get("total_energy_kwh", 0.0))

    summary_payload = {
        "timestamp": datetime.now().isoformat(),
        "output_directory": output_directory,
        "power_consumption": power_results,
        "carbon_footprint": carbon_footprint,
        "monitoring_interval_seconds": power_results.get("measurement_interval", POWER_MONITOR_INTERVAL),
    }

    if run_metadata:
        summary_payload["run_metadata"] = run_metadata

    summary_filename = f"power_monitoring_summary_{timestamp}.json"
    summary_path = os.path.join(output_directory, summary_filename)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    duration_seconds = power_results.get("duration_seconds") or 0.0
    duration_minutes = power_results.get("duration_minutes") or 0.0
    total_energy_wh = power_results.get("total_energy_wh") or 0.0
    average_power_watts = power_results.get("average_power_watts") or 0.0
    peak_power_watts = power_results.get("peak_power_watts") or 0.0
    average_cpu = power_results.get("average_cpu_usage_percent") or 0.0
    peak_cpu = power_results.get("peak_cpu_usage_percent") or 0.0
    average_memory = power_results.get("average_memory_usage_percent") or 0.0
    peak_memory = power_results.get("peak_memory_usage_percent") or 0.0
    measurement_count = int(power_results.get("measurement_count") or 0)

    print("\n" + "=" * 80)
    print("POWER MONITORING SUMMARY")
    print("=" * 80)
    print(f"Duration: {duration_seconds:.2f} seconds ({duration_minutes:.2f} minutes)")
    print(f"Total Energy: {total_energy_wh:.4f} Wh")
    print(f"Average Power: {average_power_watts:.2f} W")
    print(f"Peak Power: {peak_power_watts:.2f} W")
    print(f"Average CPU Usage: {average_cpu:.1f}% (peak {peak_cpu:.1f}%)")
    print(f"Average Memory Usage: {average_memory:.1f}% (peak {peak_memory:.1f}%)")
    print(f"Samples collected: {measurement_count}")
    print(f"CO2 Emissions: {carbon_footprint['co2_g']:.2f} g")

    if run_metadata and isinstance(run_metadata, dict):
        processed_files = run_metadata.get("processed_files")
        annotated_files = run_metadata.get("annotated_files")
        if processed_files is not None or annotated_files is not None:
            print("-" * 80)
            if processed_files is not None:
                print(f"Files processed: {processed_files}")
            if annotated_files is not None:
                print(f"Files with annotations: {annotated_files}")

    print("-" * 80)
    print("Power monitoring summary saved to:")
    print(summary_path)
    print("=" * 80)

    return summary_path


def process_directory_inception(input_directory, output_directory, nlp, matcher):
    input_directory = os.path.normpath(input_directory)
    output_directory = os.path.normpath(output_directory)

    os.makedirs(output_directory, exist_ok=True)

    processed_files = 0
    annotated_files = 0
    for filename in os.listdir(input_directory):
        if filename.endswith(".txt"):
            input_file_path = os.path.join(input_directory, filename)
            output_file_path = os.path.join(output_directory, filename.replace(".txt", "_inception.xmi"))
            entity_present = annotate_text_inception(input_file_path, output_file_path, nlp, matcher)
            processed_files += 1
            if entity_present:
                annotated_files += 1
            # if(entity_present):
            #     i = i+1
            # if i == 15:
            #     break

    return {
        "processed_files": processed_files,
        "annotated_files": annotated_files
    }

if __name__ == "__main__":
    nlp, matcher = initialize_nlp_with_entity_ruler()
    input_directory = r"/home/s27mhusa_hpc/Master-Thesis/Text_Files_Test_Data"
    output_directory = r"/home/s27mhusa_hpc/Master-Thesis/Test_Rule_Based_Annotations_1stNovember"

    print(f"Processing text files in Inception format from: {input_directory}")

    run_metadata = {
        "input_directory": input_directory,
        "output_directory": output_directory
    }

    with power_monitor(monitoring_interval=POWER_MONITOR_INTERVAL) as monitor:
        process_stats = process_directory_inception(input_directory, output_directory, nlp, matcher)

    if isinstance(process_stats, dict):
        run_metadata.update(process_stats)

    power_results = monitor.get_results()
    summary_file = save_power_monitoring_results(power_results, output_directory, run_metadata)

    if summary_file:
        print(f"Power monitoring data stored at: {summary_file}")

    print("✅ Inception annotation process completed.")