from jsonschema import validate
import json

def parse_json(raw_data):
    """
    Tries to parse a string as a JSON object

    Parameters:
      raw_data (str): String containing JSON data

    Returns:
      data (json or NoneType): JSON object if a valid JSON is encoded into the string. Otherwise None is returned
    """

    try:
        # TODO: escape all quotes (") that are not followed by a comma (,), a right curly brace (}) or a colon (:)
        data = json.loads(raw_data)
    except json.decoder.JSONDecodeError as e:
        print(f'WARNING: could not parse list from answer ({raw_data})')
        data = None

    return data


def valid_schema(json_object, schema):
    """
    Validates a JSON object such that it follows a specific schema. It checks for several fields to be present, but extra fields are allowed

    Parameters:
      json_object (json): JSON to be validated
      schema (dict): Schema used to validate the JSON object. Follow the syntax from "https://python-jsonschema.readthedocs.io/en/latest/validate/"

    Returns:
      res (bool): Returns True if the JSON follows the schema. False otherwise
    """

    try:
        validate(instance=json_object, schema=schema)
        return True
    except:
        return False


def extract_all_json(raw_data, schema=None):
    """
    Extract all valid JSONs in a string. There might be extra characters between JSON objects. If a schema is provided,
    JSON objects are filtered so that they follow the provided schema

    Parameters:
      raw_data (str): String containing JSON objects
      schema (dict or NoneType): Schema used to filter extracted JSON objects. If not provided, the filter will not be applied

    Returns:
      res (list[json]): List of all valid JSON objects encoded into the string
    """

    res = []
    start_pos = 0
    reading_json = False
    reading_list = False
    depth = 0
    for i,c in enumerate(raw_data):
        if not reading_json and not reading_list and c == '{':
            reading_json = True
            depth = 0
            start_pos = i
        if not reading_json and not reading_list and c == '[':
            reading_list = True
            depth = 0
            start_pos = i
        if reading_json and c == '{':
            depth += 1
        if reading_list and c == '[':
            depth += 1
        if reading_json and c == '}' and depth > 0:
            depth -= 1
        if reading_list and c == ']' and depth > 0:
            depth -= 1
        if (reading_json and c == '}' and depth == 0) or (reading_list and c == ']' and depth == 0):
            reading_json = False
            reading_list = False
            json_data = parse_json(raw_data[start_pos:i+1])
            if json_data is not None:
                res.append(json_data)

    if schema is not None:
        res = list(filter(lambda json_object: valid_schema(json_object, schema), res))

    return res
