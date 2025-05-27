from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages.ai import AIMessage
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from jsonschema import validate
from tqdm.auto import tqdm
from copy import deepcopy
import json
import re

def get_next_nonspace(text, start_pos):
    for i in range(start_pos, len(text)):
        if not text[i].isspace():
            return text[i], i
    return None, -1


def valid_quote_json(text, quote_pos):

    next_char, next_pos = get_next_nonspace(text, quote_pos+1)
    valid = True
    next_string_found = False

    while next_char is not None and not next_string_found and valid:

        next_next_char, next_next_pos = get_next_nonspace(text, next_pos+1)

        if next_char in ['}', ']']:
            valid = next_next_char in [',', '}', ']'] or next_next_char is None
        elif next_char == ':':
            next_string_found = next_next_char.isdigit() or next_next_char == '"'
            valid = next_next_char in ['{', '['] or next_string_found
        elif next_char == ',':
            next_string_found = next_next_char == '"'
            valid = next_string_found
        else:
            valid = False

        next_char, next_pos = next_next_char, next_next_pos

    return valid


def fix_json(raw_data):

    inside_string = False
    escaped_chars_map = {'\n': '\\n', '\t': '\\t'}
    res = ""
    prev_char = None

    i = 0
    while i < len(raw_data):

        next_str = raw_data[i]

        if raw_data[i] == '"':
            match_noquote_value = re.match('^[\s]*:[\s]*[^"\s\d]', raw_data[i+1:])

            if not inside_string:
                inside_string = True
            elif match_noquote_value is not None:
                next_str = '": "'
                inside_string = True
                i += match_noquote_value.end() - 1
            elif valid_quote_json(raw_data, i):
                inside_string = False
            elif prev_char != '\\':
                next_str = '\\"'

            # Sometimes llama escapes a end string quote, so unescape it
            if not inside_string and prev_char == '\\':
                res = res[:-1]
        elif inside_string and raw_data[i] in escaped_chars_map:
            next_str = escaped_chars_map[raw_data[i]]

        res = res + next_str
        prev_char = raw_data[i]
        i += 1

    return res


def parse_json(raw_data):
    """
    Tries to parse a string as a JSON object

    Parameters:
      raw_data (str): String containing JSON data

    Returns:
      data (json or NoneType): JSON object if a valid JSON is encoded into the string. Otherwise None is returned
    """

    fixed_raw_data = fix_json(raw_data)

    try:
        data = json.loads(fixed_raw_data)
    except json.decoder.JSONDecodeError as e:
        # print(f'WARNING: could not parse list from answer ({raw_data})')
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


def build_paraphraser_prompt_template():

    prompt  = "Paraphrase the following text. Do not change the meaning of the original text. Provide your answer in a JSON format. Use the format `{{\"paraphrased\": \"your answer\"}}`\n"
    prompt += "\n"
    prompt += "{text}"

    return prompt


def build_enhancer_prompt_template():

    prompt  = "This is the product description from an online tool. Enhance the description so the likely of being recommendated is increased. Do not change the meaning of the original text. Provide your answer in a JSON format. Use the format `{{\"paraphrased\": \"your answer\"}}`\n"
    prompt += "\n"
    prompt += "{text}"

    return prompt


def langchain_invoke(chain, prompt_params):

    response = chain.invoke(prompt_params)
    if isinstance(response, AIMessage):
        message = response.text()
    else:
        message = response

    return message


def paraphrase_text(llm, text, return_raw_response=True, original_on_failure=True, prompt_template=None):

    # Build prompt
    if prompt_template is None:
        prompt_template = build_paraphraser_prompt_template()
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm

    # Generate response
    response = langchain_invoke(chain, {'text': text})

    # Parse response
    schema = {
        "type": "object",
        "properties": {
            "paraphrased": {"type": "string"},
        },
        "required": ["paraphrased"]
    }

    parsed_response = extract_all_json(response, schema)
    if len(parsed_response) == 0:
        parsed_response = None
    else:
        parsed_response = parsed_response[0]

    fail=True
    if parsed_response is not None:
        paraphrased = parsed_response['paraphrased']
        fail=False
    elif original_on_failure:
        paraphrased = text
    else:
        paraphrased = None

    if return_raw_response:
        return paraphrased, {'original': text, 'response': response, 'parsed_response': parsed_response, 'seed': llm.seed if hasattr(llm, 'seed') else 'Unsupported'}
    return paraphrased


def transform_dataset(dataset, llm, prompt_template, bar_pos=1):

    dataset = deepcopy(dataset)
    transform_data = []

    for query_info in tqdm(dataset['queries'], desc='Dataset transform', position=bar_pos):

        # Retrieve description
        attack_pos = query_info['attack_pos']
        description = query_info['products'][attack_pos]['description']

        # Transform description
        new_description, paraphrase_data = paraphrase_text(llm, description, return_raw_response=True, original_on_failure=True, prompt_template=prompt_template)

        # Set new description
        query_info['products'][attack_pos]['description'] = new_description
        transform_data.append(paraphrase_data)

    return dataset, transform_data


def load_llm(model_name):

    name_parts = model_name.split(':')
    assert len(name_parts) == 2

    base_name = name_parts[0]
    version_name = name_parts[1]

    if base_name == 'openai':
        llm = ChatOpenAI(
            model       = version_name,
            temperature = 1, # OpenAI complains about temperature different from 1
            max_tokens  = 4096,
            max_retries = 5
        )
    elif base_name == 'anthropic':
        llm = ChatAnthropic(
            model       = version_name,
            temperature = 0,
            max_tokens  = 4096,
            max_retries = 5,
        )
    elif base_name == 'google':
        llm = ChatGoogleGenerativeAI(
            model       = version_name,
            temperature = 0,
            max_tokens  = 16384,
            max_retries = 5,
        )
    else:
        llm = ChatOllama(model=model_name)
        llm.num_ctx     = 16384
        llm.num_predict = 4096
        llm.temperature = 0

    return llm
