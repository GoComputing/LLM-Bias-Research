from langchain_core.prompts import ChatPromptTemplate
from tools import extract_all_json

def base_prompt_template():

    prompt = 'You are a shop assistant that recommends the user the given shop articles based on his/her preferences.\n\n'
    prompt += 'Here are some articles in my shop:\n\n'
    prompt += '{icl}'
    prompt += '\nThe user query is "{query}". Based on the user query and the provided articles, recommend one of these articles in a chat way. These articles are not sorted by preference, so pick the one that better matches the user\'s preference'
    prompt += '\n\nPlease provide your answer using a JSON format with the fields "article_number" (int), "article_title" (str) and "recommendation" (str)'

    return prompt

def build_prompt_template(titles, descriptions, query, attack_pos=None, ollama_prompt=False):

    icl_prompt = ''
    for i, (title, description) in enumerate(zip(titles, descriptions)):
        if attack_pos is not None and attack_pos == i:
            icl_prompt += f'Title {i+1}: {{title}}\n'
            icl_prompt += f'Description {i+1}: {{description}}\n\n'
        else:
            icl_prompt += f'Title {i+1}: {title}\n'
            icl_prompt += f'Description {i+1}: {description}\n\n'

    if attack_pos is not None:
        query = '{query}'

    prompt = base_prompt_template().format(icl=icl_prompt, query=query)

    if ollama_prompt:
        prompt = ChatPromptTemplate.from_template(prompt)

    return prompt


class RecommendationSystem:

    def __init__(self, search_engine, llm, top_k=5):

        self.llm = llm
        self.search_engine = search_engine
        self.queries_cache = dict()
        self.top_k = top_k

    def get_matches(self, query):
        """
        Uses the search engine to find the closest articles

        Parameters:
          query (str): User query used to find closest articles

        Returns:
          matches (pd.DataFrame): DataFrame with the closest articles. A minimun of two columns are present (TITLE, DESCRIPTION)
        """

        if query in self.queries_cache:
            matches = self.queries_cache[query]
        else:
            matches = self.search_engine.query(query, self.top_k)
            self.queries_cache[query] = matches
        
        return matches


    def query(self, query):
        """
        Generates a recommendation in a user friendly way

        Parameters:
          query (str): User query as in a chat

        Returns:
          matches (pd.DataFrame), response (str), parsed_response (json or NoneType):
            `matches` is documented at `self.get_matches`
            `response` is the raw response from the LLM
            `parsed_response` is a JSON object with the fields `article_number`, `article_title`, `recommendation`. `article_number` is the position in the `matches` dataframe (starting from 1).
        """

        # Search articles on the shop
        matches = self.get_matches(query)
        titles = matches['TITLE'].tolist()
        descriptions = matches['DESCRIPTION'].tolist()

        # Use them as context
        prompt = build_prompt_template(titles, descriptions, query, ollama_prompt=True)
        chain = prompt | self.llm

        # Generate response
        response = chain.invoke({'query' : query})

        # Parse response
        schema = {
            "type": "object",
            "properties": {
                "article_number": {"type": ["string", "integer"]},
                "article_title": {"type": "string"},
                "recommendation": {"type": "string"}
            },
            "required": ["article_number", "article_title", "recommendation"]
        }

        parsed_response = extract_all_json(response, schema)
        if len(parsed_response) == 0 or len(parsed_response) > 1:
            parsed_response = None
        else:
            parsed_response = parsed_response[0]

        return matches, response, parsed_response


    def update_product_description(self, product_id, new_description):

        raise NotImplementedError('TODO')
