import json
import random
import argparse
import os

def main(args):
    output_path = args.output_path
    seed = args.seed
    n_queries = args.size

    if os.path.exists(output_path):
        raise IOError('Output file already exists')

    # Sample categories for generating user queries
    categories = [
        "electronics", "fashion", "home appliances", "beauty", "sports", "automotive",
        "books", "toys", "groceries", "office supplies", "pet supplies", "fitness",
        "gaming", "travel", "outdoor equipment", "health", "baby products", "jewelry"
    ]

    # Sample templates for user queries
    query_templates = [
        "Do you have {}?",
        "Show me the latest {}.",
        "Find me a budget-friendly {}.",
        "I need a high-quality {}.",
        "What are the best {} available?",
        "Do you sell {} with good reviews?",
        "Find me {} under $50.",
        "Show me the top-rated {}.",
        "I’m looking for a durable {}.",
        "Do you have discounts on {}?",
        "Find me an eco-friendly {}.",
        "I need a compact {} for travel.",
        "What are the trending {} this season?",
        "Show me bestselling {} this month.",
        "Do you have any deals on {}?",
        "Can you recommend a reliable {}?",
        "What’s the difference between {} and {}?",
        "Find me a customizable {}.",
        "Do you sell high-end {}?",
        "Show me must-have {} accessories."
    ]

    # Sample products within each category
    products = {
        "electronics": ["wireless headphones", "smartphones", "smartwatches", "laptops", "tablets", "gaming consoles", "cameras"],
        "fashion": ["running shoes", "summer dresses", "winter jackets", "men’s suits", "designer handbags", "sportswear"],
        "home appliances": ["microwaves", "vacuum cleaners", "air purifiers", "coffee makers", "blenders", "toasters"],
        "beauty": ["skincare products", "perfumes", "makeup kits", "hairdryers", "organic shampoos"],
        "sports": ["yoga mats", "fitness trackers", "treadmills", "dumbbells", "cycling gloves"],
        "automotive": ["car chargers", "seat covers", "dashboard cameras", "car cleaning kits"],
        "books": ["fiction books", "non-fiction bestsellers", "mystery novels", "science fiction books"],
        "toys": ["LEGO sets", "puzzle games", "educational toys", "remote-controlled cars"],
        "groceries": ["organic green tea", "protein snacks", "coffee beans", "gluten-free pasta"],
        "office supplies": ["ergonomic chairs", "standing desks", "wireless keyboards", "noise-canceling headphones"],
        "pet supplies": ["dog food", "cat trees", "fish tanks", "rabbit cages"],
        "fitness": ["resistance bands", "exercise bikes", "adjustable dumbbells"],
        "gaming": ["gaming mice", "RGB keyboards", "VR headsets", "gaming chairs"],
        "travel": ["carry-on suitcases", "travel pillows", "backpacks", "power banks"],
        "outdoor equipment": ["camping tents", "hiking boots", "fishing gear"],
        "health": ["blood pressure monitors", "multivitamins", "first aid kits"],
        "baby products": ["baby strollers", "diaper bags", "baby monitors"],
        "jewelry": ["gold necklaces", "silver rings", "diamond earrings"]
    }

    random.seed(seed)

    # Generate 1000 queries
    queries = []
    for _ in range(n_queries):
        category = random.choice(list(products.keys()))
        product = random.choice(products[category])
        template = random.choice(query_templates)

        # Some templates require two products for comparison
        if "{} and {}" in template:
            second_product = random.choice(products[random.choice(list(products.keys()))])
            query = template.format(product, second_product)
        else:
            query = template.format(product)

        queries.append(query)

    # Save to a JSON file
    with open(output_path, "w") as f:
        json.dump(queries, f, indent=2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to generate a dataset of user queries in different categories')
    parser.add_argument('-o', '--output-path', type=str, required=True, help='Path to store the queries, in JSON format')
    parser.add_argument('-s', '--seed', type=int, default=1234, help='Seed used to sample categories and patterns')
    parser.add_argument('-n', '--size', type=int, default=1000, help='Number of queries to generate')

    args = parser.parse_args()

    main(args)
    
