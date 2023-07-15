import jsonlines
import numpy as np
import openai
import random
import os

MAX_TOKENS = 100

API_KEY = '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'


def get_seperated_topics(path):
    with jsonlines.open(path, 'r') as f:
        data = [topic for topic in f]
        topics_clusters = []
        for i, topic in enumerate(data):
            # data[i]['mention_text'] = np.array([' '.join(topic['flatten_tokens'][start:end + 1])
            #                                     for start, end, _ in topic['flatten_mentions']])
            mentions = topic['mentions']
            tokens = topic['tokens']
            mention_clusters = {}

            for mention in mentions:
                paragraph_id = mention[0]
                start = mention[1]
                end = mention[2]
                cluster_id = mention[3]

                # Retrieve the mention text from the tokens
                mention_text = ' '.join(tokens[paragraph_id][start:end + 1])

                mention_details = {
                    'paragraph_id': paragraph_id,
                    'paragraph_text': ' '.join(tokens[paragraph_id]),
                    'start': start,
                    'end': end,
                    'text': mention_text
                }

                # Check if the cluster ID already exists in the mention_clusters dictionary
                if cluster_id in mention_clusters:
                    mention_clusters[cluster_id].append(mention_details)
                else:
                    mention_clusters[cluster_id] = [mention_details]
            topics_clusters.append(
                {'mention_clusters': mention_clusters, 'hard_10': topic['hard_10'], 'hard_20': topic['hard_20'],
                 'relations': topic['relations']})

        not_hard_topics = [topic for topic in topics_clusters if
                           not topic['hard_10'] and not topic['hard_20'] and len(topic['relations']) > 0]
        hard_20_topics = [topic for topic in topics_clusters if
                          topic['hard_20'] and not topic['hard_10'] and len(topic['relations']) > 0]
        hard_10_topics = [topic for topic in topics_clusters if topic['hard_10'] and len(topic['relations']) > 0]
        return not_hard_topics, hard_20_topics, hard_10_topics


def use_chat_gpt_api(term, context):
    openai.api_key = API_KEY
    model_id = 'gpt-3.5-turbo'
    messages = []
    role = {'role': 'system',
            'content': f'You receive terms and respond with definitions of max '
                       f'{MAX_TOKENS} length in the scientific and computer science domains'}
    context = {'role': 'system', 'content': f'this is the context of the term: {context}'}

    term = {'role': 'user', 'content': f'{term}'}

    messages = [role, context, term]

    response = openai.ChatCompletion.create(
        model=model_id,
        messages=messages,
        max_tokens=MAX_TOKENS,
    )
    print(f'term: {term}\n context: {context}\n\n')
    print(f'term definition: {response.choices[0].message.content}')
    print('----------------------------------------\n\n')


def get_random_topics(topics, num_of_topics):
    random_topics = random.sample(topics, num_of_topics)
    random_relations = [random.sample(topic['relations'], 1)[0] for topic in random_topics]
    return random_topics, random_relations


def test_and_print_definitions(not_hard_topics, hard_20_topics, hard_10_topics):
    random_hard_10_topics, random_hard_10_relations = get_random_topics(hard_10_topics, 5)
    random_hard_20_topics, random_hard_20_relations = get_random_topics(hard_20_topics, 5)
    random_not_hard_topics, random_not_hard_relations = get_random_topics(not_hard_topics, 5)

    print('random_not_hard_topics:\n\n')
    for topic, relation in zip(random_not_hard_topics, random_not_hard_relations):
        parent_cluster = topic['mention_clusters'][relation[0]]
        child_cluster = topic['mention_clusters'][relation[1]]
        parent_mention = parent_cluster[0]['text']
        parent_paragraph = parent_cluster[0]['paragraph_text']
        child_mention = child_cluster[0]['text']
        child_paragraph = child_cluster[0]['paragraph_text']
        use_chat_gpt_api(parent_mention, parent_paragraph)

    print('yay')


main_path = "/Users/liorfo/Desktop/masters_degree/Lab/SciCo_Retrivel/data/test.jsonl"
remote_path = "/cs/labs/tomhope/forer11/SciCo_Retrivel/data/train.jsonl"

# Call the function to print the JSONL files
not_hard_topics, hard_20_topics, hard_10_topics = get_seperated_topics(main_path)
test_and_print_definitions(not_hard_topics, hard_20_topics, hard_10_topics)

# hard_10_topics = random.sample([topic for topic in topics_clusters if topic['hard_10']], 5)


# use_chat_gpt_api('cross-view training')

hard_20_mention = ''
