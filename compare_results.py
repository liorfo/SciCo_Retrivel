import jsonlines
import sys
from itertools import combinations


def generate_mention_couples(mentions):
    mention_couples = list(combinations(mentions, 2))
    return mention_couples


def get_sentence_context(mention, tokens, sentences):
    doc_id, start, end, _ = mention
    sent_start, sent_end = 0, len(tokens) - 1
    i = 0
    while i < len(sentences[doc_id]):
        sent_start, sent_end = sentences[doc_id][i]
        if start >= sent_start and end <= sent_end:
            break
        i += 1

    mention_rep = tokens[doc_id][sent_start:start] + ['<m>'] + tokens[doc_id][start:end + 1] + ['</m>']
    mention_rep_with_sep = mention_rep + tokens[doc_id][end + 1:sent_end]

    return ' '.join(mention_rep_with_sep)


def get_couple_class_tag(couple, relations):
    cluster_x, cluster_y = couple[0][-1], couple[1][-1]
    if cluster_x == cluster_y:
        return 'same cluster'
    elif [cluster_x, cluster_y] in relations:
        return 'first -> second'
    elif [cluster_y, cluster_x] in relations:
        return 'second -> first'
    else:
        return 'no relation'


if __name__ == '__main__':
    gold_path = sys.argv[1]
    sys_path = sys.argv[2]
    new_model_path = sys.argv[3]
    hard = sys.argv[4] if len(sys.argv) > 4 else None

    same_class, first_to_second, second_to_first, no_relation = '', '', '', ''

    with jsonlines.open(gold_path, 'r') as f:
        gold = [line for line in f]

    with jsonlines.open(sys_path, 'r') as f:
        system = [line for line in f]

    with jsonlines.open(new_model_path, 'r') as f:
        new_model = [line for line in f]
    if hard:
        gold = [topic for topic in gold if topic[hard] == True]
        system = [topic for topic in system if topic['id'] in [x['id'] for x in gold]]
        new_model = [topic for topic in new_model if topic['id'] in [x['id'] for x in gold]]

    for topic_index in range(len(gold)):
        all_gold_couples, all_system_couples, all_new_model_couples = generate_mention_couples(
            gold[topic_index]['mentions']), generate_mention_couples(
            system[topic_index]['mentions']), generate_mention_couples(
            new_model[topic_index]['mentions'])

        gold_relations, system_relations, new_model_relations = gold[topic_index]['relations'], system[topic_index][
            'relations'], \
            new_model[topic_index]['relations']

        for couple_index in range(len(all_gold_couples)):
            gold_couple, system_couple, new_model_couple = all_gold_couples[couple_index], all_system_couples[
                couple_index], all_new_model_couples[couple_index]

            gold_couple_class = get_couple_class_tag(gold_couple, gold_relations)
            system_couple_class = get_couple_class_tag(system_couple, system_relations)
            new_model_couple_class = get_couple_class_tag(new_model_couple, new_model_relations)

            if gold_couple_class == new_model_couple_class and gold_couple_class != system_couple_class:
                first_sentence = get_sentence_context(gold_couple[0], gold[topic_index]['tokens'],
                                                      gold[topic_index]['sentences'])
                second_sentence = get_sentence_context(gold_couple[1], gold[topic_index]['tokens'],
                                                       gold[topic_index]['sentences'])

                sentences_string = f'class: {gold_couple_class}\nfirst:\n{first_sentence}\nsecond:\n{second_sentence}\n\n\n'

                if gold_couple_class == 'same cluster':
                    same_class += sentences_string
                elif gold_couple_class == 'first -> second':
                    first_to_second += sentences_string
                elif gold_couple_class == 'second -> first':
                    second_to_first += sentences_string
                elif gold_couple_class == 'no relation':
                    no_relation += sentences_string

    with open('/cs/labs/tomhope/forer11/SciCo_Retrivel/same_class.txt', 'w') as f:
        f.write(same_class)
    with open('/cs/labs/tomhope/forer11/SciCo_Retrivel/first_to_second.txt', 'w') as f:
        f.write(first_to_second)
    with open('/cs/labs/tomhope/forer11/SciCo_Retrivel/second_to_first.txt', 'w') as f:
        f.write(second_to_first)
    with open('/cs/labs/tomhope/forer11/SciCo_Retrivel/no_relation.txt', 'w') as f:
        f.write(no_relation)

