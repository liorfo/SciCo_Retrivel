import jsonlines
import sys
from itertools import combinations
import pickle


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

    mention = ' '.join(tokens[doc_id][start:end + 1]).rstrip()
    mention_rep = tokens[doc_id][sent_start:start] + ['<m>'] + tokens[doc_id][start:end + 1] + ['</m>']
    mention_rep_with_sep = mention_rep + tokens[doc_id][end + 1:sent_end]

    return ' '.join(mention_rep_with_sep), mention


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


def get_sentences(first_to_second, no_relation, same_class, second_to_first, test_def):
    first_sentence, first_mention = get_sentence_context(gold_couple[0], gold[topic_index]['tokens'],
                                                         gold[topic_index]['sentences'])
    second_sentence, second_mention = get_sentence_context(gold_couple[1], gold[topic_index]['tokens'],
                                                           gold[topic_index]['sentences'])
    first_def = test_def[first_sentence + ' </s>']
    second_def = test_def[second_sentence + ' </s>']
    mentions[first_mention] = first_mention
    mentions[second_mention] = second_mention
    sentences_string = f'class: {gold_couple_class}, base model class: {system_couple_class}, ' \
                       f'definition model class: {new_model_couple_class}\n' \
                       f'first:\n{first_sentence}\nsecond:\n{second_sentence}\nfirst def:\n{first_def}\nsecond def:\n{second_def}\n\n\n'
    if gold_couple_class == 'same cluster':
        same_class += sentences_string
    elif gold_couple_class == 'first -> second':
        first_to_second += sentences_string
    elif gold_couple_class == 'second -> first':
        second_to_first += sentences_string
    elif gold_couple_class == 'no relation':
        no_relation += sentences_string
    return first_to_second, no_relation, same_class, second_to_first, (len(first_sentence) + len(second_sentence))


if __name__ == '__main__':
    gold_path = '/cs/labs/tomhope/forer11/SciCo_Retrivel/data/test.jsonl'
    no_def_model = '/cs/labs/tomhope/forer11/SciCo/res_no_def/system_0.6_0.4.jsonl'
    with_def_model = '/cs/labs/tomhope/forer11/SciCo/res_with_def/system_0.6_0.4.jsonl'
    hard = False

    same_class, first_to_second, second_to_first, no_relation = '', '', '', ''
    wrong_same_class, wrong_first_to_second, wrong_second_to_first, wrong_no_relation = '', '', '', ''
    mentions = {}
    num_of_tokens = 0
    num_samples = 0

    with jsonlines.open(gold_path, 'r') as f:
        gold = [line for line in f]

    with jsonlines.open(no_def_model, 'r') as f:
        system = [line for line in f]

    with jsonlines.open(with_def_model, 'r') as f:
        new_model = [line for line in f]
    if hard:
        gold = [topic for topic in gold if topic[hard] == True]
        system = [topic for topic in system if topic['id'] in [x['id'] for x in gold]]
        new_model = [topic for topic in new_model if topic['id'] in [x['id'] for x in gold]]

    with open(
            '/cs/labs/tomhope/forer11/SciCo_Retrivel/definition_handler/data/test_terms_definitions_final.pickle',
            'rb') as f:
        test_def = pickle.load(f)

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

            if gold_couple_class == new_model_couple_class and new_model_couple_class != system_couple_class:
                first_to_second, no_relation, same_class, second_to_first, _sent_len = get_sentences(first_to_second,
                                                                                                     no_relation,
                                                                                                     same_class,
                                                                                                     second_to_first,
                                                                                                     test_def)
            elif gold_couple_class != new_model_couple_class and gold_couple_class == system_couple_class:
                wrong_first_to_second, wrong_no_relation, wrong_same_class, wrong_second_to_first, sent_len = get_sentences(
                    wrong_first_to_second, wrong_no_relation,
                    wrong_same_class, wrong_second_to_first, test_def)
    print(num_samples)

    with open('/cs/labs/tomhope/forer11/SciCo_Retrivel/New_results/longformer_def_vs_no_def/same_class.txt', 'w') as f:
        f.write(same_class)
    with open('/cs/labs/tomhope/forer11/SciCo_Retrivel/New_results/longformer_def_vs_no_def/first_to_second.txt', 'w') as f:
        f.write(first_to_second)
    with open('/cs/labs/tomhope/forer11/SciCo_Retrivel/New_results/longformer_def_vs_no_def/second_to_first.txt', 'w') as f:
        f.write(second_to_first)
    with open('/cs/labs/tomhope/forer11/SciCo_Retrivel/New_results/longformer_def_vs_no_def/no_relation.txt', 'w') as f:
        f.write(no_relation)
    with open('/cs/labs/tomhope/forer11/SciCo_Retrivel/New_results/longformer_def_vs_no_def/wrong_same_class.txt', 'w') as f:
        f.write(wrong_same_class)
    with open('/cs/labs/tomhope/forer11/SciCo_Retrivel/New_results/longformer_def_vs_no_def/wrong_first_to_second.txt', 'w') as f:
        f.write(wrong_first_to_second)
    with open('/cs/labs/tomhope/forer11/SciCo_Retrivel/New_results/longformer_def_vs_no_def/wrong_second_to_first.txt', 'w') as f:
        f.write(wrong_second_to_first)
    with open('/cs/labs/tomhope/forer11/SciCo_Retrivel/New_results/longformer_def_vs_no_def/wrong_no_relation.txt', 'w') as f:
        f.write(wrong_no_relation)

    # mentions_file_string = '\n'.join([mention for mention in mentions])
    # with open(f'/cs/labs/tomhope/forer11/SciCo_Retrivel/mentions_{hard}.txt', 'w') as f:
    #     f.write(mentions_file_string)
