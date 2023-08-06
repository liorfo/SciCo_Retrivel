import openai

MAX_TOKENS = 10
API_KEY = ''

sent = ('This is similar to a standard <m> machine learning problem of learning </m> from finite samples . '
        '</s></s>Adding <m> MLP in </m> does not seem to help , yielding slightly worse result than without MLP . </s>')
label = 0


def get_gpt_response(sentences):
    openai.api_key = API_KEY
    model_id = 'gpt-3.5-turbo'

    role1 = {'role': 'system', 'content': f'you will receive two texts. in each text there is a scientific term which '
                                          f'will be inside <m> </m>. the two texts will be separated by </s></s> '
                                          f'and the second text ends with </s>'}
    role2 = {'role': 'system',
             'content': f'Please define the hierarchy between Term A and Term B using the following levels: '
                        f'0 - No relation, no hierarchical connection'
                        f'1 - Same level, co-referring terms'
                        f'2 - Term A is a hypernym of term B, A is a more general form of term of B'
                        f'3 - Term A is a hyponym of Term B, A is a more specific form of term B'}
    role3 = {'role': 'system', 'content': f'Please output only the number of the correct hierarchy level'}
    # role2 = {'role': 'system',
    #          'content': f'with the two texts, you will output the following regarding the connection between the '
    #                     f'first term and the second term: 0 - no relation, 1 - co-reference, 2 - hypernym, 3 - hyponym'}
    sentences_query = {'role': 'user', 'content': f'{sentences}'}

    messages = [role1, role2, role3, sentences_query]

    response = openai.ChatCompletion.create(
        model=model_id,
        messages=messages,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content

# x = get_gpt_response(sent)
# print(x)
