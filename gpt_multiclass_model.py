from openai import OpenAI

MAX_TOKENS = 10
API_KEY = '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'

sent1 = ('After <m> linear transformation </m> on speech subspace , speech recognizer outperforms by 7.57 % '
        '( 62.14 % to 69.71 % ) under angry stress condition . </s></s> We conduct extensive experiments that '
        'demonstrate the proposed non-rigid alignment method is ( 1 ) effective , outperforming both the state-of-the-art '
        '<m> linear transformation-based methods </m> and node representation based methods , and ( 2 ) efficient , '
        'with a comparable computational time between the proposed multi-network representation learning component '
        'and its single-network counterpart . </s>')

sent2 = ('The <m> reproducing kernel particle method </m> is an efficient mesh free technique for the numerical solution '
        'of partial differential equations . </s></s> The <m> kernel-level implementation framework </m> '
        'of VMDFS is illustrated based on Linux 2.4.22 . </s>')
sent3 = ('But this study modified the Random Forest Algorithm along the basis of signal characteristics and comparatively '
         'analyzed the accuracies of modified algorithm with those of SVM and <m> MLP </m> to prove the ability of '
         'modified algorithm . </s></s> Adding <m> MLP in </m> does not seem to help , This study employs a '
         '<m> multilayer perceptions ( MLP ) neural network </m> with genetic algorithm ( GA ) to predict the New '
         'Taiwan dollar (NTD)/U.S. dollar ( USD ) exchange rate . </s>')
sent4 = ('However , it is still in infancy and has not been applied widely in <m> educational chatbot development </m> . '
         '</s></s> These features do not yet exist in the <m> chatbot application system </m> in other studies . </s>')
sent5 = ('The <m> reproducing kernel particle method </m> is an efficient mesh free technique for the numerical solution '
         'of partial differential equations . </s></s> The <m> kernel-level implementation framework </m> of VMDFS is '
         'illustrated based on Linux 2.4.22 . </s>')
labels = [3, 0]
client = OpenAI(
    # This is the default and can be omitted
    api_key=API_KEY,
)

def get_gpt_response(sentences):
    model_id = 'gpt-3.5-turbo'

    role1 = {'role': 'system', 'content': f'you will receive two texts. in each text there is a scientific term which '
                                          f'will be inside <m> </m>. </s> is indicating the end of a text.'}
    role2 = {'role': 'system',
             'content': f'define the hierarchy between Term A and Term B using the following levels: '
                        f'1 - Same level, co-referring terms'
                        f'2 - Term A is a parent concept of term B'
                        f'3 - Term A is a child concept of Term B'
                        f'0 - none of the above, no direct relation'}
    role3 = {'role': 'system', 'content': f'Please answer with only the correct hierarchy levels from the following classes: {0, 1, 2, 3} like in the examples that follows: \n'}
    role4 = {'role': 'system',
             'content': f'here are some texts, please provide the following output as below: \n'
                        f'\n***inputs***: #1.The <m> reproducing kernel particle method </m> is an efficient mesh free technique '
                        f'for the numerical solution of partial differential equations . </s></s> The <m> kernel-level '
                        f'implementation framework </m> of VMDFS is illustrated based on Linux 2.4.22 . </s>\n'
                        f'\n#2.But this study modified the Random Forest Algorithm along the basis of signal '
                        f'characteristics and comparatively analyzed the accuracies of modified algorithm with '
                        f'those of SVM and <m> MLP </m> to prove the ability of modified algorithm . </s></s>'
                        f'Adding <m> MLP in </m> does not seem to help , '
                        f'This study employs a <m> multilayer perceptions ( MLP ) neural network </m> with genetic '
                        f'algorithm ( GA ) to predict the New Taiwan dollar (NTD)/U.S. dollar ( USD ) exchange rate . </s>\n'
                        f'\n#3.But this study modified the Random Forest Algorithm along the basis of signal characteristics '
                        f'and comparatively analyzed the accuracies of modified algorithm with those of SVM and <m> MLP </m> '
                        f'to prove the ability of modified algorithm. </s></s> A second project has involved the installation '
                        f'of 3 Cyclops Coastal Imaging Stations in Durban and Cape Town which use an <m> RGB based MLP neural '
                        f'network </m> to extract waterlines on a short timescale . </s>\n'
                        f'\n#4.However , it is still in infancy and has not been applied widely in <m> educational '
                        f'chatbot development </m> . </s></s>These features do not yet exist in the <m> chatbot '
                        f'application system </m> in other studies . </s>\n'
                        f'\n#5.The <m> reproducing kernel particle method </m> is an efficient mesh free technique for '
                        f'the numerical solution of partial differential equations . </s></s> The <m> kernel-level '
                        f'implementation framework </m> of VMDFS is illustrated based on Linux 2.4.22 . </s>\n'
                        f'***outputs***: 0 1 2 3 0'
             }
    inputs = [f'#{i + 1}.{input}\n' for i, input in enumerate(sentences)]
    inputs_string = ''.join(inputs)
    sentences_query = {'role': 'user', 'content': f'\n***inputs***: {inputs_string} ***outputs***: '}

    messages = [role1, role2, role3, role4, sentences_query]

    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )
    return response.choices[0].message.content

# x = get_gpt_response([sent1, sent3, sent2, sent4, sent5])
# print(x)
