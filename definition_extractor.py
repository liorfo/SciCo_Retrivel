import os
import jsonlines
import pickle
import textwrap
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline, AutoTokenizer, \
    AutoModelForCausalLM

main_path = "/cs/labs/tomhope/shared/unarXive_230324_open_subset/"
wizard_pretrained_model_name = "TheBloke/wizardLM-7B-HF"
falcon_pretrained_model_name = "tiiuae/falcon-40b"

mpnet_persist_directory = '/cs/labs/tomhope/forer11/unarxive_mpnet_embeddings/'
instructor_persist_directory = '/cs/labs/tomhope/forer11/unarxive_instructor_embeddings/'
e5_persist_directory = '/cs/labs/tomhope/forer11/unarxive_e5_large_embeddings/'

INSTRUCTOR, ALL_MPNET_BASE__V2, E5_LARGE = range(3)

normal_terms_contexts = {
    'unsupervised automatic facial point detection': 'Automatic perception of facial expressions with scaling '
                                                     'differences , pose variations and occlusions would greatly '
                                                     'enhance natural human robot interaction . This research '
                                                     'proposes unsupervised automatic facial point detection '
                                                     'integrated with regression-based intensity estimation for '
                                                     'facial action units ( AUs ) and emotion clustering to deal '
                                                     'with such challenges . The proposed facial point detector is '
                                                     'able to detect 54 facial points in images of faces with occlusions '
                                                     ', pose variations and scaling differences using Gabor filtering , '
                                                     'BRISK ( Binary Robust Invariant Scalable Keypoints ) , an Iterative '
                                                     'Closest Point ( ICP ) algorithm and fuzzy c-means ( FCM ) clustering .',

    'ASR': 'Any preserved information that is '
           'irrelevant to the ASR task could complicate further processing . '
           'Attributes in conventional ASR are derived by some transformation '
           'of the short‐term spectrum of speech that represents magnitude '
           'frequency components of a short segment of the signal . Origins '
           'of this representation can be traced to speech vocoding .',

    'Feature Pyramid Network': 'It can learn half and reusing '
                               'half of the preceding feature maps , and has fewer parameters . '
                               'Semantic aggregation adopts the Feature Pyramid Network ( FPN ) '
                               'into the original SSD . Adding more semantic information to each '
                               'scale to detect different size objects.',

    'spatial attention mechanisms': 'This paper proposes a target response adaptive correlation filter tracker '
                                    'with spatial attention to solve the above problems . Firstly , more useful '
                                    'feature information can be learned by making full use of the context '
                                    'information of the target area , and spatial attention mechanisms can '
                                    'be introduced to suppress the background information and reduce the '
                                    'unnecessary boundary effect . Secondly , the dynamic change of the '
                                    'target response is captured , and the most reliable response map is '
                                    'selected when the interference response map appears , to reduce the '
                                    'probability of dispositioning and train the more recognizable filter .',

    'JPEG method': 'In the receiver side , with the help of the side information , the compressed image can be '
                   'decoded and reconstructed . Simulation results show that the proposed scheme is able to '
                   'provide much higher compression performance compared with the JPEG method .',

    'video data generation': 'so with the current rate of video data generation , there is an urgent need of automatic'
                             ' video content analysis for subsequent purposes such as summarization , retrieval and '
                             'classification . And video shot boundary detection is usually the first step to segment '
                             'a video clip into meaningful shots .',
}

hard_20_terms_contexts = {
    'weight update': 'A typical learning algorithm is driven by error signals ε(n ) which are the differences between '
                     'the actual network output , y(n ) , and the desire ( or target ) output for a given input . For '
                     'a pattern learning , we can express the weight update in the following general form ∆w(n ) = '
                     'L(w(n ) , x(n ) , ε(n ) ) where L represents a learning algorithm . If we say that a neural '
                     'network can describe a model of data , then a multilayer perceptron describes the data in a '
                     'form of a hypersurface which approximates a functional relationship between x(n ) , and d(n ) .',

    'Understanding Behaviors of Neurons': '[ 17th neuron ] subsubsection : Understanding Behaviors of Neurons in C - '
                                          'LSTMs To get an intuitive understanding of how the C - LSTMs work on this '
                                          'problem , we examined the neuron activations in the last aggregation layer '
                                          'while evaluating the test set using TC - LSTMs .',

    'policy gradient methods': 'Reinforcement learning algorithms that use policy gradient methods approach an optimal'
                               ' policy faster than Q-learning but at the cost of incurring high variances in gradients'
                               ' . Among variance reduction techniques are actor-critic methods that use value and '
                               'advantage functions to train a policy actor .',

    'satellite navigation module': 'Each wing plane formation control device comprises a wing plane formation '
                                   'controller , a wing plane autopilot and a wing plane formation communication '
                                   'radio . Each wing plane formation controller comprises a satellite navigation '
                                   'module , a formation communication radio interface , an SBUS interface , a '
                                   'GNSS interface , a GCS ( Ground ControlStation ) interface and a set of '
                                   'wing plane formation controller software . The ground device comprises a '
                                   'ground control station and a ground differential base station .',

    'GRU': 'Neural machine translation ( NMT ) is a popular topic in Natural Language Processing which uses deep '
           'neural networks ( DNNs ) for translation from source to targeted languages . With the emerging technologies'
           ' , such as bidirectional Gated Recurrent Units ( GRU ) , attention mechanisms , and beam-search algorithms'
           ' , NMT can deliver improved translation quality compared to the conventional statistics-based methods , '
           'especially for translating long sentences . However , higher translation quality means more complicated '
           'models , higher computation/memory demands , and longer translation time , which causes difficulties for'
           ' practical use .',

    'ReLU': 'We establish optimal bounds in terms of network complexity and prove that rational neural networks '
            'approximate smooth functions more efficiently than ReLU networks . The flexibility and smoothness '
            'of rational activation functions make them an attractive alternative to ReLU , as we demonstrate '
            'with numerical experiments .',

    'L2': 'Secondly , controller and observer design techniques for LPV systems are examined . In particular , '
          'the stability conditions in the form of parameter dependent linear matrix inequalities ( LMI ) that '
          'result from the applications of the standard Lyapunov stability theory and other advanced techniques '
          'such as L2 and H∞ to LPV systems are discussed in detail . Also discussed are the some of the techniques '
          'for solving these LMIs .'

          'with numerical experiments .',
    'robust speech or speaker recognition': 'Accurate and effective voice activity detection ( VAD ) is a fundamental'
                                            ' step for robust speech or speaker recognition . In this study , '
                                            'we proposed a hierarchical framework approach for VAD and speech '
                                            'enhancement .',
}

hard_10_terms_contexts = {
    'end-to-end deep neural network': 'Instead we predict multiple camera pose hypotheses as well as the respective'
                                      ' uncertainty for each prediction . Towards this aim , we use Bingham '
                                      'distributions , to model the orientation of the camera pose , and a '
                                      'multivariate Gaussian to model the position , with an end-to-end deep '
                                      'neural network . By incorporating a Winner-Takes-All training scheme , '
                                      'we finally obtain a mixture model that is well suited for explaining '
                                      'ambiguities in the scene , yet does not suffer from mode collapse , '
                                      'a common problem with mixture density networks .',

    'Conv ( ) filter': 'The input resolution is normalized to 256 256 . Before the DU - Net , a Conv ( ) '
                       'filter with stride 2 and a max pooling would produce 128 features with resolution'
                       ' 64 64 . Hence , the maximum resolution of DU - Net is 64 64 .',

    'Time Series Analysis': 'What is Bootstrapping ? Estimation Confidence Sets and Hypothesis Testing '
                            'Regression Analysis Forecasting and Time Series Analysis Which Resampling '
                            'Method Should You Use ?',

    'LSTM': 'Websites of high results are referred to as authoritative sites under the search query , thus providing'
            ' a new perspective to measure website authoritativeness . By comparing the three model experiments with'
            ' Word2vec , CNN and LSTM , the experimental results on open datasets show that it is effective to use '
            'these three models , of which the LSTM model works best .',

    'MLP': 'ECG(Electrocardiogram ) , a field of Bio-signal , is generally experimented with classification algorithms'
           ' most of which are SVM(Support Vector Machine ) , MLP(Multilayer Perceptron ) . But this study modified '
           'the Random Forest Algorithm along the basis of signal characteristics and comparatively analyzed the '
           'accuracies of modified algorithm with those of SVM and MLP to prove the ability of modified algorithm . '
           'The R-R interval extracted from ECG is used in this study and the results of established researches '
           'which experimented co-equal data are also comparatively analyzed .',

    'categorical data analysis technique': """This paper introduces a categorical data analysis technique called "
                                           'Hildebrand's del . The advantages of del are ; it allows for the
                                            testing of customized prediction rules , it provides a strength of a ...""",
}


def get_query(term, context):
    return f'try to define the term: {term}. the term {term} was in the following context: {context}'


def save_arxive_lists(path):
    counter = 0
    for index, (root, dirs, files) in enumerate(os.walk(path)):
        all_data = []
        print(f'reading {root}...')
        for file in files:
            if file.endswith(".jsonl"):
                with jsonlines.open(f'{root}/{file}', 'r') as f:
                    data = [article for article in f]
                    all_data.extend(data)
                    counter += len(data)
                    # print(data)
        if index > 0:
            print('saving...')
            with open(f"/cs/labs/tomhope/forer11/arXiv_data_handler/open_set_list_{index}", "wb") as fp:
                pickle.dump(all_data, fp)
    print("Number of JSONL files: " + str(counter))


def process_arxive_to_docs():
    formatted_docs = []
    for root, dirs, files in os.walk("/cs/labs/tomhope/forer11/arXiv_data_handler/"):
        for file in files:
            print(f'reading {root + file}...')
            with open(root + file, "rb") as fp:
                docs = pickle.load(fp)
                for doc in docs:
                    page_content = doc['abstract']['text']
                    # Create an instance of Document with content and metadata
                    metadata = {key: value for key, value in doc['metadata'].items() if
                                isinstance(value, (str, int, float)) and key != 'abstract'}
                    formatted_docs.append(Document(page_content=page_content, metadata=metadata))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(formatted_docs)
    return texts


def get_instructor_embeddings():
    # the default instruction is: 'Represent the document for retrieval:'
    return HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                         model_kwargs={"device": "cuda"},
                                         cache_folder='/cs/labs/tomhope/forer11/cache/')


def get_mpnet_embeddings():
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                 cache_folder='/cs/labs/tomhope/forer11/cache/', model_kwargs={"device": "cuda"})


def get_e5_embeddings():
    return HuggingFaceEmbeddings(model_name='intfloat/e5-large',
                                 cache_folder='/cs/labs/tomhope/forer11/cache/', model_kwargs={"device": "cuda"})


def get_embeddings_model(embedding_type):
    if embedding_type == INSTRUCTOR:
        return get_instructor_embeddings()
    elif embedding_type == ALL_MPNET_BASE__V2:
        return get_mpnet_embeddings()
    elif embedding_type == E5_LARGE:
        return get_e5_embeddings()
    else:
        return None


def embed_and_store(texts=[], load=True, embedding_type=INSTRUCTOR, persist_directory=instructor_persist_directory):
    print('Created Embeddings')

    embedding = get_embeddings_model(embedding_type)

    if load:
        print(f'loading Vector embeddings from {persist_directory}...')
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    else:
        print(f'creating Vector embeddings to {persist_directory}...')
        vectordb = Chroma.from_documents(documents=texts,
                                         embedding=embedding,
                                         persist_directory=persist_directory)
    return vectordb


def get_wizard_model(pretrained_model_name):
    os.environ['TRANSFORMERS_CACHE'] = '/cs/labs/tomhope/forer11/cache/'
    os.environ['HF_MODULES_CACHE'] = '/cs/labs/tomhope/forer11/cache/'
    os.environ['HF_EMBEDDINGS_CACHE'] = '/cs/labs/tomhope/forer11/cache/'
    os.environ['HF_METRICS_CACHE'] = '/cs/labs/tomhope/forer11/cache/'
    os.environ['HF_DATASETS_CACHE'] = '/cs/labs/tomhope/forer11/cache/'
    os.environ['HF_DATASETS_DOWNLOADED_EVALUATED_PATH'] = '/cs/labs/tomhope/forer11/cache/'
    os.environ['HUGGINGFACE_HUB_CACHE'] = '/cs/labs/tomhope/forer11/cache/'

    tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_name)
    model = LlamaForCausalLM.from_pretrained(pretrained_model_name,
                                             load_in_8bit=False,
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             low_cpu_mem_usage=True,
                                             cache_dir='/cs/labs/tomhope/forer11/cache/')
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=1024,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def get_falcon_model(pretrained_model_name):
    os.environ['PYTORCH_TRANSFORMERS_CACHE'] = '/cs/labs/tomhope/forer11/cache/'
    os.environ['TRANSFORMERS_CACHE'] = '/cs/labs/tomhope/forer11/cache/'
    os.environ['HUGGINGFACE_HUB_CACHE'] = '/cs/labs/tomhope/forer11/cache/'
    os.environ['HF_HOME'] = '/cs/labs/tomhope/forer11/cache/'

    tokenizer = AutoTokenizer.from_pretrained('/cs/labs/tomhope/forer11/cache/falcon-40b', local_files_only=True)
    pipe = pipeline(
        "text-generation",
        model=pretrained_model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


def process_response(response, show_abstracts):
    print(wrap_text_preserve_newlines(response['result']))
    print('\n\nFrom:')
    for i, source in enumerate(response["source_documents"]):
        title = source.metadata['title']
        print(f'{i + 1}. {title}')
        if show_abstracts:
            print(f'Abstract: {source.page_content}\n')


def get_definition(qa_chain, term, context):
    query = get_query(term, context)
    response = qa_chain(query)
    return response


def process_terms(qa_chain, terms, terms_title, show_abstracts=False):
    for term in terms:
        context = terms[term]
        query = get_query(term, context)
        response = qa_chain(query)
        print(f'\n\nProcessing term: {term} from {terms_title}\n\n')
        process_response(response, show_abstracts)


def get_definition_retrieval_model(
        load_model=True,
        should_save_arxive_lists=False,
        embedding_type=INSTRUCTOR,
        persist_directory=instructor_persist_directory,
        k=3):
    if should_save_arxive_lists:
        save_arxive_lists(main_path)
    texts = []
    if not load_model:
        print('creating docs...')
        texts = process_arxive_to_docs()
        print('creating vectors...')

    vectordb = embed_and_store(texts, load_model, embedding_type, persist_directory)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    model = get_wizard_model(wizard_pretrained_model_name)

    qa_chain = RetrievalQA.from_chain_type(llm=model,
                                           chain_type="stuff",
                                           retriever=retriever,
                                           return_source_documents=True)
    return qa_chain


def run_example_retrieval():
    qa_chain = get_definition_retrieval_model()
    process_terms(qa_chain, normal_terms_contexts, 'normal terms', False)
    process_terms(qa_chain, hard_20_terms_contexts, 'hard 20 terms', False)
    process_terms(qa_chain, hard_10_terms_contexts, 'hard 10 terms', False)

# run_example_retrieval()
