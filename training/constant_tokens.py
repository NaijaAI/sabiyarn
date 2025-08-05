from transformers import AutoTokenizer
MASK=-100
tokenizer = AutoTokenizer.from_pretrained("BeardedMonster/SabiYarn-125M")
# tokenizer = AutoTokenizer.from_pretrained("Aletheia-ng/SabiYarn-125M")

lang_id_token = tokenizer.encode("<lang_ID>")[0]
lang_id_label_token = tokenizer.encode("<lang_ID_label>")[0]
classify_token = tokenizer.encode("<classify>")[0]
sentiment_token = tokenizer.encode("<sentiment>")[0]
topic_token = tokenizer.encode("<topic>")[0]
qa_token = tokenizer.encode("<qa>")[0]
answer_token = tokenizer.encode("<answer>")[0]
tag_token = tokenizer.encode("<tag>")[0]
diacritize_token = tokenizer.encode("<diacritize>")[0]
correct_token = tokenizer.encode("<correct>")[0]
clean_token = tokenizer.encode("<clean>")[0]
summarize_token = tokenizer.encode("<summarize>")[0]
summary_token = tokenizer.encode("<summary>")[0]
title_token = tokenizer.encode("<title>")[0]
headline_token = tokenizer.encode("<headline>")[0]
context_token = tokenizer.encode("<context>")[0]
end_of_text_token = tokenizer.encode("|end_of_text|")[0]
translate_token = tokenizer.encode("<translate>")[0]
lang_id_label_token2 = tokenizer.encode("<lang_id_label>")[0]
ner_token2 = tokenizer.encode("<ner>")[0]
ner_token = tokenizer.encode("<NER>")[0]
str_token = tokenizer.encode("<STR>")[0]  # semantic text relatedness
lang_id_token2 = tokenizer.encode("<identify>")[0]
lang_id_label_token2 = tokenizer.encode('<lang_id>')[0]
summarize_token2 = tokenizer.encode('<text>')[0]
prompt_token = tokenizer.encode('<prompt>')[0]
response_token = tokenizer.encode("<response>")[0]

######################################################################
# Comment this section during real training
eng = tokenizer.encode("<eng>")[0]
yor = tokenizer.encode("<yor>")[0]
ibo = tokenizer.encode("<ibo>")[0]
hau = tokenizer.encode("<hau>")[0]
pcm = tokenizer.encode("<pcm>")[0]
urh = tokenizer.encode("<urh>")[0]
efik = tokenizer.encode("<efi>")[0]
ff = tokenizer.encode("<ff>")[0]
ful = tokenizer.encode("<ful>")[0]
fuv = tokenizer.encode("<fuv>")[0]

#######################################################################

prompting_tokens = [
    lang_id_token,
    classify_token,
    qa_token,
    diacritize_token,
    clean_token,
    summarize_token,
    title_token,
    translate_token,
    ner_token2,
    ner_token,
    str_token,
    lang_id_token2, 
    summarize_token2, 
    prompt_token   
]

# Language special tokens
action_tokens = [ 
                 # other tags
                tokenizer.encode("<response>")[0],                
                tokenizer.encode("<toxic>")[0],
                tokenizer.encode("<intent>")[0],
                tokenizer.encode("<score>")[0],
                tokenizer.encode("<answer>")[0],           
                tag_token,
                correct_token,
                lang_id_label_token,
                sentiment_token,
                topic_token,
                answer_token,
                summary_token,
                headline_token,
                
                
                 # language iso codes
                tokenizer.encode("<eng>")[0],
                tokenizer.encode("<yor>")[0],
                tokenizer.encode("<ibo>")[0],
                tokenizer.encode("<hau>")[0],
                tokenizer.encode("<pcm>")[0],
                tokenizer.encode("<ff>")[0],
                tokenizer.encode("<fuv>")[0],
                tokenizer.encode("<ful>")[0],
                tokenizer.encode("<urh>")[0],
                tokenizer.encode("<efi>")[0],
                # tokenizer.encode("<kea>")[0],
                # tokenizer.encode("<lug>")[0], 
                # tokenizer.encode("<tsn>")[0], 
                # tokenizer.encode("<afr>")[0], 
                # tokenizer.encode("<din>")[0], 
                # tokenizer.encode("<xsm>")[0],
                # tokenizer.encode("<zu>")[0], 
                # tokenizer.encode("<tmh>")[0], 
                # tokenizer.encode("<ti>")[0], 
                # tokenizer.encode("<tzm>")[0],
                # tokenizer.encode("<ny>")[0], 
                # tokenizer.encode("<arb>")[0], 
                # tokenizer.encode("<dyu>")[0], 
                # tokenizer.encode("<eng>")[0], 
                # tokenizer.encode("<kea>")[0], 
                # tokenizer.encode("<fra>")[0], 
                # tokenizer.encode("<kab>")[0], 
                # tokenizer.encode("<amh>")[0],
                # tokenizer.encode("<pcm>")[0], 
                # tokenizer.encode("<hau>")[0], 
                # tokenizer.encode("<swh>")[0], 
                # tokenizer.encode("<snq>")[0], 
                # tokenizer.encode("<ful>")[0], 
                # tokenizer.encode("<ton>")[0],
                # tokenizer.encode("<vag>")[0], 
                # tokenizer.encode("<nup>")[0],
                # tokenizer.encode("<kmb>")[0],
                # tokenizer.encode("<mey>")[0],
                # tokenizer.encode("<luo>")[0], 
                # tokenizer.encode("<sn>")[0], 
                # tokenizer.encode("<nus>")[0],
                # tokenizer.encode("<ven>")[0], 
                # tokenizer.encode("<oke>")[0], 
                # tokenizer.encode("<yor>")[0], 
                # tokenizer.encode("<xh>")[0],
                # tokenizer.encode("<son>")[0],
                # tokenizer.encode("<igl>")[0], 
                # tokenizer.encode("<kik>")[0],
                # tokenizer.encode("<wolof>")[0], 
                # tokenizer.encode("<sag>")[0], 
                # tokenizer.encode("<aku>")[0], 
                # tokenizer.encode("<tso>")[0], 
                # tokenizer.encode("<ewe>")[0], 
                # tokenizer.encode("<ngl>")[0], 
                # tokenizer.encode("<run>")[0],
                # tokenizer.encode("<gah>")[0],
                # tokenizer.encode("<bm>")[0],
                # tokenizer.encode("<kbp>")[0],
                # tokenizer.encode("<umb>")[0],
                # tokenizer.encode("<aka>")[0], 
                # tokenizer.encode("<lin>")[0], 
                # tokenizer.encode("<urh>")[0],
                # tokenizer.encode("<tum>")[0],
                # tokenizer.encode("<nso>")[0],
                # tokenizer.encode("<ssw>")[0],
                # tokenizer.encode("<fat>")[0], 
                # tokenizer.encode("<som>")[0],
                # tokenizer.encode("<vai>")[0], 
                # tokenizer.encode("<tag>")[0], 
                # tokenizer.encode("<sot>")[0],
                # tokenizer.encode("<mos>")[0],
                # tokenizer.encode("<tiv>")[0],
                # tokenizer.encode("<kon>")[0], 
                # tokenizer.encode("<fon>")[0],
                # tokenizer.encode("<twi>")[0],
                # tokenizer.encode("<nde>")[0],
                # tokenizer.encode("<bem>")[0],
                # tokenizer.encode("<knc>")[0],
                # tokenizer.encode("<nya>")[0], 
                # tokenizer.encode("<orm>")[0], 
                # tokenizer.encode("<oro>")[0], 
                # tokenizer.encode("<mlg>")[0], 
                # tokenizer.encode("<shi>")[0], 
                # tokenizer.encode("<lus>")[0], 
                # tokenizer.encode("<gaa>")[0],
                # tokenizer.encode("<ibb>")[0], 
                # tokenizer.encode("<kin>")[0], 
                # tokenizer.encode("<mzw>")[0], 
                # tokenizer.encode("<kam>")[0], 
                
                
                            
                
]