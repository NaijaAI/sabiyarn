from transformers import AutoTokenizer
from omegaconf import OmegaConf
import os

MASK=-100

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root =  os.path.dirname(current_dir)

# print(project_root)
config_path = os.path.join(project_root, "config", "config.yaml")
# print(config_path)
config = OmegaConf.load(config_path)

tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer.name)

lang_id_token = tokenizer.encode("<lang_ID>")[-1]
lang_id_label_token = tokenizer.encode("<lang_ID_label>")[-1]
classify_token = tokenizer.encode("<classify>")[-1]
sentiment_token = tokenizer.encode("<sentiment>")[-1]
topic_token = tokenizer.encode("<topic>")[-1]
qa_token = tokenizer.encode("<qa>")[-1]
answer_token = tokenizer.encode("<answer>")[-1]
tag_token = tokenizer.encode("<tag>")[-1]
diacritize_token = tokenizer.encode("<diacritize>")[-1]
correct_token = tokenizer.encode("<correct>")[-1]
clean_token = tokenizer.encode("<clean>")[-1]
summarize_token = tokenizer.encode("<summarize>")[-1]
summary_token = tokenizer.encode("<summary>")[-1]
title_token = tokenizer.encode("<title>")[-1]
headline_token = tokenizer.encode("<headline>")[-1]
context_token = tokenizer.encode("<context>")[-1]
end_of_text_token =  tokenizer.encode("<|end_of_text|>")[-1]
translate_token = tokenizer.encode("<translate>")[-1]
lang_id_label_token2 = tokenizer.encode("<lang_id_label>")[-1]
ner_token2 = tokenizer.encode("<ner>")[-1]
ner_token = tokenizer.encode("<NER>")[-1]
str_token = tokenizer.encode("<STR>")[-1]  # semantic text relatedness
lang_id_token2 = tokenizer.encode("<identify>")[-1]
lang_id_label_token2 = tokenizer.encode('<lang_id>')[-1]
summary_token2 = tokenizer.encode('<text>')[-1]
prompt_token = tokenizer.encode('<prompt>')[-1]
response_token = tokenizer.encode("<response>")[-1]


######################################################################
# Comment this section during real training. it is used only for testing training/utils.py
# eng = tokenizer.encode("<eng>")[-1]
# yor = tokenizer.encode("<yor>")[-1]
# ibo = tokenizer.encode("<ibo>")[-1]
# hau = tokenizer.encode("<hau>")[-1]
# pcm = tokenizer.encode("<pcm>")[-1]
# urh = tokenizer.encode("<urh>")[-1]
# efik = tokenizer.encode("<efi>")[-1]
# ff = tokenizer.encode("<ff>")[-1]
# ful = tokenizer.encode("<ful>")[-1]
# fuv = tokenizer.encode("<fuv>")[-1]

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
    summary_token2, 
    prompt_token   
]

# Language special tokens
action_tokens = [ 
                 # other tags
                tokenizer.encode("<response>")[-1],                
                tokenizer.encode("<toxic>")[-1],
                tokenizer.encode("<intent>")[-1],
                tokenizer.encode("<score>")[-1],
                tokenizer.encode("<answer>")[-1],           
                tag_token,
                correct_token,
                lang_id_label_token,
                lang_id_label_token2,
                sentiment_token,
                topic_token,
                answer_token,
                summarize_token,
                headline_token,
                
                #Comment the below code when testing training/utils.py
                 # language iso codes
                tokenizer.encode("<eng>")[-1],
                tokenizer.encode("<yor>")[-1],
                tokenizer.encode("<ibo>")[-1],
                tokenizer.encode("<hau>")[-1],
                tokenizer.encode("<pcm>")[-1],
                tokenizer.encode("<ff>")[-1],
                tokenizer.encode("<fuv>")[-1],
                tokenizer.encode("<ful>")[-1],
                tokenizer.encode("<urh>")[-1],
                tokenizer.encode("<efi>")[-1],
                tokenizer.encode("<kea>")[-1],
                tokenizer.encode("<lug>")[-1], 
                tokenizer.encode("<tsn>")[-1], 
                tokenizer.encode("<afr>")[-1], 
                tokenizer.encode("<din>")[-1], 
                tokenizer.encode("<xsm>")[-1],
                tokenizer.encode("<zu>")[-1], 
                tokenizer.encode("<tmh>")[-1], 
                tokenizer.encode("<ti>")[-1], 
                tokenizer.encode("<tzm>")[-1],
                tokenizer.encode("<ny>")[-1], 
                tokenizer.encode("<arb>")[-1], 
                tokenizer.encode("<dyu>")[-1], 
                tokenizer.encode("<eng>")[-1], 
                tokenizer.encode("<kea>")[-1], 
                tokenizer.encode("<fra>")[-1], 
                tokenizer.encode("<kab>")[-1], 
                tokenizer.encode("<amh>")[-1],
                tokenizer.encode("<swh>")[-1], 
                tokenizer.encode("<snq>")[-1], 
                tokenizer.encode("<ton>")[-1],
                tokenizer.encode("<vag>")[-1], 
                tokenizer.encode("<nup>")[-1],
                tokenizer.encode("<kmb>")[-1],
                tokenizer.encode("<mey>")[-1],
                tokenizer.encode("<luo>")[-1], 
                tokenizer.encode("<sn>")[-1], 
                tokenizer.encode("<nus>")[-1],
                tokenizer.encode("<ven>")[-1], 
                tokenizer.encode("<oke>")[-1], 
                tokenizer.encode("<xh>")[-1],
                tokenizer.encode("<son>")[-1],
                tokenizer.encode("<igl>")[-1], 
                tokenizer.encode("<kik>")[-1],
                tokenizer.encode("<wolof>")[-1], 
                tokenizer.encode("<sag>")[-1], 
                tokenizer.encode("<aku>")[-1], 
                tokenizer.encode("<tso>")[-1], 
                tokenizer.encode("<ewe>")[-1], 
                tokenizer.encode("<ngl>")[-1], 
                tokenizer.encode("<run>")[-1],
                tokenizer.encode("<gah>")[-1],
                tokenizer.encode("<bm>")[-1],
                tokenizer.encode("<kbp>")[-1],
                tokenizer.encode("<umb>")[-1],
                tokenizer.encode("<aka>")[-1], 
                tokenizer.encode("<lin>")[-1], 
                tokenizer.encode("<tum>")[-1],
                tokenizer.encode("<nso>")[-1],
                tokenizer.encode("<ssw>")[-1],
                tokenizer.encode("<fat>")[-1], 
                tokenizer.encode("<som>")[-1],
                tokenizer.encode("<vai>")[-1], 
                tokenizer.encode("<tag>")[-1], 
                tokenizer.encode("<sot>")[-1],
                tokenizer.encode("<mos>")[-1],
                tokenizer.encode("<tiv>")[-1],
                tokenizer.encode("<kon>")[-1], 
                tokenizer.encode("<fon>")[-1],
                tokenizer.encode("<twi>")[-1],
                tokenizer.encode("<nde>")[-1],
                tokenizer.encode("<bem>")[-1],
                tokenizer.encode("<knc>")[-1],
                tokenizer.encode("<nya>")[-1], 
                tokenizer.encode("<orm>")[-1], 
                tokenizer.encode("<oro>")[-1], 
                tokenizer.encode("<mlg>")[-1], 
                tokenizer.encode("<shi>")[-1], 
                tokenizer.encode("<lus>")[-1], 
                tokenizer.encode("<gaa>")[-1],
                tokenizer.encode("<ibb>")[-1], 
                tokenizer.encode("<kin>")[-1], 
                tokenizer.encode("<mzw>")[-1], 
                tokenizer.encode("<kam>")[-1], 
                
]