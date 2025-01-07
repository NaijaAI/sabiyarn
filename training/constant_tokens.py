from transformers import AutoTokenizer
# from omegaconf import OmegaConf

# config = OmegaConf.load("./config/mistral_config.yaml")

tokenizer = AutoTokenizer.from_pretrained("BeardedMonster/sabiYarn")
MASK = -100
lang_id_token = tokenizer.encode("<lang_ID>")[0]
lang_id_label_token = tokenizer.encode("<lang_ID_label>")[0]
classify_token= tokenizer.encode("<classify>")[0]
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
lang_id_label_token2 = tokenizer.encode('<lang_id_label>')[0]
ner_token2 = tokenizer.encode('<ner>')[0]
#classify_token = 5
ner_token = tokenizer.encode("<NER>")[0]
eng = tokenizer.encode("<eng>")[0]
yor = tokenizer.encode("<yor>")[0]
ibo= tokenizer.encode("<ibo>")[0]
hau = tokenizer.encode("<hau>")[0]
pcm= tokenizer.encode("<pcm>")[0]
urh = tokenizer.encode("<urh>")[0]
efik = tokenizer.encode("<efi>")[0]
ful = tokenizer.encode("<ful>")[0]
fuv = tokenizer.encode("<fuv>")[0]
ff = tokenizer.encode("<ff>")[0]
prompt_token = tokenizer.encode("<prompt>")[0]
response_token = tokenizer.encode("<response>")[0]


action_tokens = [yor, ibo, eng, hau, pcm, ful, ff, fuv, urh, efik,tag_token, correct_token, lang_id_label_token, sentiment_token, topic_token, answer_token,
                 summary_token, headline_token, response_token
                 ]  


