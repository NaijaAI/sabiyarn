from transformers import AutoTokenizer
lang_id_token = 26290
lang_id_label_token = 102
classify_token = 42303
sentiment_token = 13896
topic_token = 49704
ner_token = 30827
qa_token = 23846
answer_token = 47682
tag_token = 7203
diacritize_token = 19588
correct_token = 48220
clean_token = 31666
summarize_token = 114
summary_token = 16150
title_token = 27246
headline_token = 49296
end_of_token = 14004
translate_token = 34635
# classify_token = 5
ner_token = 30827
eng = 1516
yor = 40754
ibo = 26762
hau = 13679
pcm = 28112
prompt = 37160
response = 2474
MASK = -100

# from omegaconf import OmegaConf

# config = OmegaConf.load("./config/mistral_config.yaml")

tokenizer = AutoTokenizer.from_pretrained("BeardedMonster/SabiYarn-125M")

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
# classify_token = 5
ner_token = tokenizer.encode("<NER>")[0]
eng = tokenizer.encode("<eng>")[0]
yor = tokenizer.encode("<yor>")[0]
ibo = tokenizer.encode("<ibo>")[0]
hau = tokenizer.encode("<hau>")[0]
pcm = tokenizer.encode("<pcm>")[0]
ff = tokenizer.encode("<ff>")[0]
fuv = tokenizer.encode("fuv")[0]
ful = tokenizer.encode("ful")[0]
urh = tokenizer.encode("<urh>")[0]
efik = tokenizer.encode("<efi>")[0]
prompt_token = tokenizer.encode("<prompt>")[0]
response_token = tokenizer.encode("<response>")[0]


action_tokens = [
    yor,
    ibo,
    eng,
    hau,
    pcm,
    tag_token,
    correct_token,
    lang_id_label_token,
    sentiment_token,
    topic_token,
    answer_token,
    summary_token,
    headline_token,
    response_token,
]
