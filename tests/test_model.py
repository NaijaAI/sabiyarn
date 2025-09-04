import sys
import os
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Aletheia-ng/SabiYarn_test") # add the token argument please.

# Add the project root to path so we can import sabiyarn as a package
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

# Import the required dependencies for the tests
import torch
from sabiyarn.model import ModelArgs, SabiYarn, AttentionType
from sabiyarn.MLA import MLAConfig
# from sabiyarn.differential_attention import DiffAttnArgs

# Try to import cut_cross_entropy, skip test if not available
try:
    from cut_cross_entropy import linear_cross_entropy
    cce_available = True
except ImportError:
    cce_available = False
    linear_cross_entropy = None
    
    
def test_mla_model():
    print("🧪 Testing MLA Model...")
    try:
        mla_config = MLAConfig(
            hidden_size=256, num_heads=8, max_seq_len=32, max_batch_size=2,
            attention_dropout=0.0, q_lora_rank=64, qk_rope_head_dim=16,
            kv_lora_rank=32, v_head_dim=32, qk_nope_head_dim=16,
            attention_bias=False, original_seq_len=32, rope_theta=10000.0,
            rope_factor=1, beta_fast=32, beta_slow=1, mscale=1.
        )
        config = ModelArgs(
            dim=256, n_layers=2, n_heads=8, vocab_size=1000,
            max_batch_size=2, max_seq_len=32, attention_type=AttentionType.MLA,
            mla_config=mla_config
        )
        model = SabiYarn(config)
        model.eval()
        with torch.no_grad():
            model = model.float()
            tokens = torch.randint(0, 1000, (1, 16))
            hidden_states, logits = model(tokens, start_pos=0)
        
        assert hidden_states.shape == (1, 16, 256)
        assert logits.shape == (1, 16, 1000)
        print("✅ MLA model training test passed!")
        
        input_ids = torch.Tensor([[23,68,126,18,555,68,72,39,45,67,91,804]]).to(torch.int64)
        
        model.eval()
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=50,
                use_multi_token=False
            )
        print("Generated output: ", generated)
        print("✅ MLA model generation test passed!")
        return True
    except Exception as e:
        print(f"❌ MLA model test failed: {e}")
        
        return False
    

def load_sabiyarn_checkpoint(checkpoint_path='/mnt/c/Users/Jeffrey Paul/Downloads/ckpt_0000360.pt', device= "cpu", max_tokens = 50, ): #"cuda" if torch.cuda.is_available() else
    """
    Loads the SabiYarn model and optimizer state from a saved checkpoint.
    
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Rebuild ModelArgs (saved as checkpoint["model_args"])
    model_args = checkpoint["model_args"]
    
    # Reinitialize model
    model = SabiYarn(model_args).to(device)

    # Load weights
    model.load_state_dict(checkpoint["model"], strict=False)
    model = model.to(torch.float32)
    # Optimizer (optional — only if you want to resume training)
    optimizer = None
    if "optimizer" in checkpoint:
        optimizer = torch.optim.AdamW(model.parameters())  # or same optimizer you used
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Training state
    iter_num = checkpoint.get("iter_num", 0)
    best_val_loss = checkpoint.get("best_val_loss", None)
    config = checkpoint.get("config", None)

    print(f"✅ Loaded model. Iteration={iter_num}, Best Val Loss={best_val_loss}")
    
    input_ids = tokenizer.encode("omo, things get ", return_tensors="pt").to(device)  
    input_ids = input_ids[:, 1:].to(torch.int64)
    
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens= max_tokens,
            use_multi_token=model.use_multi_token
        )

    output_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    
    print("=" * 50)
    print(f"Input: The boy")
    print(f"Generated: {output_text}")
    print("=" * 50)
    return model, optimizer, iter_num, best_val_loss, config



dummy_data = {
    "english": [
        "The sun is shining today.",
        "<translate> The sun is shining today. <yor>",
        "<translate> I like to eat rice and beans. <ibo>",
        "<translate> I like to eat rice and beans. <swh>",
        "<translate> I like to eat rice and beans. <hau>",
        "<translate> I like to eat rice and beans. <amh>",
        "<translate> I like to eat rice and beans. <xho>",
        "<classify> The sun is shining today. <topic> ",
        "<classify> I like to eat rice and beans. <sentiment>",
        "<translate> The sun is shining today. <som>",
        "<translate> I like to eat rice and beans. <yor>",
        "<lang_ID> I like to eat rice and beans. <lang_ID_label>",
    ],
    "yoruba": [
        "<translate> Oòrùn ń ràn lónìí. <eng>",
        "<classify> Mo fẹ́ jẹ iresi ati ewa. <topic>",
        "<classify> Oòrùn ń ràn lónìí. <sentiment>",
        "<lang_ID> Mo fẹ́ jẹ iresi ati ewa. <lang_ID_label>",
        "Oòrùn ń ràn lónìí.",
        "<translate> Mo fẹ́ jẹ iresi ati ewa. <eng>",
    ],
    "igbo": [
        "Anyanwụ na-enwu taa.",
        "<translate> A na m ahụ ụtọ iri osikapa na agwa. <eng>",
        "<classify> Anyanwụ na-enwu taa. <topic>",
        "<classify> A na m ahụ ụtọ iri osikapa na agwa. <sentiment>",
        "<lang_ID> Anyanwụ na-enwu taa. <lang_ID_label>",
        "A na m ahụ ụtọ iri osikapa na agwa."
    ],
    "hausa": [
        "Rana tana haskakawa yau.",
        "<classify> Ina son cin shinkafa da wake. <sentiment>",
        "<classify> Rana tana haskakawa yau. <topic>",
        "<translate> Ina son cin shinkafa da wake. <eng>",
        "<lang_ID> Rana tana haskakawa yau. <lang_ID_label>",
        "Ina son cin shinkafa da wake.",
    ],
    "pidgin": [
        "Sun dey shine today.",
        "<translate> I like chop rice and beans. <eng>",
        "<classify> Sun dey shine today. <topic>",
        "<classify> I like chop rice and beans. <sentiment>",
        "<lang_ID> Sun dey shine today. <lang_ID_label>",
        "I like chop rice and beans.",
    ],
   
    "swahili": [
        "Jua linawaka leo.",
        "<translate> Napenda kula wali na maharagwe. <eng>",
        "<classify> Jua linawaka leo. <sentiment>",
        "<classify> Napenda kula wali na maharagwe. <topic>",
        "<lang_ID> Jua linawaka leo. <lang_ID_label>",
        "Napenda kula wali na maharagwe.",
    ],
    "xhosa": [
        "Ilanga liyakhanya namhlanje.",
        "<translate> Ndiyathanda ukutya irayisi kunye neembotyi. <eng>",
            "<classify> Ilanga liyakhanya namhlanje. <sentiment>",
        "<classify> Ndiyathanda ukutya irayisi kunye neembotyi. <topic>",
            "<lang_ID> Ilanga liyakhanya namhlanje. <lang_ID_label>",
        "Ndiyathanda ukutya irayisi kunye neembotyi.",
    ],
    "amharic": [
        "ፀሐይ ዛሬ ታይቷል።",
        "<translate> ሩዝ እና ባቄላ መብላት እወዳለሁ። <eng>",
         "<classify> ፀሐይ ዛሬ ታይቷል። <sentiment>",
        "<classify> ሩዝ እና ባቄላ መብላት እወዳለሁ። <topic>",
         "<lang_ID> ፀሐይ ዛሬ ታይቷል። <lang_ID_label>",
        "ሩዝ እና ባቄላ መብላት እወዳለሁ።",
    ],
    "somali": [
        "Qoraxdu maanta way iftiimaysaa.",
        "<translate> Waxaan jecelahay cunista bariis iyo digir. <eng>",
        "<classify> Qoraxdu maanta way iftiimaysaa. <sentiment>",
        "<classify> Waxaan jecelahay cunista bariis iyo digir. <topic>",
        "<lang_ID> Qoraxdu maanta way iftiimaysaa. <lang_ID_label>",
        "Waxaan jecelahay cunista bariis iyo digir.",
    ],
    "kinyarwanda": [
        "Izuba riraka uyu munsi.",
        "<translate> Nkunda kurya umuceri n'ibishyimbo. <eng>",
        "<classify> Izuba riraka uyu munsi. <sentiment>",
        "<classify> Nkunda kurya umuceri n'ibishyimbo. <topic>",
        "<lang_ID> Izuba riraka uyu munsi. <lang_ID_label>",
        "Nkunda kurya umuceri n'ibishyimbo.",
    ],
    "bemba": [
        "Acenje cala kuponona lelo.",
        "<translate> Ndeumfwa bwino ukulya umuceeri na beans. <eng>",
        "<classify> Acenje cala kuponona lelo. <sentiment>",
        "<classify> Ndeumfwa bwino ukulya umuceeri na beans. <topic>",
        "<lang_ID> Acenje cala kuponona lelo. <lang_ID_label>",
        "Ndeumfwa bwino ukulya umuceeri na beans.",
    ],
    "bambara": [
        "Tile be na taa bi.",
        "<classify> N bɛ nà rice ni nie bèɛ. <topic>",
        "<classify> Tile be na taa bi. <sentiment>",
        "<translate> N bɛ nà rice ni nie bèɛ. <eng>",
        "<lang_ID> Tile be na taa bi. <lang_ID_label>",
        "N bɛ nà rice ni nie bèɛ.",
    ],
    "sesotho": [
        "<translate> Letsatsi lea chaba kajeno. <eng>",
        "<translate> Ke rata ho ja raese le linaoa. <eng>",
        "<classify> Letsatsi lea chaba kajeno. <topic>",
        "<classify> Ke rata ho ja raese le linaoa. <sentiment>",
        "<lang_ID> Letsatsi lea chaba kajeno. <lang_ID_label>",
        "Ke rata ho ja raese le linaoa.",
    ],
    "northern_sotho": [
        "<translate> Letšatši le a phadima lehono. <eng>",
        "<translate> Ke rata go ja raese le dinawa. <eng>",
        "<classify> Letšatši le a phadima lehono. <topic>",
        "<classify> Ke rata go ja raese le dinawa. <sentiment>",
        "<lang_ID> Letšatši le a phadima lehono.<lang_ID_label>",
        "Ke rata go ja raese le dinawa."
    ],
    "twi": [
        "<translate> Owia rebɔ nnɛ. <eng>",
        "<translate> Mepɛ sɛ mede ɛmo ne abobɔ berɛ. <eng>",
        "<classify> Owia rebɔ nnɛ. <sentiment>",
        "<classify> Mepɛ sɛ mede ɛmo ne abobɔ berɛ. <topic>",
        "<lang_ID> Mepɛ sɛ mede ɛmo ne abobɔ berɛ. <lang_ID_label>",
        "Mepɛ sɛ mede ɛmo ne abobɔ berɛ."
    ],
    "dioula": [
        "<translate> Sole ye taa kɛ bɛɛ. <eng>",
        "<translate> N b’a fisa rice ni nie. <eng>",
         "<classify> Sole ye taa kɛ bɛɛ. <sentiment>",
        "<classify> N b’a fisa rice ni nie. <topic>",
         "Sole ye taa kɛ bɛɛ.",
         "<lang_ID> Sole ye taa kɛ bɛɛ. <lang_ID_label>",
       
    ],
    "shona": [
        "<translate> Zuva riri kubuda nhasi. <eng>",
        "<translate> Ndinoda kudya mupunga nembambaira. <eng>",
        "<classify> Zuva riri kubuda nhasi. <sentiment>",
        "<classify> Ndinoda kudya mupunga nembambaira. <topic>",
         "Ndinoda kudya mupunga nembambaira.",
         "<lang_ID> Ndinoda kudya mupunga nembambaira. <lang_ID_label>"
    ],
    "zulu": [
        "<translate> Ilanga likhanya namhlanje. <eng>",
        "<translate> Ngithanda ukudla irayisi nobhontshisi. <eng>",
        "<classify> Ilanga likhanya namhlanje. <sentiment>",
        "<classify> Ngithanda ukudla irayisi nobhontshisi. <topic>",
        "Ngithanda ukudla irayisi nobhontshisi.",
        "<lang_ID> Ngithanda ukudla irayisi nobhontshisi. <lang_ID_label>"
    ],
    "arithmetic": [
        "<prompt> 2 + 2 = <response>",
        "<promptp> 5 - 3 = <response>",
        "<prompt> 6 × 7 = <response>",
        "<prompt> 12 ÷ 4 = <response>"
    ]
}

checkpoint_path = "/mnt/c/Users/Jeffrey Paul/Downloads/out/out/20250823_215631_self_attention_256d_10L_8H_dense_hjovgi/ckpt.pt"

def load_test_sabiyarn_checkpoint(checkpoint_path=checkpoint_path, device= "cpu", max_tokens = 20, ): #"cuda" if torch.cuda.is_available() else
    """
    Loads the SabiYarn model and optimizer state from a saved checkpoint.
    
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Rebuild ModelArgs (saved as checkpoint["model_args"])
    model_args = checkpoint["model_args"]
    
    # Reinitialize model
    model = SabiYarn(model_args).to(device)

    # Load weights
    model.load_state_dict(checkpoint["model"], strict=False)
    model = model.to(torch.float32)
   
    # Training state
    iter_num = checkpoint.get("iter_num", 0)
    best_val_loss = checkpoint.get("best_val_loss", None)
    config = checkpoint.get("config", None)

    print(f"✅ Loaded model. Iteration={iter_num}, Best Val Loss={best_val_loss}")
    
    for lang, data_list in dummy_data.items():
        print(f"Testing {lang} language data ... ")
       
        for each  in data_list:
            task = "Monolingual Text Generation"
            if "<translate>" in each:
                task = "Translation"
            elif "<topic>" in each:
                task = "Topic classification"
            elif "<sentiment>" in each:
                task = "Sentiment Classification"
            elif "<lang_ID>" in each:
                task = "Language Identification"
                
            print(f"Task: {task}")
            print("="*50)
            input_ids = tokenizer.encode(each, return_tensors="pt").to(device)  
            input_ids = input_ids[:, 1:].to(torch.int64) if input_ids[0][0] == 128000 else input_ids
            input_len = len(input_ids[0])
            
            model.eval()
            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    max_new_tokens= max_tokens,
                    # use_multi_token=model.use_multi_token
                )

            output_text = tokenizer.decode(generated[0].tolist()[input_len:], skip_special_tokens=False)

            print(f"Input sentence: {each}")
            print(f"Generated: {output_text}")
            print("=" * 50)
    return model, iter_num, best_val_loss, config

if __name__ == "__main__":
    print("🚀 Running MLA Tests")
    print("=" * 40)
    print('transformers version: ', transformers.__version__)
    print('pytorch version: ', torch.__version__)
    # success1 = test_mla_model()
    # success2 = test_mla_distributed()
    
    # if success1: # and success2:
    #     print("\n✓ All MLA tests passed!")
    #     print("✓ MLA implementation working correctly!")
    # else:
    #     print("\n✗ Some MLA tests failed!")
        
        
    # load_test_sabiyarn_checkpoint()
    prompt_template = "[INST] <<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  \
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and \
positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer \
to a question, please don't share false information.<</SYS>> \
{}[/INST]"

    prompt = prompt_template.format(" who is elon musk?")
    tokens = tokenizer(prompt, return_tensors="pt").input_ids

    print("Length of processed tokens: ", len(tokens[0]))
    print("Decoded tokens: ", tokenizer.decode(tokens[0]))