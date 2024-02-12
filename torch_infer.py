from pytorch_modules import inference_utils

def forward(text, model_dir, tokenizer_dir):
    
    model = inference_utils.load_model(model_dir)
    tokenized_text = inference_utils.clean_and_tokenize_text(text, tokenizer_dir)
    
    pred = inference_utils.inference_step(tokenized_text, model)
    
    return pred

if __name__ == '__main__':
    text = "DIE LJAHS:DLKJASD*!(@&#!(P@:___)"
    pred = int(forward(text, model_dir='models/test_model.pt', tokenizer_dir='models/tokenizer/'))
    print(pred)