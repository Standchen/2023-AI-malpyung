# 2023 AI malpyung (Emotional Analysis)
This repository contains several approaches tried to solve emotional analysis (감정 분석) challenge from *2023 인공지능 언어 능력 평가 대회*.

The challenge is to analyze what emotions the speaker has about a target in the given sentence. These emotions are consisted of total 8 distinct categories: joy(기쁨), anticipation(기대), trust(신뢰), surprise(놀람), disgust(혐오), fear(공포), anger(분노) and sadness(슬픔).

Three approaches are made to solve the task, incorporating BERT and GPT models.
All the resulting models are designed to fit in RTX 4090 and assessed using the F1-score(micro), as the challenge requires.

# Approaches
## BERT Models
**bert_plain.py**: Fine-tune a BERT model for multi-label sequence classification by attaching a fully-connected layer to the output representation of CLS token.
**bert_GRU.py**: Fine-tune a BERT model for multi-label sequence classification by attaching a bidrectional GRU layer on top of the model. The representations from GRU is then concatenated to CLS token representation, and passed to multiple fully-connected layers for classification.

## GPT Model
**gpt_train.py**: Fine-tune a GPT model to answer what emotion is expressed in the given sentence. To accommodate the model within a 24GB VRAM, the base model is quantized to 4-bit representation and fine-tuned employing QLoRA.
**gpt_inference.py**: Perform inference using the fine-tuned model from `gpt_train.py`. Presence of each emotion is determined based on transition probabilities of the model.

# Quick Start
Training the models from scratch requires the corpus dataset, but it is not included due to distribution restrictions. It can be accessed  from [국립국어원 인공지능AI말평](https://kli.korean.go.kr/) [[Competition Link]](https://kli.korean.go.kr/benchmark/taskOrdtm/taskList.do?taskOrdtmId=103&clCd=END_TASK&subMenuId=sub01).

To train the BERT-based models, simply run either `python bert_plain.py` or `python bert_GRU.py`.

For the GPT-based model, fine-tune the model with `python gpt_train.py`. Then, run the following command to perform inference on `eval` dataset.
```
python gpt_inference.py --model-ckpt-path {model_checkpoint_path}
```

# Results

Used [klue/roberta-large](https://huggingface.co/klue/roberta-large) as the base for BERT models and [nlpai-lab/kullm-polyglot-12.8b-v2](https://huggingface.co/nlpai-lab/kullm-polyglot-12.8b-v2) for the GPT model.  
All the models are trained with default configurations in the script.

Below are F1-scores of resulting models obtained on the evaluation dataset.

|Model|F1-score (micro)|
|------------|-----|
|BERT (plain)|0.872|
|BERT (GRU)  |0.875|
|GPT         |0.878|
