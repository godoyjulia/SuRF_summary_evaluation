import os
import json
import math
import numpy as np

# removendo NaN
# diretorios = ['cstnews_data','GPTextSum_data','temario_data']
# for diretorio in diretorios:
#     for root, dirs, files in os.walk(diretorio):
#         for file in files:
#             if file.endswith(".json"):
#                 filepath = os.path.join(root, file)
#                 with open(filepath, 'r', encoding='utf-8') as f:
#                     text = f.read()
#                     print('%s read' % filepath)
#                 with open(filepath, 'w', encoding='utf-8') as f:
#                     f.write(text.replace(': NaN', ': 0'))
#                     print('%s updated' % filepath)

# fonte: https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#:~:text=F1%20score%20is%20a%20common%20metric,of%20words%20in%20the%20ground%20truth.&text=F1%20score%20is%20a,in%20the%20ground%20truth.&text=is%20a%20common%20metric,of%20words%20in%20the
# these functions are heavily influenced by the HF squad_metrics.py script
def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    # def remove_articles(text):
    #     regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    #     return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    # return white_space_fix(remove_articles(remove_punc(lower(s))))
    return white_space_fix(remove_punc(lower(s)))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
    print('prediction', prediction)
    print('truth', truth)
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)

diretorios = ['cstnews_data','GPTextSum_data','temario_data']
file_ok = []
for diretorio in diretorios:
    for root, dirs, files in os.walk(diretorio):
        for file in files:
            nome = file[:-14] # remove '_conteudo.json e '_fieldade.json'
            if nome not in file_ok and not file.endswith('_results.json'):
                integrado = []
                with open(os.path.join(root, f'{nome}_conteudo.json'), encoding='utf-8') as json_conteudo:
                    data_conteudo = json.load(json_conteudo)
                with open(os.path.join(root, f'{nome}_fieldade.json'), encoding='utf-8') as json_fieldade:
                    data_fieldade = json.load(json_fieldade)

                for item_conteudo in data_conteudo:
                    id = item_conteudo['id']

                    # calcula EM e F1score das perguntas sobre conteudo do resumo
                    conteudo_qg_qa = item_conteudo['conteudo_qg-qa']
                    conteudo_pred   = [yhat['resposta_obtida'] for yhat in conteudo_qg_qa]
                    conteudo_target = [y['resposta_ouro'] for y in conteudo_qg_qa]
                    conteudo_f1 = np.average([compute_f1(prediction=conteudo_pred[i], truth=conteudo_target[i]) for i in range(len(conteudo_pred))])
                    conteudo_em = np.average([compute_exact_match(prediction=conteudo_pred[i], truth=conteudo_target[i]) for i in range(len(conteudo_pred))])
                    item_conteudo['conteudo_f1'] = 0 if math.isnan(conteudo_f1) else conteudo_f1
                    item_conteudo['conteudo_em'] = 0 if math.isnan(conteudo_em) else conteudo_em

                    # calcula EM e F1score das perguntas sobre fieldade do resumo
                    item_fieldade = [x for x in data_fieldade if x['id']==id][0]
                    fieldade_qg_qa = item_fieldade['fieldade_qg-qa']
                    fieldade_pred   = [yhat['resposta_obtida'] for yhat in fieldade_qg_qa]
                    fieldade_target = [y['resposta_ouro'] for y in fieldade_qg_qa]
                    fieldade_f1 = np.average([compute_f1(prediction=fieldade_pred[i], truth=fieldade_target[i]) for i in range(len(fieldade_pred))])
                    fieldade_em = np.average([compute_exact_match(prediction=fieldade_pred[i], truth=fieldade_target[i]) for i in range(len(fieldade_pred))])
                    item_fieldade['fieldade_f1'] = 0 if math.isnan(fieldade_f1) else fieldade_f1
                    item_fieldade['fieldade_em'] = 0 if math.isnan(fieldade_em) else fieldade_em

                    # coloca todo conteudo calculado em 1 objeto (item_conteudo)
                    item_conteudo['fieldade_qg-original'] = item_fieldade['fieldade_qg-original']
                    item_conteudo['fieldade_qg-qa'] = item_fieldade['fieldade_qg-qa']
                    item_conteudo['fieldade_pontuacao'] = item_fieldade['fieldade_pontuacao']
                    item_conteudo['fieldade_similaridade-respostas'] = item_fieldade['fieldade_similaridade-respostas']
                    item_conteudo['fieldade_f1'] = item_fieldade['fieldade_f1']
                    item_conteudo['fieldade_em'] = item_fieldade['fieldade_em']
                            
                    integrado.append(item_conteudo)
                
                with open(os.path.join(root, f'{nome}.json'), 'w', encoding='utf-8') as json_file:
                    json.dump(integrado, json_file, indent=4, ensure_ascii=False)
