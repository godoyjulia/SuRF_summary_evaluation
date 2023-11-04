# conda env= tcc-pipeline-avaliacao
from question_generation.pipelines import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from evaluate import load
import sys
sys.path.insert(0, '/home/godoy/Documents/pucrs/tcc/avaliacao/emnlp19_moverscore')
from moverscore_v2 import word_mover_score # https://github.com/AIPHES/emnlp19-moverscore/tree/master
from collections import defaultdict
import numpy as np
import json

def salvar(file, output):
    # precisa criar um json vazio antes (apenas com [])
    with open(file, "r+", encoding='utf-8') as jsonFile:
        data = json.load(jsonFile)
        jsonFile.truncate(0)
        jsonFile.seek(0)

        data.append(output)
        json.dump(data, jsonFile, indent=4, ensure_ascii=False)
        jsonFile.close()


class Avaliacao:

    def __init__(self, nome, id, resumo_original, resumo_gerado, texto_original):
        self.id = id
        self.nome = nome
        self.resumo_original = resumo_original
        self.resumo_gerado = resumo_gerado
        self.texto_original = texto_original
        self.bertScore = load('bertscore')
        self.rouge = load('rouge')

    def metrica_cosseno(self, resposta1, resposta2):
        # qnt mais perto de 1, mais similar
        vectorizer = CountVectorizer()

        vectorizer.fit([resposta1, resposta2])
        vector1, vector2 = vectorizer.transform([resposta1, resposta2])

        cosine_sim = cosine_similarity(vector1, vector2)
        return cosine_sim[0][0]
    
    def metrica_bertscore(self, resposta1, resposta2):
        # qnt mais perto de 1, mais similar
        return self.bertScore.compute(predictions=[resposta1], references=[resposta2], lang='pt').get('f1')[0]

    def metrica_moverscore(self, resposta1, resposta2):
        idf_dict_hyp =  defaultdict(lambda: 1.)
        idf_dict_ref = defaultdict(lambda: 1.)

        return word_mover_score([resposta1], [resposta2], idf_dict_ref, idf_dict_hyp, \
                                  stop_words=[], n_gram=1, remove_subwords=True)[0]
    
    def metrica_rouge(self, resposta1, resposta2):
        # qnt mais perto de 1, mais similar
        return self.rouge.compute(predictions=[resposta1], references=[resposta2])
        

    def avaliacao_similaridade_respostas(self, qa, threshold):
        metrica = {}
        metrica['cosseno'] = []
        metrica['bertscore'] = []
        metrica['moverscore'] = []
        metrica['rouge1'] = []
        metrica['rouge2'] = []
        metrica['rougeL'] = []
        metrica['rougeLsum'] = []
        for qa_ in qa:
            resposta1 = qa_['resposta_ouro']
            resposta2 = qa_['resposta_obtida']
            
            # analise das metricas
            metrica['cosseno'].append(self.metrica_cosseno(resposta1, resposta2))
            metrica['bertscore'].append(self.metrica_bertscore(resposta1, resposta2))
            metrica['moverscore'].append(self.metrica_moverscore(resposta1, resposta2))

            rouges = self.metrica_rouge(resposta1, resposta2)
            for k, v in rouges.items():
                metrica[k].append(v)

            # print(metrica)

        metrica_ = metrica.copy() # safecopy, metrica antes de alteracoes de pontuacao
        for k in metrica.keys():
            # metrica[k] = np.mean(metrica[k]) # media dos resultados
            metrica[k] = np.mean([1 if sim>=threshold else 0 for sim in metrica[k]]) # media da 'pontuacao' (positiva caso sim. maior q threshold)

        return metrica, metrica_


    def avaliacao_fieldade(self):
        self.gera_peruntas_sobre = self.resumo_gerado
        self.extrai_respostas_sobre = self.resumo_original

        qg_ = self.gera_perguntas() 
        qg = self.validacao_pergunta(qg_)
        qa = self.extrai_respostas(qg, fieldade=True)
        pontuacao, sim_respostas = self.avaliacao_similaridade_respostas(qa, threshold=0.7)

        # print('AVALIAÇÃO FIELDADE')
        # print('AVALIAÇÃO FIELDADE. qg', json.dumps(qg, indent=2, ensure_ascii=False), '\n')
        # print('AVALIAÇÃO FIELDADE. qa', json.dumps(qa, indent=2, ensure_ascii=False), '\n')
        # print('AVALIAÇÃO FIELDADE. similaridade entre as respostas',pontuacao)

        file = f'{self.nome}_fieldade.json'
        output = {}
        output['id'] = self.id
        output['text'] = self.texto_original
        output['summary'] = self.resumo_original
        output['summary_gen'] = self.resumo_gerado
        output['fieldade_qg-original'] = qg_
        output['fieldade_qg-qa'] = qa
        output['fieldade_pontuacao'] = pontuacao
        output['fieldade_similaridade-respostas'] = sim_respostas

        salvar(file, output)

    def avaliacao_conteudo(self):
        self.gera_peruntas_sobre = self.resumo_original
        self.extrai_respostas_sobre = self.resumo_gerado

        qg_ = self.gera_perguntas() 
        qg = self.validacao_pergunta(qg_)
        qa = self.extrai_respostas(qg, fieldade=False)
        pontuacao, sim_respostas = self.avaliacao_similaridade_respostas(qa, threshold=0.7)

        # print('AVALIAÇÃO CONTEÚDO')
        # print('AVALIAÇÃO CONTEÚDO. qg', json.dumps(qg, indent=2, ensure_ascii=False), '\n')
        # print('AVALIAÇÃO CONTEÚDO. qa', json.dumps(qa, indent=2, ensure_ascii=False), '\n')
        # print('AVALIAÇÃO CONTEÚDO. similaridade entre as respostas',pontuacao)

        file = f'{self.nome}_conteudo.json'
        output = {}
        output['id'] = self.id
        output['text'] = self.texto_original
        output['summary'] = self.resumo_original
        output['summary_gen'] = self.resumo_gerado
        output['conteudo_qg-original'] = qg_
        output['conteudo_qg-qa'] = qa
        output['conteudo_pontuacao'] = pontuacao
        output['conteudo_similaridade-respostas'] = sim_respostas

        salvar(file, output)


    def extrai_respostas(self, perguntas_respostas_alvo, fieldade:bool):
        if fieldade:
            txt_extrai_resposta_sobre = self.texto_original
        else:
            txt_extrai_resposta_sobre = self.extrai_respostas_sobre


        model_name = "pierreguillou/t5-base-qa-squad-v1.1-portuguese"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        resultado = []
        for idx, qa in enumerate(perguntas_respostas_alvo):
            input_text = f"question: {qa['question']} context: {txt_extrai_resposta_sobre}"
            label = qa['answer']

            inputs = tokenizer(input_text, return_tensors="pt")

            outputs = model.generate(inputs["input_ids"],
                                         max_length=32, 
                                         num_beams=2, 
                                         early_stopping=True)
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

            resultado.append({'id':idx,
                              'pergunta': qa['question'],
                              'resposta_ouro': label,
                              'resposta_obtida': pred})
        return resultado



    def gera_perguntas(self):
        path_to_checkpoint = 'vabatista/question-generation-t5-small-pt-br-2'
        path_to_tokenizer = 'vabatista/question-generation-t5-small-pt-br-2'
        nlp = pipeline("question-generation",
                       model=path_to_checkpoint,
                       tokenizer=path_to_tokenizer,
                       ans_model=path_to_checkpoint,
                       ans_tokenizer=path_to_tokenizer,
                       qg_format='highlight', use_cuda=False)

        return nlp(self.gera_peruntas_sobre)

    def validacao_pergunta(self, perguntas_respostas_alvo):
        '''
            teste com o extrator de respostas, buscando no mesmo texto que o gerador de perguntas usou
            caso não consiga encontrar a resposta (exact match seja diferente)
            então descartamos essa pergunta, pois não a consideramos prática
        '''
        extrai_respostas_sobre_original = self.extrai_respostas_sobre
        self.extrai_respostas_sobre = self.gera_peruntas_sobre

        qa_val = self.extrai_respostas(perguntas_respostas_alvo, fieldade=False)

        idx_remover_perguntas = []
        for qa_val_ in qa_val:
            id = qa_val_['id']
            if qa_val_['resposta_ouro'] != qa_val_['resposta_obtida']: # na validacao esta buscando por EM (exact match), talvez nao seja a melhor forma? fazer a verificacao da similaridade tambem?
                idx_remover_perguntas.append(id)
        
        novo_perguntas_respostas_alvo = []
        for idx, item in enumerate(perguntas_respostas_alvo):
            if idx not in idx_remover_perguntas:
                novo_perguntas_respostas_alvo.append(item)

        # print(f'validacao das perguntas removeu {len(idx_remover_perguntas)} das {len(qa_val)} perguntas')

        self.extrai_respostas_sobre = extrai_respostas_sobre_original
        return novo_perguntas_respostas_alvo


# resumo_original = """
# O ministro da Defesa, Nelson Jobim, informou que a economista Solange Vieira será
# a nova presidente da Agência Nacional de Aviação Civil. Ela substitui o atual presidente,
# Milton Zuanazzi, porque ele está renunciando devido a críticas incisivas da oposição desde
# o acidente com o Airbus da TAM. Os dois ministros que davam sustentação a Zuanazzi no
# cargo, Dilma Rousseff e Walfrido dos Mares Guia, avaliam que ele se tornou uma figura muito
# vinculada à crise aérea e passaram a defender sua substituição. Solange foi escolhida devido à
# sua experiência quando comandou os fundos de pensão do país, promovendo uma expressiva
# reforma em sua legislação e aumentando a transparência do setor. O acidente da TAM ainda
# levou a uma análise CPI do Apagão da Câmara sobre a administração da infra-estrutura dos
# aeroportos.
# """

# resumo_gerado = """ 
# A economista Solange Vieira, de 38 anos, será a nova presidente da Agência Nacional
# de Aviação Civil no governo Fernando Henrique Cardoso. O ministro da defesa, Nelson Jobim,
# informou no fim da noite desta terça-feira que a economista Solange Vieira será o novo
# presidente na Agência Nacional de Aviação civil. Ele já renunciou e deve entregar o cargo
# nos próximos dias. Os dois diretores de agências têm mandato de cinco anos e só podem sair
# por renúncia, decisão judicial ou acusação de improbidade administrativa. Quando o último
# remanescente da diretoria da Anac à época do acidente, Josef Barat, diretor de relações
# internacionais, pesquisas e capacitação da Anac, também deve sair.
# """


# from datetime import datetime
# import gc

# t_ini = datetime.now()
# av = Avaliacao(nome='teste', id=1, resumo_original=resumo_original, resumo_gerado=resumo_gerado, texto_original=texto_original)
# av.avaliacao_conteudo()

# print()
# del av
# gc.collect()
# print()

# av = Avaliacao(nome='teste', id=1, resumo_original=resumo_original, resumo_gerado=resumo_gerado, texto_original=texto_original)
# av.avaliacao_fieldade()
# t_fim = datetime.now()

# print()
# del av
# gc.collect()
# print()

# print(f'tempo de execucao :{(t_fim-t_ini).seconds} segundos')