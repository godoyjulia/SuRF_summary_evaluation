from pipeline_avaliacao import Avaliacao
from datasets import load_dataset
from datetime import datetime
import json
import gc

t_ini_proc = datetime.now()


nome_dataset = 'arthurmluz/temario_data-wiki_results'
dataset = load_dataset(nome_dataset)['validation']

print(dataset.num_rows)

nome = nome_dataset.split('/')[1]

print(f'EXECUTANDO {nome}')

# criar arquivo de output 
with open(f'{nome}_fieldade.json', 'w', encoding='utf-8') as jsonFile:
    json.dump([], jsonFile, indent=4)
with open(f'{nome}_conteudo.json', 'w', encoding='utf-8') as jsonFile:
    json.dump([], jsonFile, indent=4)


for idx_row in range(dataset.num_rows):

    id = dataset['id'][idx_row]
    texto_original = dataset['text'][idx_row]
    resumo_original = dataset['summary'][idx_row]
    resumo_gerado = dataset['gen_summary'][idx_row]

    t_ini = datetime.now()
    av = Avaliacao(nome=nome, id=id, resumo_original=resumo_original, resumo_gerado=resumo_gerado, texto_original=texto_original)
    av.avaliacao_conteudo()

    del av
    gc.collect()

    av = Avaliacao(nome=nome, id=id, resumo_original=resumo_original, resumo_gerado=resumo_gerado, texto_original=texto_original)
    av.avaliacao_fieldade()
    t_fim = datetime.now()

    del av
    gc.collect()

    print(f'({id}) concluido em {(t_fim-t_ini).seconds} segundos')



t_fim_proc = datetime.now()
print(f'concluído ({(t_fim_proc-t_ini_proc).min} min)')





