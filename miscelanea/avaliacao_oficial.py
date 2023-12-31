from pipeline_avaliacao import Avaliacao
from datasets import load_dataset
from datetime import datetime
from pathlib import Path
import json
import gc

t_ini_proc = datetime.now()

datasets = [
    # 'arthurmluz/GPTextSum2_data-cstnews_results', # ok
    # 'arthurmluz/GPTextSum2_data-temario_results', # ok
    # 'arthurmluz/GPTextSum2_data-wiki_results', # ok
    # 'arthurmluz/GPTextSum2_data-wiki_1024_results', # ok
    # 'arthurmluz/GPTextSum2_data-wiki_cstnews_results', # ok
    # 'arthurmluz/GPTextSum2_data-wiki_cstnews_1024_results', # ok
    # 'arthurmluz/GPTextSum2_data-wiki_temario_results', # ok
    # 'arthurmluz/GPTextSum2_data-wiki_gptextsum2_results', # ok
    # 'arthurmluz/GPTextSum2_data-xlsum_results', # ok
    # 'arthurmluz/GPTextSum2_data-xlsum_cstnews_results', # ok
    # 'arthurmluz/GPTextSum2_data-xlsum_cstnews_1024_results', # ok

    # 'arthurmluz/cstnews_data-xlsum_temario_results', # ok
    # 'arthurmluz/temario_data-xlsum_temario_results', # ok
    # 'arthurmluz/GPTextSum2_data-xlsum_temario_results', # ok

    # 'arthurmluz/cstnews_data-xlsum_gptextsum2_results', # ok
    # 'arthurmluz/temario_data-xlsum_gptextsum2_results', # ok
    # 'arthurmluz/GPTextSum2_data-xlsum_gptextsum2_results', # ok

    'arthurmluz/cstnews_data-wiki_gptextsum2_results',
    'arthurmluz/temario_data-wiki_gptextsum2_results',
# laboratorio
# 'wikilingua_data-xlsum_temario_results',
# 'xlsum_data-xlsum_temario_results',
# 'wikilingua_data-xlsum_gptextsum2_results',
# 'xlsum_data-xlsum_gptextsum2_results',
]



item_problema_passou = False
for nome_dataset in datasets:
    dataset = load_dataset(nome_dataset)['validation']

    print('total rows', dataset.num_rows)

    output_dir= f"resultados/{nome_dataset.split('/')[1].split('-')[0]}"
    nome = f"{output_dir}/{nome_dataset.split('/')[1]}"

    print(f'EXECUTANDO {nome}')

    # criar arquivo de output 
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f'{nome}_fieldade.json', 'w', encoding='utf-8') as jsonFile:
        json.dump([], jsonFile, indent=4)
    with open(f'{nome}_conteudo.json', 'w', encoding='utf-8') as jsonFile:
        json.dump([], jsonFile, indent=4)

    for idx_row in range(dataset.num_rows):

        id = dataset['id'][idx_row]
        texto_original = dataset['text'][idx_row]
        resumo_original = dataset['summary'][idx_row]
        resumo_gerado = dataset['gen_summary'][idx_row]
        
        # caso de problema no meio de uma execucao, para continuar a partir do id com erro, descomente:
        # if not item_problema_passou and id != 4278: continue
        # elif id == 4278: item_problema_passou=True

        t_ini = datetime.now()
        av = Avaliacao(nome=nome, id=id, resumo_original=resumo_original, resumo_gerado=resumo_gerado, texto_original=texto_original)
        av.avaliacao_conteudo()

        # pq memoria
        del av
        gc.collect()

        av = Avaliacao(nome=nome, id=id, resumo_original=resumo_original, resumo_gerado=resumo_gerado, texto_original=texto_original)
        av.avaliacao_fieldade()
        t_fim = datetime.now()

        del av
        gc.collect()

        print(f'({id}) concluido em {(t_fim-t_ini).seconds} segundos')



    t_fim_proc = datetime.now()
    print(f'concluído ({(t_fim_proc-t_ini_proc).seconds/60} min)')

    with open('logs_exec.txt', 'a+') as f:
        f.write(f'concluido {nome_dataset}')
