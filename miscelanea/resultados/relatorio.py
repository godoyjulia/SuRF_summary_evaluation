import os
import json
import numpy as np
import pandas as pd

df = pd.DataFrame(columns=[ 'dataset','modelo',
                                               'quality_rouge1','quality_rouge2','quality_rougel','quality_f1','quality_em',
                                               'quality_cosseno','quality_bertscore','quality_moverscore',
                                               'faithfulness_rouge1','faithfulness_rouge2','faithfulness_rougel','faithfulness_f1','faithfulness_em',
                                               'faithfulness_cosseno','faithfulness_bert','faithfulness_moverscore'])
diretorios = ['cstnews_data','GPTextSum2_data','temario_data','wikilingua_data','xlsum_data']
for diretorio in diretorios:
    for root, dirs, files in os.walk(diretorio):
        for file in files:
            if file.endswith('_results.json'):
                config = file.replace('_results.json', '')\
                              .replace('xlsumm','XLSum')\
                              .replace('xlsum','XLSum')\
                              .replace('wiki', 'Wiki')\
                              .replace('cstnews', 'CSTNews')\
                              .replace('temario', 'TeMario')\
                              .replace('gptextsum2', 'GPTextSum')\
                              .replace('GPTextSum2', 'GPTextSum')\
                              .split('_data-')
                modelo, dataset = config[1].replace('_',' '), config[0]

                with open(os.path.join(root, file), encoding='utf-8') as json_file:
                    data = json.load(json_file)

                quality_cosseno,\
                quality_bertscore,\
                quality_moverscore,\
                quality_rouge1,quality_rouge2,quality_rougel,\
                quality_f1,quality_em = [],[],[],[],[],[],[],[]

                faithfulness_cosseno,\
                faithfulness_bertscore,\
                faithfulness_moverscore,\
                faithfulness_rouge1,faithfulness_rouge2,faithfulness_rougel,\
                faithfulness_f1,faithfulness_em = [],[],[],[],[],[],[],[]

                for item in data:
                    quality_cosseno.append(item['conteudo_pontuacao']['cosseno'])
                    quality_bertscore.append(item['conteudo_pontuacao']['bertscore'])
                    quality_moverscore.append(item['conteudo_pontuacao']['moverscore'])
                    quality_rouge1.append(item['conteudo_pontuacao']['rouge1'])
                    quality_rouge2.append(item['conteudo_pontuacao']['rouge2'])
                    quality_rougel.append(item['conteudo_pontuacao']['rougeL'])
                    quality_f1.append(item['conteudo_f1'])
                    quality_em.append(item['conteudo_em'])

                    faithfulness_cosseno.append(item['fieldade_pontuacao']['cosseno'])
                    faithfulness_bertscore.append(item['fieldade_pontuacao']['bertscore'])
                    faithfulness_moverscore.append(item['fieldade_pontuacao']['moverscore'])
                    faithfulness_rouge1.append(item['fieldade_pontuacao']['rouge1'])
                    faithfulness_rouge2.append(item['fieldade_pontuacao']['rouge2'])
                    faithfulness_rougel.append(item['fieldade_pontuacao']['rougeL'])
                    faithfulness_f1.append(item['fieldade_f1'])
                    faithfulness_em.append(item['fieldade_em'])

                quality_cosseno = round(np.average(quality_cosseno), 4)
                quality_bertscore = round(np.average(quality_bertscore), 4)
                quality_moverscore = round(np.average(quality_moverscore), 4)
                quality_rouge1,quality_rouge2,quality_rougel = round(np.average(quality_rouge1), 4), round(np.average(quality_rouge2), 4), round(np.average(quality_rougel), 4)
                quality_f1,quality_em = round(np.average(quality_f1), 4), round(np.average(quality_em), 4)

                faithfulness_cosseno = round(np.average(faithfulness_cosseno), 4)
                faithfulness_bertscore = round(np.average(faithfulness_bertscore), 4)
                faithfulness_moverscore = round(np.average(faithfulness_moverscore), 4)
                faithfulness_rouge1,faithfulness_rouge2,faithfulness_rougel = round(np.average(faithfulness_rouge1), 4), round(np.average(faithfulness_rouge2), 4), round(np.average(faithfulness_rougel))
                faithfulness_f1,faithfulness_em = round(np.average(faithfulness_f1), 4), round(np.average(faithfulness_em), 4)

                df.loc[len(df)] = [dataset,modelo,
                                   quality_rouge1,quality_rouge2,quality_rougel,quality_f1,quality_em,
                                   quality_cosseno,quality_bertscore,quality_moverscore,
                                   faithfulness_rouge1,faithfulness_rouge2,faithfulness_rougel,faithfulness_f1,faithfulness_em,
                                   faithfulness_cosseno,faithfulness_bertscore,faithfulness_moverscore,
                                ]
                
df.sort_values(['dataset','modelo'], inplace=True)
df.to_csv('relatorio.csv', index=False, encoding='utf-8')