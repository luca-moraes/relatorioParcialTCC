import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        
    blocks = content.split('\n\n')
    data = []

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue

        dataset_info = lines[0]
        total_data = int(re.search(r'Total de dados: (\d+)', lines[1]).group(1))
        treino_data = int(re.search(r'Quantidade de dados de treino: (\d+)', lines[2]).group(1))
        teste_data = int(re.search(r'Quantidade de dados de teste: (\d+)', lines[3]).group(1))
        eqm_before = float(re.search(r'Erro quadrático médio \(antes da clippagem\): ([\d\.]+)', lines[4]).group(1))
        ema_before = float(re.search(r'Erro médio absoluto \(antes da clippagem\): ([\d\.]+)', lines[5]).group(1))
        
        # Ajustando a regex para pesos otimizados com possibilidade de sinais negativos
        pesos_otimizados_match = re.search(r'Pesos otimizados: \[([\d\.\s, -]+)\]', lines[6])
        if pesos_otimizados_match:
            pesos_otimizados_str = pesos_otimizados_match.group(1).replace(',', ' ')
            pesos_otimizados = list(map(float, pesos_otimizados_str.split()))
        else:
            pesos_otimizados = []

        eqm_after = float(re.search(r'Erro quadrático médio \(após clippagem\): ([\d\.]+)', lines[7]).group(1))
        ema_after = float(re.search(r'Erro médio absoluto \(após clippagem\): ([\d\.]+)', lines[8]).group(1))
        num_testes = int(re.search(r'Numero de Testes: (\d+)', lines[9]).group(1))
        erro_geral = float(re.search(r'Erro Geral do Algoritmo: ([\d\.]+)%', lines[10]).group(1))
        
        erro_faixa = {}
        for line in lines[12:]:
            faixa_match = re.search(r'(\d+-\d+)% Quantidade: (\d+) - Percentual: ([\d\.]+)', line)
            if faixa_match:
                faixa, quantidade, percentual = faixa_match.groups()
                erro_faixa[faixa] = (int(quantidade), float(percentual))

        data.append({
            'info': dataset_info,
            'total_data': total_data,
            'treino_data': treino_data,
            'teste_data': teste_data,
            'eqm_before': eqm_before,
            'ema_before': ema_before,
            'pesos_otimizados': pesos_otimizados,
            'eqm_after': eqm_after,
            'ema_after': ema_after,
            'num_testes': num_testes,
            'erro_geral': erro_geral,
            'erro_faixa': erro_faixa
        })

    return data

def plot_bars(erro_faixa, title, output_dir):
    faixas = list(erro_faixa.keys())
    quantidades = [erro_faixa[faixa][0] for faixa in faixas]
    percentuais = [erro_faixa[faixa][1] for faixa in faixas]
    
    plt.figure(figsize=(10, 6))
    plt.bar(faixas, quantidades, color='blue', alpha=0.7)
    plt.xlabel('Faixa de Erro (%)')
    plt.ylabel('Quantidade')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, title + '_quantidade.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.bar(faixas, percentuais, color='green', alpha=0.7)
    plt.xlabel('Faixa de Erro (%)')
    plt.ylabel('Percentual')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, title + '_percentual.png'))
    plt.close()

def create_table(data, output_dir, filename):
    df = pd.DataFrame(data, columns=['Info', 'Total Data', 'Treino Data', 'Teste Data', 'EQM Before', 'EMA Before', 'Pesos Otimizados', 'EQM After', 'EMA After', 'Num Testes', 'Erro Geral'])
    df.to_csv(os.path.join(output_dir, filename), index=False)

def main(input_file):
    data = parse_data(input_file)
    
    categorized_data = defaultdict(list)

    for entry in data:
        dataset_info = entry['info']
        lang, base_large = dataset_info.split()[:2]
        category = f"{lang} {base_large}"

        output_dir = f'../resultadosRegressao/dados_{lang}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        categorized_data[category].append([
            dataset_info,
            entry['total_data'],
            entry['treino_data'],
            entry['teste_data'],
            entry['eqm_before'],
            entry['ema_before'],
            entry['pesos_otimizados'],
            entry['eqm_after'],
            entry['ema_after'],
            entry['num_testes'],
            entry['erro_geral']
        ])
        
        plot_bars(entry['erro_faixa'], dataset_info, output_dir)
    
    for category, data_list in categorized_data.items():
        lang = category.split()[0]
        output_dir = f'../resultadosRegressao/dados_{lang}'
        create_table(data_list, output_dir, f"{category.replace(' ', '_')}.csv")

if __name__ == '__main__':
    input_file = '../resultadosRegressao/dadosReunidos.txt'
    main(input_file)
