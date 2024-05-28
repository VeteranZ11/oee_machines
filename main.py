import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import argparse

# Ustawienie maksymalnej liczby wyświetlanych wierszy i kolumn
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 50)

# Funkcja do konwersji hh:mm na minuty
def convert_to_minutes(time_str):
    if pd.isna(time_str):
        return 0
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes

# Parsowanie argumentów wiersza poleceń
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('search_number', type=str, help='The search number to filter machine names')
args = parser.parse_args()
search_number = args.search_number

# Ładowanie danych z pliku CSV
file_path = 'tmp325c.csv'
data = pd.read_csv(file_path, encoding='utf-16', sep='\t')

# Czyszczenie danych
cleaned_data = data.dropna(axis=1, how='all')

# Filtracja nazw maszyn zawierających podany numer
matching_machine_names = cleaned_data[cleaned_data['Name (Allgemeine Informationen)'].str.contains(search_number, na=False)]
unique_matching_machines = matching_machine_names['Name (Allgemeine Informationen)'].unique()
print(f"Maszyny zawierające '{search_number}': {unique_matching_machines}")
if len(unique_matching_machines) == 0:
    print(f"No machines found containing '{search_number}'")
    exit()

machine_name = unique_matching_machines[0]

# Filtracja danych dla wybranej maszyny
filtered_machine_data = cleaned_data[cleaned_data['Name (Allgemeine Informationen)'] == machine_name]
filtered_machine_data = filtered_machine_data.sort_values(['Schichtdatum', 'Schichtnummer', 'Beginn'])
filtered_machine_data['Schichtdatum'] = pd.to_datetime(filtered_machine_data['Schichtdatum'], format='%d.%m.%Y')

# Konwersja kolumn z wartościami liczbowymi
numeric_columns = ['Produzierte Menge', 'Gutmenge', 'Geplante Menge']
for column in numeric_columns:
    filtered_machine_data[column] = filtered_machine_data[column].str.replace('.', '').str.replace(',', '.').astype(float).astype(int)

# Filtracja wierszy z wartościami mniejszymi niż 10, ale różnymi od zera
for column in numeric_columns:
    filtered_machine_data = filtered_machine_data[~((filtered_machine_data[column] < 10) & (filtered_machine_data[column] != 0))]

# Usunięcie wierszy, gdzie 'Produzierte Menge' lub 'Gutmenge' są równe 0, ale 'Geplante Menge' jest większe niż 0
filtered_machine_data = filtered_machine_data[~(
    ((filtered_machine_data['Produzierte Menge'] == 0) & (filtered_machine_data['Geplante Menge'] > 0)) |
    ((filtered_machine_data['Gutmenge'] == 0) & (filtered_machine_data['Geplante Menge'] > 0))
)]

print(filtered_machine_data)

# Agregacja danych dziennych
aggregated_data = filtered_machine_data.groupby('Schichtdatum')[numeric_columns].sum()
aggregated_data.reset_index(inplace=True)
aggregated_data['Efficiency (%)'] = (aggregated_data['Produzierte Menge'] / aggregated_data['Geplante Menge']) * 100
aggregated_data['Efficiency (%)'] = aggregated_data['Efficiency (%)'].fillna(0).replace([float('inf'), -float('inf')], 0)
mean_efficiency = aggregated_data[aggregated_data['Efficiency (%)'] !=0]['Efficiency (%)'].mean()

# Konwersja kolumn czasowych do minut
time_columns = [
    'Stillstandszeit (geplant)', 'Stillstandszeit (ungeplant)', 
    'Stillstandszeit (ohne Einfluss)', 'Stillstandszeit (ohne Grund)', 
    'Hauptnutzungszeit'
]
for column in time_columns:
    filtered_machine_data[column] = filtered_machine_data[column].apply(convert_to_minutes)

# Agregacja czasów
aggregated_times = filtered_machine_data.groupby('Schichtdatum')[time_columns].sum()

# Obliczenie całkowitego czasu dla każdej kategorii
total_times = filtered_machine_data[time_columns].sum()
total_stillstand_and_usage_time = total_times.sum()

# Obliczenie procentu "ohne Grund" i "ungeplant"
percentage_ohne_grund = (total_times['Stillstandszeit (ohne Grund)'] / total_stillstand_and_usage_time) * 100
percentage_ungeplant = (total_times['Stillstandszeit (ungeplant)'] / total_stillstand_and_usage_time) * 100

print(f"Percentage of 'Stillstandszeit (ohne Grund)': {percentage_ohne_grund:.2f}%")
print(f"Percentage of 'Stillstandszeit (ungeplant)': {percentage_ungeplant:.2f}%")

# Tworzenie wykresów
fig, axs = plt.subplots(3, 1, figsize=(12, 8))
fig.text(0.01, 0.99, f'{machine_name}', fontsize=18, va='top', ha='left', fontweight='bold')

# Pierwszy wykres: Przegląd danych produkcyjnych
axs[0].plot(aggregated_data['Schichtdatum'], aggregated_data['Produzierte Menge'], 
            label='Produzierte Menge', marker='o', color='#ab32ad')
axs[0].plot(aggregated_data['Schichtdatum'], aggregated_data['Gutmenge'], 
            label='Gutmenge', marker='o', linestyle='--', color='#389e2c')
axs[0].plot(aggregated_data['Schichtdatum'], aggregated_data['Geplante Menge'], 
            label='Geplante Menge', marker='o', color='#2da5ad')
axs[0].set_title('Production Data Overview')
axs[0].set_ylabel('Quantity')
axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axs[0].tick_params(axis='x', rotation=20)

# Drugi wykres: Dzienne efektywności produkcji
axs[1].plot(aggregated_data['Schichtdatum'], aggregated_data['Efficiency (%)'], 
            label='Daily Efficiency', color='green', marker='o')
axs[1].axhline(mean_efficiency, color='red', linestyle='--', label=f'Mean Efficiency: {mean_efficiency:.2f}%')
axs[1].set_title('Daily Production Efficiency')
axs[1].set_ylabel('Efficiency (%)')
axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axs[1].tick_params(axis='x', rotation=20)

# Trzeci wykres: Dzienne użytkowanie maszyn i czas przestoju w minutach
bottom = np.zeros(len(aggregated_times))
for column in time_columns:
    axs[2].bar(aggregated_times.index, aggregated_times[column], bottom=bottom, label=column)
    bottom += aggregated_times[column].values
axs[2].set_title('Daily Machine Usage and Downtime in Minutes')
axs[2].set_ylabel('Minutes')
axs[2].legend(title='Time Categories', bbox_to_anchor=(1.05, 1), loc='upper left')
axs[2].tick_params(axis='x', rotation=20)

# Grid dla wszystkich wykresów
axs[0].grid(True, linestyle='-', linewidth=0.5, which='both')
axs[1].grid(True, linestyle='-', linewidth=0.5, which='both')

# Opcjonalnie: Ustawienie minor ticks dla bardziej szczegółowej siatki
from matplotlib.ticker import AutoMinorLocator
axs[0].xaxis.set_minor_locator(AutoMinorLocator())
axs[0].yaxis.set_minor_locator(AutoMinorLocator())
axs[1].xaxis.set_minor_locator(AutoMinorLocator())
axs[1].yaxis.set_minor_locator(AutoMinorLocator())
axs[2].xaxis.set_minor_locator(AutoMinorLocator())
axs[2].yaxis.set_minor_locator(AutoMinorLocator())

# Formatowanie etykiet dat na osi x
date_format = mdates.DateFormatter('%d-%m')
for ax in axs:
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_fontsize(8)

# Ustawienie minor locator dla osi x
minor_locator = mdates.DayLocator(interval=1)
for ax in axs:
    ax.xaxis.set_minor_locator(minor_locator)

plt.tight_layout()  # Zapewnienie, że layout jest dostosowany do nowych ustawień osi
fig.subplots_adjust(top=0.9)
plt.show()
