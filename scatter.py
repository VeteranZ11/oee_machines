import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

# Ustawienie maksymalnej liczby wyświetlanych wierszy i kolumn
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 50)

# Ładowanie danych z pliku CSV
file_path = 'tmp325c.csv'
data = pd.read_csv(file_path, encoding='utf-16', sep='\t')

# Czyszczenie danych
cleaned_data = data.dropna(axis=1, how='all')

# Lista do przechowywania wyników
results = []

# Iteracja przez wszystkie unikalne nazwy maszyn
unique_machines = cleaned_data['Name (Allgemeine Informationen)'].unique()

for machine_name in unique_machines:
    try:
        # Filtracja danych dla danej maszyny
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
        
        # Agregacja danych dziennych
        aggregated_data = filtered_machine_data.groupby('Schichtdatum')[numeric_columns].sum()
        aggregated_data.reset_index(inplace=True)
        aggregated_data['Efficiency (%)'] = (aggregated_data['Produzierte Menge'] / aggregated_data['Geplante Menge']) * 100
        aggregated_data['Efficiency (%)'] = aggregated_data['Efficiency (%)'].fillna(0).replace([float('inf'), -float('inf')], 0)
        
        # Obliczenie średniej efektywności
        mean_efficiency = aggregated_data[aggregated_data['Efficiency (%)'] !=0]['Efficiency (%)'].mean()
        
        # Konwersja kolumn czasowych do minut
        time_columns = [
            'Stillstandszeit (geplant)', 'Stillstandszeit (ungeplant)', 
            'Stillstandszeit (ohne Einfluss)', 'Stillstandszeit (ohne Grund)', 
            'Hauptnutzungszeit'
        ]
        
        def convert_to_minutes(time_str):
            if pd.isna(time_str):
                return 0
            hours, minutes = map(int, time_str.split(':'))
            return hours * 60 + minutes
        
        for column in time_columns:
            filtered_machine_data[column] = filtered_machine_data[column].apply(convert_to_minutes)
        
        # Agregacja czasów
        aggregated_times = filtered_machine_data.groupby('Schichtdatum')[time_columns].sum()
        total_times = filtered_machine_data[time_columns].sum()
        total_stillstand_and_usage_time = total_times.sum()
        
        # Obliczenie procentu Hauptnutzungszeit uwzględniając Stillstandszeit (geplant)
        percentage_hauptnutzungszeit = ((total_times['Stillstandszeit (ohne Grund)']) / total_stillstand_and_usage_time) * 100
        
        # Zapis wyników do listy
        results.append({
            'Machine Name': machine_name,
            'Mean Efficiency': mean_efficiency,
            'Percentage (Hauptnutzungszeit)': percentage_hauptnutzungszeit
        })
    
    except Exception as e:
        print(f"Error processing machine {machine_name}: {e}")

# Konwersja wyników do DataFrame
results_df = pd.DataFrame(results)

# Usunięcie wierszy zawierających NaN w kolumnach 'Mean Efficiency' i 'Percentage (Hauptnutzungszeit)'
results_df = results_df.dropna(subset=['Mean Efficiency', 'Percentage (Hauptnutzungszeit)'])

# Klastrowanie k-means
kmeans = KMeans(n_clusters=3, random_state=0).fit(results_df[['Mean Efficiency', 'Percentage (Hauptnutzungszeit)']])
results_df['Cluster'] = kmeans.labels_

# Mapowanie kolorów klastrów
cluster_colors = {0: 'To check', 1: 'Normal', 2: 'Good'}
results_df['Color'] = results_df['Cluster'].map(cluster_colors)

# Tworzenie wykresu punktowego za pomocą Plotly z kolorami klastrów
fig = px.scatter(
    results_df,
    x='Mean Efficiency',
    y='Percentage (Hauptnutzungszeit)',
    hover_data=['Machine Name', 'Mean Efficiency', 'Percentage (Hauptnutzungszeit)'],
    labels={
        'Mean Efficiency': 'Mean Efficiency (%)',
        'Percentage (Hauptnutzungszeit)': 'Percentage (Hauptnutzungszeit) (%)'
    },
    title='Scatter Plot of Machine Efficiency vs. Percentage of Hauptnutzungszeit',
    color='Color'
)

# Aktualizacja układu wykresu dla klasycznego stylu z białym tłem
fig.update_layout(
    xaxis_title='Mean Efficiency (%)',
    yaxis_title='Percentage (Hauptnutzungszeit) (%)',
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(
        family='Arial, sans-serif',
        size=12,
        color='black'
    ),
    xaxis=dict(
        gridcolor='lightgrey',
        showline=True,
        linewidth=1,
        linecolor='black'
    ),
    yaxis=dict(
        gridcolor='lightgrey',
        showline=True,
        linewidth=1,
        linecolor='black'
    ),
    legend=dict(
        bgcolor='white',
        bordercolor='black',
        borderwidth=1
    )
)

# Wyświetlenie wykresu
fig.show()