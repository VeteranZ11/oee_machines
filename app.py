import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import base64

# Ustawienie trybu "wide mode"
st.set_page_config(layout="wide")

# Funkcja do konwersji czasu w formacie 'hh:mm' lub 'hh:mm:ss' na minuty
def convert_to_minutes(time_str):
    try:
        parts = time_str.split(':')
        hours, minutes = map(int, parts[:2])  # Weź tylko godziny i minuty, zignoruj sekundy
        return hours * 60 + minutes
    except:
        return np.nan

# Funkcja do ładowania i czyszczenia danych
def load_data(file_path, encoding, separator):
    try:
        data = pd.read_csv(file_path, encoding=encoding, sep=separator)
        cleaned_data = data.dropna(axis=1, how='all')
        cleaned_data['Schichtdatum'] = pd.to_datetime(cleaned_data['Schichtdatum'], format='%d.%m.%Y')
        return cleaned_data
    except UnicodeDecodeError:
        print(f"Failed to load data with {encoding} encoding and {separator} delimiter. Try changing encoding.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}. Check file format and delimiter settings.")
        return None

# Funkcja do filtrowania danych
def filter_data(data, search_number, type):
    if type == 'machine':
        matching_machine_names = data[data['Name (Allgemeine Informationen)'].str.contains(search_number, na=False)]
        unique_matching_machines = matching_machine_names['Name (Allgemeine Informationen)'].unique()
        return unique_matching_machines
    elif type == 'part':
        matching_part_names = data[data['Name (Artikel)'].str.contains(search_number, na=False)]
        unique_matching_parts = matching_part_names['Name (Artikel)'].unique()
        return unique_matching_parts


# Funkcja do obliczania wskaźników OEE
def calculate_oee(filtered_machine_data):
    oee_data = filtered_machine_data[['Beginn', 'Ende', 'Schichtdatum', 'Schichtnummer', 'Name (Auftrag)',
                                      'Produzierte Menge', 'Gutmenge', 'Geplante Menge', 'Zykluszeit (geplant)', 
                                      'Hauptnutzungszeit', 'Betriebszeit', 'Name (Artikel)',
                                      'Stillstandszeit (geplant)', 'Stillstandszeit (ungeplant)',
                                      'Stillstandszeit (ohne Einfluss)', 'Stillstandszeit (ohne Grund)',
                                      'Name (Aktueller Standort)']]
    
    oee_data['Hauptnutzungszeit'] = oee_data['Hauptnutzungszeit'].apply(convert_to_minutes)
    oee_data['Betriebszeit'] = oee_data['Betriebszeit'].apply(convert_to_minutes)
    
    # Zbiór do przechowywania odfiltrowanych przypadków
    filtered_out_data = pd.DataFrame()

    # Dodanie filtrowania, aby usunąć wiersze, gdzie Betriebszeit < Hauptnutzungszeit
    #oee_data = oee_data[oee_data['Betriebszeit'] >= oee_data['Hauptnutzungszeit']]
    
    # Filtr Betriebszeit < Hauptnutzungszeit
    filter1 = oee_data['Betriebszeit'] < oee_data['Hauptnutzungszeit']
    filtered_out_data = pd.concat([filtered_out_data, oee_data[filter1]])
    oee_data = oee_data[~filter1]

    oee_data['Produzierte Menge'] = oee_data['Produzierte Menge'].str.replace('.', '').str.replace(',', '.').astype(float)
    oee_data['Gutmenge'] = oee_data['Gutmenge'].str.replace('.', '').str.replace(',', '.').astype(float)
    oee_data['Geplante Menge'] = oee_data['Geplante Menge'].str.replace('.', '').str.replace(',', '.').astype(float)
    oee_data['Geplante Menge'] = round(oee_data['Geplante Menge'])
    oee_data['Zykluszeit (geplant)'] = oee_data['Zykluszeit (geplant)'].str.replace(',', '.').astype(float)
    
    # # Dodanie filtrowania, aby usunąć wiersze, gdzie Gutmenge jest ujemne
    # oee_data = oee_data[oee_data['Gutmenge'] >= 0]

    # Filtr Gutmenge jest ujemne
    filter2 = oee_data['Gutmenge'] < 0
    filtered_out_data = pd.concat([filtered_out_data, oee_data[filter2]])
    oee_data = oee_data[~filter2]

    oee_data['Planned Produzierte Menge'] = np.where(oee_data['Zykluszeit (geplant)'] != 0,
                                                     oee_data['Hauptnutzungszeit'] / oee_data['Zykluszeit (geplant)'],
                                                     0)
    oee_data['Planned Produzierte Menge'] = round(oee_data['Planned Produzierte Menge'])
    
    oee_data['Availability'] = np.where(oee_data['Betriebszeit'] != 0, oee_data['Hauptnutzungszeit'] / oee_data['Betriebszeit'], 0)
    oee_data['Performance'] = np.where(oee_data['Planned Produzierte Menge'] != 0,
                                       oee_data['Produzierte Menge'] / oee_data['Planned Produzierte Menge'],
                                       0)
    oee_data['Quality'] = np.where(oee_data['Produzierte Menge'] != 0, 
                                   oee_data['Gutmenge'] / oee_data['Produzierte Menge'], 
                                   0)
    oee_data['OEE'] = oee_data['Availability'] * oee_data['Performance'] * oee_data['Quality']
    oee_data = oee_data[oee_data['OEE'] != 0]

    return oee_data, filtered_out_data

# Funkcja do obliczania OEE dla wszystkich maszyn
def calculate_oee_for_all_machines(data):
    all_machines = data['Name (Allgemeine Informationen)'].unique()
    oee_summary = []
    all_filtered_out_data = pd.DataFrame()

    for machine in all_machines:
        try:
            filtered_machine_data = data[data['Name (Allgemeine Informationen)'] == machine]
            oee_data, filtered_out_data = calculate_oee(filtered_machine_data)

            # Dodanie odfiltrowanych danych do głównego dataframe
            all_filtered_out_data = pd.concat([all_filtered_out_data, filtered_out_data])
            
            if not oee_data.empty:
                avg_oee = oee_data['OEE'].mean() * 100  # Przelicz OEE na procenty
                avg_availability = oee_data['Availability'].mean() * 100  # Przelicz Availability na procenty
                avg_performance = oee_data['Performance'].mean() * 100  # Przelicz Performance na procenty
                avg_quality = oee_data['Quality'].mean() * 100  # Przelicz Quality na procenty
                location = filtered_machine_data['Name (Aktueller Standort)'].iloc[0]  # Dodaj lokalizację maszyny
                oee_summary.append({
                    'Machine': machine,
                    'OEE': avg_oee,
                    'Availability': avg_availability,
                    'Performance': avg_performance,
                    'Quality': avg_quality,
                    'Location': location  # Dodaj kolumnę lokalizacji
                })
        except Exception as e:
            st.write(f"Error processing machine {machine}: {e}")
    return pd.DataFrame(oee_summary), all_filtered_out_data

# Funkcja do rysowania scatter plot z użyciem Plotly
def plot_scatter(oee_summary, y_axis):
    # Zamień wszystkie wartości na procenty
    oee_summary['OEE'] = oee_summary['OEE']
    oee_summary[y_axis] = oee_summary[y_axis]

    fig = px.scatter(oee_summary, x='OEE', y=y_axis, hover_name='Machine',
                     title=f'OEE vs {y_axis}',
                     labels={'OEE': 'OEE (%)', y_axis: f'{y_axis} (%)'},
                     color='Location',  # Dodaj kolorowanie na podstawie lokalizacji
                     template='plotly_white')
    fig.update_traces(marker=dict(size=12))

    # Dodaj przerywane linie na 75% dla osi X i Y
    fig.add_vline(75,
        line=dict(
            color="LightSeaGreen",
            width=2,
            dash="dash",
        ),
    )
    fig.add_hline(75,
        line=dict(
            color="LightSeaGreen",
            width=2,
            dash="dash",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

# Funkcja do rysowania wykresu liniowego z użyciem Plotly
def plot_line_chart(oee_data):
    # Agregacja danych według Schichtdatum
    oee_data = oee_data[['Schichtdatum', 'Produzierte Menge', 'Gutmenge', 'Geplante Menge', 'Name (Artikel)']]
    daily_data = oee_data.groupby('Schichtdatum').sum().reset_index()
    
    fig = go.Figure()
    
    # Dodanie wykresu dla Produzierte Menge
    fig.add_trace(go.Scatter(
        x=daily_data['Schichtdatum'], 
        y=daily_data['Produzierte Menge'], 
        mode='lines+markers', 
        name='Produzierte Menge',
        line=dict(color='#a87531')
    ))

    # Dodanie wykresu dla Gutmenge
    fig.add_trace(go.Scatter(
        x=daily_data['Schichtdatum'], 
        y=daily_data['Gutmenge'], 
        mode='lines+markers', 
        name='Gutmenge',
        line=dict(color='#32a852')
    ))

    # Dodanie wykresu dla Geplante Menge
    fig.add_trace(go.Scatter(
        x=daily_data['Schichtdatum'], 
        y=daily_data['Geplante Menge'], 
        mode='lines+markers', 
        name='Geplante Menge',
        line=dict(color='#3677c2')
    ))

    # Dodanie pionowych linii przerywanych dla momentów zmiany części
    change_dates = oee_data[oee_data['Name (Artikel)'].shift() != oee_data['Name (Artikel)']]['Schichtdatum']
    for change_date in change_dates:
        fig.add_vline(x=change_date, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(title='Production Data Over Time',
                      yaxis_title='Quantity',
                      template='plotly_white',
                      legend=dict(
                      orientation="h",
                      yanchor="bottom",
                      y=1.02,
                      xanchor="center",
                      x=0.5
                    ))
    
    st.plotly_chart(fig, use_container_width=True)

# Funkcja do rysowania wykresu liniowego dla kolumny 'OEE' z użyciem Plotly
def plot_oee_chart(oee_data):
    oee_data = oee_data[['Schichtdatum', 'OEE']]

    # Przeliczenie wartości OEE na procenty
    oee_data['OEE'] = oee_data['OEE'] * 100

    # Agregacja danych według Schichtdatum
    daily_oee = oee_data.groupby('Schichtdatum').mean().reset_index()
    
    mean_oee = round(oee_data['OEE'].mean(), 2)  # Przeliczenie średniej wartości OEE na procenty

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_oee['Schichtdatum'], 
        y=daily_oee['OEE'], 
        mode='lines+markers', 
        name='OEE'
    ))

    # Dodanie linii odniesienia y = 75
    fig.add_hline(
        y=75, 
        line_dash="dash", 
        line_color="#9e3a2c", 
        annotation_text="Target OEE = 75%",
        annotation_position="bottom left"
    )
    
    # Dodanie linii odniesienia mean_oee
    fig.add_hline(
        y=mean_oee, 
        line_dash="dash", 
        line_color="#32a852", 
        annotation_text=f"Mean OEE = {mean_oee}%",
        annotation_position="bottom right"
    )

    fig.update_layout(
        title='OEE Over Time',
        yaxis_title='OEE [%]',
        template='plotly_white',
        yaxis=dict(range=[0, 120]),  # Ustawienie granic osi Y
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_downtime_percentage_bar_chart(oee_data):

    col1, col2 = st.columns([3, 1])

    with col1:
        # Konwersja czasu na minuty
        time_columns = ['Stillstandszeit (geplant)', 'Stillstandszeit (ungeplant)', 
                        'Stillstandszeit (ohne Einfluss)', 'Stillstandszeit (ohne Grund)', 
                        'Hauptnutzungszeit']
        
        for col in time_columns:
            oee_data[col] = oee_data[col].apply(convert_to_minutes)
        
        # Agregacja danych według Schichtdatum
        downtime_data = oee_data.groupby('Schichtdatum')[time_columns].sum().reset_index()

        # Obliczanie procentowego udziału czasów
        downtime_data['Total'] = downtime_data[time_columns].sum(axis=1)
        for col in time_columns:
            downtime_data[col + ' (%)'] = (downtime_data[col] / downtime_data['Total']) * 100
        
        # Usunięcie wierszy zawierających NaN
        downtime_data = downtime_data.dropna()

        # Definiowanie kolorów
        colors = {
            'Stillstandszeit (geplant)': '#298487',
            'Stillstandszeit (ungeplant)': '#c49c35',
            'Stillstandszeit (ohne Einfluss)': '#3dd9b2',
            'Stillstandszeit (ohne Grund)': '#c73c32',
            'Hauptnutzungszeit': '#3bc449'
        }

        # Tworzenie wykresu słupkowego skumulowanego z danymi procentowymi
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=downtime_data['Schichtdatum'], 
            y=downtime_data['Stillstandszeit (geplant) (%)'], 
            name='Stillstandszeit (geplant)',
            marker_color=colors['Stillstandszeit (geplant)']
        ))

        fig.add_trace(go.Bar(
            x=downtime_data['Schichtdatum'], 
            y=downtime_data['Stillstandszeit (ungeplant) (%)'], 
            name='Stillstandszeit (ungeplant)',
            marker_color=colors['Stillstandszeit (ungeplant)']
        ))

        fig.add_trace(go.Bar(
            x=downtime_data['Schichtdatum'], 
            y=downtime_data['Stillstandszeit (ohne Einfluss) (%)'], 
            name='Stillstandszeit (ohne Einfluss)',
            marker_color=colors['Stillstandszeit (ohne Einfluss)']
        ))

        fig.add_trace(go.Bar(
            x=downtime_data['Schichtdatum'], 
            y=downtime_data['Stillstandszeit (ohne Grund) (%)'], 
            name='Stillstandszeit (ohne Grund)',
            marker_color=colors['Stillstandszeit (ohne Grund)']
        ))

        fig.add_trace(go.Bar(
            x=downtime_data['Schichtdatum'], 
            y=downtime_data['Hauptnutzungszeit (%)'], 
            name='Hauptnutzungszeit',
            marker_color=colors['Hauptnutzungszeit']
        ))

        # Layout wykresu
        fig.update_layout(
            barmode='stack',
            title='Downtime and Working Time by Day',
            yaxis_title='Percentage [%]',
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Obliczanie sumarycznych czasów dla całego miesiąca
        total_downtime = downtime_data[time_columns].sum()
        total_time = total_downtime.sum()
        percentage_summary = (total_downtime / total_time) * 100

        pie_fig = go.Figure(go.Pie(
            labels=percentage_summary.index,
            values=percentage_summary.values,
            marker=dict(colors=[colors[col] for col in time_columns]),
            showlegend=False,  # Ukryj legendę
            textinfo='percent'  # Wyświetl etykiety i procenty bezpośrednio na wykresie
        ))

        pie_fig.update_layout(
            title='Summary of Downtime and Working Time',
            template='plotly_white'
        )

        st.plotly_chart(pie_fig, use_container_width=True)























# Function to load and display image using base64 encoding
def get_image_as_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Path to the logo image file
logo_path = "logo.jpg"
logo_base64 = get_image_as_base64(logo_path)

# Streamlit UI
st.divider()

# Centering the logo using HTML and CSS with base64 encoded image
st.markdown(
    f"""
    <div style='text-align: center;'>
        <img src='data:image/jpg;base64,{logo_base64}' width='400'>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h1 style='text-align: center; color: #2F5A97;'>MACHINES</h1>",
    unsafe_allow_html=True
)
st.divider()

col1, col2 = st.columns([2, 1])

with st.expander('LOAD MACHINE DATA'):

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        - **Machine Analysis:** Browse and analyze production data for selected machines.
        - **Downtime Analysis:** Analyze and visualize machine downtimes and working times.
        - **OEE Analysis:** Calculate and visualize OEE metrics for all machines.
        - **System Errors:** Review and analyze filtered data points that do not meet the specified conditions.
        """)
    
    with col2:
        encoding_sep_options = st.selectbox('Wybierz kodowanie', options=['iso-8859-1', 'UTF-16'])
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", label_visibility='collapsed')

st.divider()

if uploaded_file is not None:
    if encoding_sep_options == 'UTF-16':
        data = load_data(uploaded_file, 'utf-16', '\t')
    elif encoding_sep_options == 'iso-8859-1':
        data = load_data(uploaded_file, 'iso-8859-1', ';')
    
    col1, col2 = st.columns([1, 3])

    with col1:
        # Sekcja wyboru numeru maszyny lub wpisania ręcznie
        selection_method = st.radio("Select search method", 
                                    ("Choose machine from list", 
                                    "Choose part from list"),
                                    horizontal=False)
        
        machine_name = None  # Inicjalizacja zmiennej machine_name
        part_name = None

    with col2:

        if selection_method == "Choose machine from list":
            unique_matching_machines = data['Name (Allgemeine Informationen)'].unique()
            machine_name = st.selectbox("Choose a machine from the list", unique_matching_machines)
            st.subheader(f"Machine selected: {machine_name}")

        elif selection_method == "Choose part from list":
            unique_matching_parts = data['Name (Auftrag)'].unique()
            part_name = st.selectbox("Choose a part from a list", unique_matching_parts)
            st.subheader(f"Part selected {part_name}")
    
    st.divider()
    
    if machine_name or part_name:

        if selection_method == "Choose part from list":
            filtered_machine_data = data[data['Name (Auftrag)'] == part_name]
            filtered_machine_data = filtered_machine_data.sort_values(['Schichtdatum', 'Schichtnummer', 'Beginn'])
        
        else:
            filtered_machine_data = data[data['Name (Allgemeine Informationen)'] == machine_name]
            filtered_machine_data = filtered_machine_data.sort_values(['Schichtdatum', 'Schichtnummer', 'Beginn'])
        
        oee_data, filtered_out_data = calculate_oee(filtered_machine_data)
        
        if oee_data.empty:
            st.write('There is no data to show.')
        else:
            with st.expander('SHOW MACHINE DATA'):
                st.dataframe(oee_data, use_container_width=True, hide_index=True)

            with st.expander('SHOW RESULTS'):

                col1, col2 = st.columns(2)

                with col1:
                    plot_line_chart(oee_data)
                
                with col2:
                    plot_oee_chart(oee_data)
                
                plot_downtime_percentage_bar_chart(filtered_machine_data)
            
            
    st.divider()
            

    # Obliczanie i wyświetlanie OEE dla wszystkich maszyn
    with st.expander("OEE SUMMARY"):

        col1, col2 = st.columns([1, 2])

        with col1:
            oee_summary, filtered_out = calculate_oee_for_all_machines(data)
            st.dataframe(oee_summary, use_container_width=True, height=500, hide_index=True)
        
        with col2:
            y_axis = st.selectbox("Select Y-axis for scatter plot", ["Availability", "Performance", "Quality"])
            plot_scatter(oee_summary, y_axis)

    with st.expander('SYSTEM ERRORS'):

        col1,col2 = st.columns([1, 1])
            
        with col1:
            filtered_out['Produzierte Menge'] = filtered_out['Produzierte Menge'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            filtered_out['Produzierte Menge'] = pd.to_numeric(filtered_out['Produzierte Menge'], errors='coerce').fillna(0)

            filtered_out['Gutmenge'] = filtered_out['Gutmenge'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            filtered_out['Gutmenge'] = pd.to_numeric(filtered_out['Gutmenge'], errors='coerce').fillna(0)

            filtered_out['Geplante Menge'] = filtered_out['Geplante Menge'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            filtered_out['Geplante Menge'] = pd.to_numeric(filtered_out['Geplante Menge'], errors='coerce').fillna(0).round()

            st.dataframe(filtered_out[['Schichtdatum', 'Schichtnummer',
                                      'Produzierte Menge', 'Gutmenge',
                                      'Geplante Menge', 'Hauptnutzungszeit',
                                      'Betriebszeit', 'Name (Artikel)',
                                      'Name (Aktueller Standort)']], 
                                      use_container_width=True, 
                                      hide_index=True,
                                      height=500)
            
        with col2:
            # Ensure the relevant columns are of numeric type
            filtered_out['Betriebszeit'] = pd.to_numeric(filtered_out['Betriebszeit'], errors='coerce').fillna(0)
            filtered_out['Hauptnutzungszeit'] = pd.to_numeric(filtered_out['Hauptnutzungszeit'], errors='coerce').fillna(0)
            filtered_out['Gutmenge'] = pd.to_numeric(filtered_out['Gutmenge'], errors='coerce').fillna(0)

            # Split the filtered out data into two based on the conditions
            condition1 = filtered_out['Betriebszeit'] < filtered_out['Hauptnutzungszeit']
            condition2 = filtered_out['Gutmenge'] < 0

            # Create a combined DataFrame with an additional column for the condition
            filtered_out['Condition'] = 'Condition 1'
            filtered_out.loc[condition2, 'Condition'] = 'Condition 2'

            # Count occurrences for each department and condition
            department_counts = filtered_out.groupby(['Name (Aktueller Standort)', 'Condition']).size().reset_index(name='Count')

            # Pivot the table to get a format suitable for a stacked barplot
            department_counts_pivot = department_counts.pivot(index='Name (Aktueller Standort)', columns='Condition', values='Count').fillna(0)
            department_counts_pivot = department_counts_pivot.reset_index()

            # Calculate total occurrences for sorting
            department_counts_pivot['Total'] = department_counts_pivot['Condition 1'] + department_counts_pivot['Condition 2']

            # Sort the DataFrame by total occurrences
            department_counts_pivot = department_counts_pivot.sort_values(by='Total', ascending=True)

            # Create the stacked barplot using Plotly
            fig = go.Figure(data=[
                go.Bar(name='Betriebszeit < Hauptnutzungszeit', y=department_counts_pivot['Name (Aktueller Standort)'], x=department_counts_pivot['Condition 1'], orientation='h', marker=dict(color='rgba(246, 78, 139, 0.6)')),
                go.Bar(name='Gutmenge < 0', y=department_counts_pivot['Name (Aktueller Standort)'], x=department_counts_pivot['Condition 2'], orientation='h', marker=dict(color='rgba(58, 71, 80, 0.6)'))
            ])

            # Update the layout for stacked barplot
            fig.update_layout(barmode='stack', 
                              xaxis_title='Number of Occurrences', 
                              title='Number of Occurrences per Department')

            # Display the plot in Streamlit
            st.plotly_chart(fig, use_container_width=True)

