import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import requests
import polyline
import folium
import math
from geopy.geocoders import Nominatim
from datetime import datetime
from streamlit_lottie import st_lottie
from folium.plugins import HeatMap
from streamlit_image_select import image_select
import random
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey


# Set up page config
st.set_page_config(page_title="Travel Route Dashboard", layout="wide", page_icon="🚗")


# Create a state mapping dictionary
state_mapping = {
        "Alabama": "AL",
        "Alaska": "AK",
        "Arizona": "AZ",
        "Arkansas": "AR",
        "California": "CA",
        "Colorado": "CO",
        "Connecticut": "CT",
        "Delaware": "DE",
        "Florida": "FL",
        "Georgia": "GA",
        "Hawaii": "HI",
        "Idaho": "ID",
        "Illinois": "IL",
        "Indiana": "IN",
        "Iowa": "IA",
        "Kansas": "KS",
        "Kentucky": "KY",
        "Louisiana": "LA",
        "Maine": "ME",
        "Maryland": "MD",
        "Massachusetts": "MA",
        "Michigan": "MI",
        "Minnesota": "MN",
        "Mississippi": "MS",
        "Missouri": "MO",
        "Montana": "MT",
        "Nebraska": "NE",
        "Nevada": "NV",
        "New Hampshire": "NH",
        "New Jersey": "NJ",
        "New Mexico": "NM",
        "New York": "NY",
        "North Carolina": "NC",
        "North Dakota": "ND",
        "Ohio": "OH",
        "Oklahoma": "OK",
        "Oregon": "OR",
        "Pennsylvania": "PA",
        "Rhode Island": "RI",
        "South Carolina": "SC",
        "South Dakota": "SD",
        "Tennessee": "TN",
        "Texas": "TX",
        "Utah": "UT",
        "Vermont": "VT",
        "Virginia": "VA",
        "Washington": "WA",
        "West Virginia": "WV",
        "Wisconsin": "WI",
        "Wyoming": "WY",
    }

@st.cache_data
def get_states_on_route(coordinates):
    # Using Nominatim for reverse geocoding
    geolocator = Nominatim(user_agent="my_route_app")
    states = []
    
    # Process coordinates with larger step size to avoid API limits
    step = max(1, len(coordinates) // 30)

    
    for coord in coordinates[::step]:
        try:
            time.sleep(1)  # Respect API rate limits
            location = geolocator.reverse(f"{coord[0]}, {coord[1]}")
            
            if location and location.raw.get('address'):
                state = location.raw['address'].get('state')
                if state:
                    # Convert full state name to abbreviation
                    state_abbrev = state_mapping.get(state, state)
                    if state_abbrev and state_abbrev not in states:
                        states.append(state_abbrev)
                        
        except Exception as e:
            print(f"Error processing coordinates {coord}: {str(e)}")
            continue
    
    return states


def get_safety_index(states):
    # Read the CSV file
    safety_data = pd.read_csv("data/data-wwXo2.csv")
    
    # Create dictionary with state abbreviations as keys
    safety_dict = {state_mapping[state]: score 
                        for state, score in zip(safety_data['State'], safety_data['Safety Score'])}

    # Return only the safety scores for the states in the route
    return {state: safety_dict.get(state, 0) for state in states}

# Function to load Lottie animations
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie animation URLs

# Load the data with error handling
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("./data/enriched_travel_routes.csv")
        # Check if necessary columns are present
        required_columns = ['from_city', 'to_city', 'start_lat', 'start_lng', 'end_lat', 'end_lng', 'mode', 'cost', 'time', 'distance', 'line_color']
        for column in required_columns:
            if column not in data.columns:
                raise KeyError(f"Missing required column: {column}")
        # Clean up mode values for consistency
        data['mode'] = data['mode'].str.strip()
    except FileNotFoundError:
        st.error("Data files not found. Please upload the required data.")
        return None
    except KeyError as e:
        st.error(f"Data file is missing required columns: {e}")
        return None
    return data

# Get current weather information based on coordinates
@st.cache_data
def get_weather_by_coords(lat, lon):
    api_key = "bd5e378503939ddaee76f12ad7a97608"  # Replace with your actual API key
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric"
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Unable to retrieve weather data for the selected destination.")
        return None

data = load_data()
if data is None:
    st.stop()

def calculate_distance(coord1, coord2):
    # Implement distance calculation (e.g., using haversine formula)
    # For simplicity, we'll use a placeholder function
    return ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)**0.5 * 111  # Rough approximation in km

@st.cache_data
def find_major_cities(route_points, min_distance=100, max_cities=10):
    geolocator = Nominatim(user_agent="my_app")
    major_cities = []
    last_city_coords = None
    step = max(1, len(route_points) // 20)  # Reduce the number of points to check

    for point in route_points[::step]:
        if len(major_cities) >= max_cities:
            break

        location = geolocator.reverse(f"{point[0]}, {point[1]}")
        if location and 'city' in location.raw['address']:
            city_name = location.raw['address']['city']
            if not last_city_coords or calculate_distance(last_city_coords, point) >= min_distance:
                major_cities.append({"name": city_name, "lat": point[0], "lng": point[1]})
                last_city_coords = point

    return major_cities

# Main page layout
st.title("Ways to Travel to Your Destination")
st.write("""
Compare flight, rail, and road options with key details like safety scores, scenic stops, average costs, and weather conditions. Discover popular attractions, explore eco-friendly travel choices by checking carbon emissions, and find everything you need for a smooth, enjoyable journey—all in one place!
""")

col1, col2 = st.columns([1, 2])

# Dropdown inputs for from and to locations
with col1:
    st.markdown("### Select Travel Options")
    from_city = st.selectbox("Select From City", list(data["from_city"].unique()))
    to_city = st.selectbox("Select To City", list(data["to_city"].unique()))

    # Filter data based on selection
    filtered_data = data.copy()
    filtered_data = filtered_data[(filtered_data["from_city"] == from_city) & (filtered_data["to_city"] == to_city)]

    # Check if filtered data is empty
    if filtered_data.empty:
        st.warning("No data available for the selected criteria. Please adjust your filters.")
        st.stop()

    # Rank routes by cost, time, and distance
    filtered_data['ranking_score'] = (filtered_data['cost'] * 0.4) + (filtered_data['distance'] * 0.3) + (filtered_data['time'] * 0.3)
    ranked_data = filtered_data.sort_values(by='ranking_score', ascending=True)

    # Display travel options as cards
    selected_mode = st.session_state.get('selected_mode', None)
    selected_data = st.session_state.get('selected_data', None)
    st.markdown("### Ways to Travel to {}".format(to_city))

    for index, row in ranked_data.iterrows():
        # if row['mode'] == 'train':
        #     continue
        icon = "✈️" if row['mode'] == 'flight' else "🚗"
        duration_hours = row['time'] // 3600
        duration_minutes = (row['time'] % 3600) // 60

        # Card layout for each travel option
        if st.button(f"{icon} {row['mode'].capitalize()} from {row['from_city']} to {row['to_city']} - Duration: {duration_hours}h {duration_minutes}min - Cost: ${row['cost']}", key=f"button_{index}"):
            selected_mode = row['mode']
            selected_data = row
            st.session_state['selected_mode'] = selected_mode
            st.session_state['selected_data'] = selected_data
    
    
    if selected_mode:
        # Carbon Emission Index Visualization
        def plot_carbon_emission_index_pie():
            # Generate sample travel mode data
            travel_modes = pd.DataFrame({
                'Mode': ['Car', 'Flight', 'Bus', 'Train'],
                'Cost ($)': [120, 200, 60, 90],
                'Duration (hours)': [6, 1.5, 8, 5],
                'CO2 Emissions (kg)': [100, 150, 50, 80]
            })

            # Calculate a carbon emission index: CO₂ emissions per hour
            travel_modes['Carbon Emission Index'] = travel_modes['CO2 Emissions (kg)'] / travel_modes['Duration (hours)']

            # Add some explanatory text to make the chart more informative
            st.write("""
            **Carbon Emission Index** represents the CO₂ emissions per hour of travel. 
            A higher index means that the travel mode emits more CO₂ per hour, 
            indicating a potentially less eco-friendly choice. This chart shows the 
            proportion of each mode's Carbon Emission Index relative to the total.
            """)

            # Create a pie chart for the Carbon Emission Index
            fig_index_pie = px.pie(
                travel_modes,
                names='Mode',
                values='Carbon Emission Index',
                title='Carbon Emission Index Distribution (CO₂ Emissions per Hour)',
                hole=0.3
            )

            # Update the pie chart to show more details
            fig_index_pie.update_traces(
                textinfo='label+percent+value', 
                textfont_size=14,
                hovertemplate='Mode: %{label}<br>CO₂ per hour: %{value:.2f} kg<br>%{percent}',
                marker=dict(line=dict(color='#000000', width=1))
            )

            # Display the chart in Streamlit
            st.plotly_chart(fig_index_pie, use_container_width=True)

        st.title("Carbon Emission Index Visualization")
        plot_carbon_emission_index_pie()



# Display map with selected route
with col2:
    if selected_mode:
        st.markdown("### Route Map for Selected Mode")

        mode_data = filtered_data[filtered_data['mode'] == selected_mode]
        
        # Ensure coordinates are numeric
        mode_data = mode_data.dropna(subset=["start_lat", "start_lng", "end_lat", "end_lng"])
        mode_data["start_lat"] = pd.to_numeric(mode_data["start_lat"], errors='coerce')
        mode_data["start_lng"] = pd.to_numeric(mode_data["start_lng"], errors='coerce')
        mode_data["end_lat"] = pd.to_numeric(mode_data["end_lat"], errors='coerce')
        mode_data["end_lng"] = pd.to_numeric(mode_data["end_lng"])
        mode_data = mode_data.dropna(subset=["start_lat", "start_lng", "end_lat", "end_lng"])

        # Add a checkbox to toggle weather info
        show_weather = st.checkbox("Show Weather Information on Map", value=False, key="show_weather_checkbox")

        # Incorporate GraphHopper API for road routes
        if selected_mode == "road":
            # Extract coordinates
            start_lng = mode_data.iloc[0]['start_lng']
            start_lat = mode_data.iloc[0]['start_lat']
            end_lng = mode_data.iloc[0]['end_lng']
            end_lat = mode_data.iloc[0]['end_lat']

            # GraphHopper API request
            url = "https://graphhopper.com/api/1/route"
            query = {
                "key": "7d1691c3-6102-459c-a504-26635d40b2c4",
                "points_encoded": "true"
            }
            payload = {
                "points": [
                    [start_lng, start_lat],
                    [end_lng, end_lat]
                ]
            }
            headers = {"Content-Type": "application/json"}

            response = requests.post(url, json=payload, headers=headers, params=query)

            if response.status_code == 200:
                data = response.json()
                # Get route points from the GraphHopper response
                route_points = polyline.decode(data['paths'][0]['points'])

                # Find major cities along the route
                major_cities = find_major_cities(route_points)

                if 'paths' in data and len(data['paths']) > 0 and 'points' in data['paths'][0]:
                    encoded_polyline = data['paths'][0]['points']
                    try:
                        coordinates = polyline.decode(encoded_polyline)
                        route_map = folium.Map(
                            location=[(start_lat + end_lat) / 2, (start_lng + end_lng) / 2],
                            zoom_start=6,
                        )
                        
                        # Add the route as a polyline
                        folium.PolyLine(
                            locations=coordinates,
                            color="blue",
                            weight=5,
                            opacity=0.8,
                        ).add_to(route_map)
                        
                        # Add start and end markers with special icons
                        folium.Marker(
                            location=[start_lat, start_lng],
                            popup="<b>Start</b>",
                            icon=folium.Icon(color="green", icon="glyphicon-play")
                        ).add_to(route_map)
                        folium.Marker(
                            location=[end_lat, end_lng],
                            popup="<b>End</b>",
                            icon=folium.Icon(color="red", icon="glyphicon-stop")
                        ).add_to(route_map)
                        
                        # Add markers for important cities
                        for city in major_cities:
                            folium.Marker(
                                location=[city['lat'], city['lng']],
                                popup=f"<b>{city['name']}</b>",
                                icon=folium.Icon(color="orange", icon="info-sign")
                            ).add_to(route_map)
                        
                        # Always show route and major cities, add weather only if checkbox is selected
                        if show_weather:
                            # Add weather information for start, major cities, and end point
                            heat_map_data = []
                            # Add weather data for the start point
                            weather_data_start = get_weather_by_coords(start_lat, start_lng)
                            if weather_data_start:
                                weather_icon = "cloud" if "cloud" in weather_data_start['weather'][0]['description'].lower() else "sun" if "clear" in weather_data_start['weather'][0]['description'].lower() else "cloud-rain"
                                folium.Marker(
                                    location=[start_lat, start_lng],
                                    popup=f"<b>Start</b><br>Weather: {weather_data_start['weather'][0]['description'].capitalize()}<br>Temperature: {weather_data_start['main']['temp']} °C<br>Humidity: {weather_data_start['main']['humidity']}%<br>Pressure: {weather_data_start['main']['pressure']} hPa",
                                    icon=folium.Icon(color="blue", icon=weather_icon)
                                ).add_to(route_map)
                                heat_map_data.append([start_lat, start_lng, weather_data_start['main']['temp']])

                            # Add weather data for major cities along the route
                            for city in major_cities:
                                # Get weather data for each stop
                                weather_data = get_weather_by_coords(city['lat'], city['lng'])
                                if weather_data:
                                    weather_icon = "cloud" if "cloud" in weather_data['weather'][0]['description'].lower() else "sun" if "clear" in weather_data['weather'][0]['description'].lower() else "cloud-rain"
                                    folium.Marker(
                                        location=[city['lat'], city['lng']],
                                        popup=f"<b>{city['name']}</b><br>Weather: {weather_data['weather'][0]['description'].capitalize()}<br>Temperature: {weather_data['main']['temp']} °C<br>Humidity: {weather_data['main']['humidity']}%<br>Pressure: {weather_data['main']['pressure']} hPa",
                                        icon=folium.Icon(color="blue", icon=weather_icon)
                                    ).add_to(route_map)
                                    heat_map_data.append([city['lat'], city['lng'], weather_data['main']['temp']])
                        
                            # Add weather information for the destination
                            weather_data_end = get_weather_by_coords(end_lat, end_lng)
                            if weather_data_end:
                                weather_icon = "cloud" if "cloud" in weather_data_end['weather'][0]['description'].lower() else "sun" if "clear" in weather_data_end['weather'][0]['description'].lower() else "cloud-rain"
                                folium.Marker(
                                    location=[end_lat, end_lng],
                                    popup=f"<b>End</b><br>Weather: {weather_data_end['weather'][0]['description'].capitalize()}<br>Temperature: {weather_data_end['main']['temp']} °C<br>Humidity: {weather_data_end['main']['humidity']}%<br>Pressure: {weather_data_end['main']['pressure']} hPa",
                                    icon=folium.Icon(color="blue", icon=weather_icon)
                                ).add_to(route_map)
                                heat_map_data.append([end_lat, end_lng, weather_data_end['main']['temp']])
                        
                            # Add a heat map layer to visualize temperature
                            if heat_map_data:
                                HeatMap(heat_map_data, radius=15, blur=10, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}).add_to(route_map)
                        
                        # Display the map in Streamlit
                        st.markdown("### Route Visualization Map")
                        st.write("The road route map visualizes optimal paths, highlighting popular cities along the way where you can stop, explore, and enjoy local attractions. It also provides real-time weather information for each city, helping you plan your stops and travel experience better.")
                        st.components.v1.html(route_map._repr_html_(), height=500)

                        # Get states through which the route passes
                        states_on_route = get_states_on_route(coordinates)
                        
                        # Get safety index for these states
                        safety_data = get_safety_index(states_on_route)

                        
                        st.markdown("### Safety Score of states along the route")
                        st.write("The map displays the safety scores of states along the route, giving travelers valuable insights into the safety levels of different areas.")
                        # Create the choropleth map
                        fig = go.Figure(go.Choropleth(
                            locations=list(safety_data.keys()),
                            z=list(safety_data.values()),
                            locationmode='USA-states',
                            colorscale='RdYlGn',  # Red to Yellow to Green scale
                            colorbar=dict(
                                title=dict(
                                    text="Safety Score (%)",
                                    font=dict(
                                        size=14,  # Increased title font size
                                        family="Arial",
                                        color="black"
                                    ),
                                ),
                                xpad=10,
                                tickfont=dict(
                                    size=14  # Optional: adjust tick label size if needed
                                )
                            ),
                            text=[f"State: {state}<br>Safety Score: {safety_data[state]:.1f}" for state in safety_data.keys()],
                            hoverinfo='text',
                            hovertemplate="<b>%{text}</b><extra></extra>",
                            hoverlabel=dict(
                                bgcolor="white",
                                font_size=16,
                                font_family="Arial",
                                bordercolor='black'
                            ),
                            zmin=100,
                            zmax=0
                        ))
                        
                        fig.add_trace(go.Scattergeo(
                            locationmode='USA-states',
                            locations=list(safety_data.keys()),  # State abbreviations
                            text=list(safety_data.keys()),  # State names (use full names if needed)
                            mode='text',  # Only display text
                            textfont=dict(
                                size=14,
                                color='black'
                            ),
                            hoverinfo='skip',
                            showlegend=False
                        ))

                        # Update layout
                        fig.update_layout(
                            # title_text='Safety Index of States Along Rouet',
                            geo=dict(
                                scope='usa',
                                showlakes=True,
                                lakecolor='rgb(255, 255, 255)',
                                showland=True,
                                landcolor='rgb(217, 217, 217)',
                            ),
                            height=600,
                            margin=dict(l=0, r=0, t=30, b=0)
                        )
                        # Display the map
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error decoding polyline: {e}")
                else:
                    st.error("Invalid API response. 'paths' or 'points' key is missing.")
            else:
                st.error(f"Failed to retrieve route data. Status Code: {response.status_code}")

        # Incorporate Folium for flight routes with arc
        elif selected_mode == "flight":
            # Extract coordinates
            start_lng = mode_data.iloc[0]['start_lng']
            start_lat = mode_data.iloc[0]['start_lat']
            end_lng = mode_data.iloc[0]['end_lng']
            end_lat = mode_data.iloc[0]['end_lat']

            # Function to calculate intermediate points for a curved arc
            def generate_arc_points(start, end, num_points=100):
                lat1, lon1 = start
                lat2, lon2 = end

                points = []
                for i in range(num_points + 1):
                    f = i / num_points
                    delta_lat = lat2 - lat1
                    delta_lon = lon2 - lon1
                    lat = lat1 + f * delta_lat
                    lon = lon1 + f * delta_lon + (math.sin(f * math.pi) * 0.1)  # Adding curvature
                    points.append((lat, lon))
                return points

            # Generate arc points for the flight route
            arc_coordinates = generate_arc_points((start_lat, start_lng), (end_lat, end_lng))

            # Create a folium map centered on the route
            route_map = folium.Map(
                location=[(start_lat + end_lat) / 2, (start_lng + end_lng) / 2],
                zoom_start=4,
            )

            # Add the route as a polyline with arc
            folium.PolyLine(
                locations=arc_coordinates,  # Curved arc for flight
                color="red",
                weight=3,
                opacity=0.8,
            ).add_to(route_map)

            # Add start and end markers with special icons
            folium.Marker(
                location=[start_lat, start_lng], popup="<b>Start</b>", icon=folium.Icon(color="green", icon="glyphicon-play")
            ).add_to(route_map)
            folium.Marker(
                location=[end_lat, end_lng], popup="<b>End</b>", icon=folium.Icon(color="red", icon="glyphicon-stop")
            ).add_to(route_map)

            # Add markers for important cities
            major_cities = find_major_cities(arc_coordinates)
            for city in major_cities:
                folium.Marker(
                    location=[city['lat'], city['lng']],
                    popup=f"<b>{city['name']}</b>",
                    icon=folium.Icon(color="orange", icon="info-sign")
                ).add_to(route_map)

            if show_weather:
                # Add weather information as interactive markers
                weather_data_start = get_weather_by_coords(start_lat, start_lng)
                if weather_data_start:
                    weather_icon = "cloud" if "cloud" in weather_data_start['weather'][0]['description'].lower() else "sun" if "clear" in weather_data_start['weather'][0]['description'].lower() else "cloud-rain"
                    folium.Marker(
                        location=[start_lat, start_lng],
                        popup=f"<b>Start</b><br>Weather: {weather_data_start['weather'][0]['description'].capitalize()}<br>Temperature: {weather_data_start['main']['temp']} °C<br>Humidity: {weather_data_start['main']['humidity']}%<br>Pressure: {weather_data_start['main']['pressure']} hPa",
                        icon=folium.Icon(color="blue", icon=weather_icon)
                    ).add_to(route_map)
                
                weather_data_end = get_weather_by_coords(end_lat, end_lng)
                if weather_data_end:
                    weather_icon = "cloud" if "cloud" in weather_data_end['weather'][0]['description'].lower() else "sun" if "clear" in weather_data_end['weather'][0]['description'].lower() else "cloud-rain"
                    folium.Marker(
                        location=[end_lat, end_lng],
                        popup=f"<b>End</b><br>Weather: {weather_data_end['weather'][0]['description'].capitalize()}<br>Temperature: {weather_data_end['main']['temp']} °C<br>Humidity: {weather_data_end['main']['humidity']}%<br>Pressure: {weather_data_end['main']['pressure']} hPa",
                        icon=folium.Icon(color="blue", icon=weather_icon)
                    ).add_to(route_map)
                    
                    # Add a heat map layer to visualize temperature
                    heat_map_data = [[start_lat, start_lng, weather_data_start['main']['temp']], [end_lat, end_lng, weather_data_end['main']['temp']]]
                    HeatMap(heat_map_data, radius=15, blur=10, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}).add_to(route_map)
            
            # Display the map in Streamlit
            st.markdown("### Flight Route Visualization Map")
            st.write("The map visualizes flight routes with layovers between cities.")
            st.components.v1.html(route_map._repr_html_(), height=500)

            

        else:
            # Map visualization for non-road and non-flight routes
            layer = pdk.Layer(
                "LineLayer",
                data=mode_data,
                get_source_position=["start_lng", "start_lat"],
                get_target_position=["end_lng", "end_lat"],
                get_color="[200, 30, 0, 160]",
                get_width=5,
            )
            view_state = pdk.ViewState(
                latitude=mode_data["start_lat"].mean(),
                longitude=mode_data["start_lng"].mean(),
                zoom=5,
                pitch=0,
            )
            r = pdk.Deck(layers=[layer], initial_view_state=view_state,
                         tooltip={"text": "{from_city} to {to_city}\nMode: {mode}\nCost: ${cost}\nDistance: {distance} miles\nTime: {time} seconds"})
            st.pydeck_chart(r)
        # Photo previews and 360° views of scenic stops
        st.markdown("### Scenic Stops Along the Way")
        st.write("Scenic stops along the way offer beautiful and interesting locations to visit during your journey, providing opportunities to explore attractions, relax, and enjoy the landscape before continuing to your destination.")

        # Sample data for scenic stops (replace with your actual data)
        scenic_stops = {
            "Los Angeles": [
            {"name": "Chicago river walk", "image": "images/chicago_riverwalk.jpg", "panorama": "https://www.google.com/maps/embed?pb=!4v1733174017731!6m8!1m7!1sCAoSLEFGMVFpcE9WMDN3RWV6SGoybUhDMklETEs5STA4UE5NTy1VaFZCRlV6UWl2!2m2!1d41.88733743772273!2d-87.63478439783088!3f292.7229228917138!4f17.69280523725233!5f0.4000000000000002"},
            {"name": "Las Vegas", "image": "images/vegas.jpg", "panorama": "https://www.google.com/maps/embed?pb=!4v1733173182490!6m8!1m7!1sCAoSLEFGMVFpcE5yekdZWGtiVXBEdDd6cml4RmREYnhld1R5MzBXb2RCOGNVeFZf!2m2!1d36.11220550104704!2d-115.1745682198069!3f57.95407050288079!4f12.810681144433886!5f0.4000000000000002"},
            {"name": "Sloan's Lake park", "image": "images/sloan-lake-denver-co.jpg", "panorama": "https://www.google.com/maps/embed?pb=!4v1733171889438!6m8!1m7!1shbrs_9jnRWTCVjVE6B9TPQ!2m2!1d39.75264617864302!2d-105.045225873763!3f211.63222377856758!4f0.9387787829215029!5f1.9587109090973311"},],
            "Houston": [
            {"name": "Phoenix Camelback", "image": "images/phoenix.jpg", "panorama": "https://www.google.com/maps/embed?pb=!4v1733175015079!6m8!1m7!1sCAoSLEFGMVFpcE9JaGxLYWMtUmlrNnpZRG5pbVNDN2tlaFFlNGpwREFHY2J1bk9r!2m2!1d33.51008702896119!2d-111.9511173142467!3f144.68954511894327!4f-15.076732948727539!5f0.4000000000000002"},
            {"name": "San antonio", "image": "images/san_antonio.jpg", "panorama": "https://www.google.com/maps/embed?pb=!4v1733175274907!6m8!1m7!1sCAoSK0FGMVFpcE9oSVhSbHFaT1RjYnZYYVRhQnJsbTZzYXhuTFhwY0lDRGpMTDA.!2m2!1d29.46087286364939!2d-98.47716808259752!3f303.0645151300677!4f4.546095019335468!5f0.4000000000000002"},
            {"name": "Dripping springs", "image": "images/dripping_springs.jpg", "panorama": "https://www.google.com/maps/embed?pb=!4v1733175612136!6m8!1m7!1sCAoSLEFGMVFpcFBjc0d2Mmw1YnJsc1BSNGs1OGNjR2NlQXVyQzZsWk1GbW9sUnFQ!2m2!1d32.32768359066174!2d-106.5876748567992!3f295.9016391882129!4f19.093065352111253!5f0.7820865974627469"},],

        }


        # Image selection for photo previews
        selected_image = image_select(
            label="Select a scenic stop",
            images=[stop["image"] for stop in scenic_stops[to_city]],
            captions=[stop["name"] for stop in scenic_stops[to_city]],
            use_container_width=False
        )

        # Display the selected image
        if selected_image:
            selected_stop = next(stop for stop in scenic_stops[to_city] if stop["image"] == selected_image)

            st.markdown(
                f"""
                    <iframe
                    src="{selected_stop['panorama']}"
                    width="100%"
                    height="500"
                    frameborder="0"
                    style="border:0;"
                    allowfullscreen
                    aria-hidden="false"
                    tabindex="0">
                    </iframe>
                    """,
                    unsafe_allow_html=True,
                )

# Route Statistics Visualization (Overall Comparison for Selected Cities)
st.markdown("""
<div id="route-statistics" class="container mt-5">
    <h2 class="text-center" style="color: #4e73df;">Route Statistics for Selected Cities</h2>
</div>
""", unsafe_allow_html=True)
st.write("The chart displays the route statistics for selected cities, showing the average cost per mode of travel (flight, rail, and road). This visualization provides insights into how costs vary across different cities and travel routes, aiding in more informed travel planning.")

# Compare All Modes for Selected Cities
if not filtered_data.empty:
    # Bar Chart: Average Cost per Mode for Selected Cities
    st.write("#### Average Cost per Mode for Selected Cities")
    avg_cost_per_mode = filtered_data.groupby('mode')['cost'].mean().reset_index()
    fig_avg_cost = px.bar(avg_cost_per_mode, x='mode', y='cost', color='mode', title='Average Cost per Mode for Selected Cities', color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig_avg_cost, use_container_width=True)

    # # Scatter Plot: Distance vs Cost for All Modes for Selected Cities
    # st.write("#### Distance vs Cost for All Modes for Selected Cities")
    # fig_all_modes = px.scatter(filtered_data, x='distance', y='cost', color='mode', size='time',
    #                            labels={"distance": "Distance (miles)", "cost": "Cost ($)", "time": "Travel Time (seconds)"},
    #                            title="Distance vs Cost for All Modes for Selected Cities", color_discrete_sequence=px.colors.qualitative.Pastel)
    # st.plotly_chart(fig_all_modes, use_container_width=True)

# Footer with Bootstrap
st.markdown("""
<div class="container-fluid bg-dark text-white text-center mt-5 p-3 footer">
    <p>Travel Route Dashboard © 2024. Made with Streamlit, Bootstrap, and Plotly.</p>
</div>
""", unsafe_allow_html=True)

# Display weather information for the selected destination city
if selected_mode and 'mode_data' in locals() and not mode_data.empty:
    weather_data = get_weather_by_coords(mode_data.iloc[0]['end_lat'], mode_data.iloc[0]['end_lng'])
    if weather_data:
        st.markdown("### Current Weather in {}".format(to_city))
        col1, col2 = st.columns([1, 3])

        with col2:
            # Display main weather information with enhanced styling
            st.markdown(
                f"""<div style='font-size: 1.5em; font-weight: bold; color: #4e73df;'>
                    {weather_data['weather'][0]['description'].capitalize()}<br>
                    Temperature: <span style='color: #ff7f0e;'>{weather_data['main']['temp']} °C</span><br>
                    Humidity: <span style='color: #2ca02c;'>{weather_data['main']['humidity']}%</span><br>
                    Pressure: <span style='color: #d62728;'>{weather_data['main']['pressure']} hPa</span>
                </div>""",
                unsafe_allow_html=True
            )

        # Compact and visually rich interactive visualizations
        st.markdown("### Weather Data at a Glance")
        gauge_col1, gauge_col2 = st.columns([1, 1])

        with gauge_col1:
            # Temperature Gauge
            temp_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=weather_data['main']['temp'],
                title={'text': "Temperature (°C)"},
                delta={'reference': 20},  # Assuming 20°C as a baseline comfortable temperature
                gauge={'axis': {'range': [-10, 40]}, 'bar': {'color': "blue"}}
            ))
            st.plotly_chart(temp_fig, use_container_width=False)

        with gauge_col2:
            # Humidity Gauge
            humidity_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=weather_data['main']['humidity'],
                title={'text': "Humidity (%)"},
                delta={'reference': 50},  # Assuming 50% as baseline comfortable humidity
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "green"}}
            ))
            st.plotly_chart(humidity_fig, use_container_width=False)

        # with gauge_col3:
        #     # Pressure Gauge
        #     pressure_fig = go.Figure(go.Indicator(
        #         mode="gauge+number+delta",
        #         value=weather_data['main']['pressure'],
        #         title={'text': "Pressure (hPa)"},
        #         delta={'reference': 1013},  # Assuming 1013 hPa as baseline atmospheric pressure
        #         gauge={'axis': {'range': [900, 1100]}, 'bar': {'color': "orange"}}
        #     ))
        #     st.plotly_chart(pressure_fig, use_container_width=False)

        # # Interactive weather visualization using Plotly
        # st.markdown("### Interactive Weather Visualization")
        # weather_fig = go.Figure()
        # weather_fig.add_trace(go.Scatterpolar(
        #     r=[weather_data['main']['temp'], weather_data['main']['humidity'], weather_data['wind']['speed'], weather_data['main']['pressure']],
        #     theta=['Temperature', 'Humidity', 'Wind Speed', 'Pressure'],
        #     fill='toself',
        #     name='Current Weather',
        #     marker_color='rgba(106, 181, 135, 0.7)'
        # ))
        # weather_fig.update_layout(
        #     polar=dict(
        #         radialaxis=dict(visible=True, range=[0, max(weather_data['main']['temp'], 100)]),
        #         angularaxis=dict(direction='clockwise')
        #     ),
        #     showlegend=True
        # )
        # st.plotly_chart(weather_fig, use_container_width=True)

import pandas as pd
import random
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Enriched dummy data for popular cities visited
city_data = {
    'City': ['New York City', 'Las Vegas', 'Orlando', 'Chicago', 'Los Angeles', 'San Francisco', 'Seattle', 'Miami', 'Dallas', 'Boston'],
    'Rank': [i for i in range(1, 11)],
    'Description': [
        'The city that never sleeps, famous for Times Square and Central Park.',
        'The entertainment capital, known for its vibrant nightlife and casinos.',
        'Home to Disney World, a family-friendly vacation destination.',
        'Known for its architecture, museums, and deep-dish pizza.',
        'City of Angels, known for Hollywood and beautiful beaches.',
        'The Golden Gate City, famous for its iconic bridge and tech scene.',
        'The Emerald City, known for its coffee culture and Space Needle.',
        'A tropical paradise, known for its beaches and Latin-American culture.',
        'A bustling city known for its cultural diversity and skyline.',
        'A historic city, known for its rich American history and seafood.'
    ],
    'Visitors (in millions)': [65, 42, 75, 55, 50, 30, 25, 22, 18, 15],
    'Average Rating': [4.8, 4.6, 4.7, 4.5, 4.8, 4.7, 4.6, 4.5, 4.3, 4.4]
}

# Create DataFrame
df = pd.DataFrame(city_data)

# Function to draw the pyramid chart
def draw_pyramid():
    fig = go.Figure()
    colors = ['#4D77FF', '#4D9AFF', '#88C9FF', '#C5E4FF', '#6CA0DC', '#6EDB8E', '#F4D03F', '#E67E22', '#E74C3C', '#8E44AD']

    for index, row in df.iterrows():
        rank = row['Rank']
        label = row['City']
        description = row['Description']
        visitors = row['Visitors (in millions)']
        rating = row['Average Rating']
        hover_text = f"<b>{rank}. {label}</b><br>{description}<br>Visitors: {visitors}M<br>Rating: {rating}"

        fig.add_trace(go.Bar(
            y=[10 - rank],
            x=[2 + rank / 2],
            name=label,
            orientation='h',
            marker=dict(color=colors[index % len(colors)], line=dict(color='black', width=1)),
            hoverinfo='text',
            text=hover_text,
            textposition='inside',
            textfont=dict(family='Arial', size=14, color='white')
        ))

    fig.update_layout(
        title={
            'text': 'Pyramid of Popular Destination Cities',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=28, color='#333', family='Arial Black')
        },
        xaxis=dict(title='Popularity', showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        barmode='stack',
        showlegend=False,
        height=700,
        width=600,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='#f9f9f9',
        plot_bgcolor='#f9f9f9',
        annotations=[
            dict(
                text="<b>Hover over each bar to learn more about each city.</b>",
                x=0.5,
                y=-0.1,
                showarrow=False,
                xref="paper",
                yref="paper",
                align="center",
                font=dict(size=16, family='Arial', color='#4e73df')
            )
        ]
    )

    # Display the final version of the pyramid chart
    st.plotly_chart(fig, use_container_width=True, key='pyramid_chart')

# Function to draw additional visualizations for the selected city
def draw_city_info(city, chart_type):
    # Create some enriched dummy data for the selected city
    data = {
        'Attraction': ['Museum', 'Park', 'Landmark', 'Theater', 'Shopping District'],
        'Visitors (in millions)': [random.randint(1, 10) for _ in range(5)],
        'Popularity Score': [random.randint(50, 100) for _ in range(5)],
        'Ticket Price ($)': [random.randint(10, 50) for _ in range(5)],
        'Rating': [round(random.uniform(3.5, 5.0), 1) for _ in range(5)],
        'Category': ['Historical', 'Recreational', 'Cultural', 'Entertainment', 'Shopping']
    }
    city_df = pd.DataFrame(data)

    st.write(f"### Attractions in {city}")

    if chart_type == 'Bar Chart':
        fig = px.bar(city_df, x='Attraction', y='Visitors (in millions)', color='Popularity Score',
                     title=f'Major Attractions in {city}', labels={'Visitors (in millions)': 'Visitors (in Millions)'},
                     color_continuous_scale='Viridis', hover_data=['Ticket Price ($)', 'Rating'])
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == 'Bubble Chart':
        fig = px.scatter(city_df, x='Attraction', y='Visitors (in millions)',
                         size='Ticket Price ($)', color='Popularity Score',
                         title=f'Bubble Chart of Attractions in {city}', labels={'Visitors (in millions)': 'Visitors (in Millions)'},
                         color_continuous_scale='Viridis', hover_data=['Rating', 'Category'])
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == 'Radar Chart':
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=city_df['Popularity Score'],
            theta=city_df['Attraction'],
            fill='toself',
            name='Popularity Score'
        ))

        fig.add_trace(go.Scatterpolar(
            r=city_df['Visitors (in millions)'],
            theta=city_df['Attraction'],
            fill='toself',
            name='Visitors (in Millions)'
        ))

        fig.update_layout(
            title=f'Radar Chart of Attractions in {city}',
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == 'Treemap':
        fig = px.treemap(city_df, path=['Category', 'Attraction'], values='Visitors (in millions)',
                         color='Rating', hover_data=['Ticket Price ($)', 'Popularity Score'],
                         title=f'Treemap View of Attractions in {city}', color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

# Page Layout in Streamlit
st.title("Pyramid of Popular Destination Cities")
st.write("Explore the most popular US destination cities visited by travelers. Each city has its own charm and unique attractions.")

st.markdown("---")

# Draw the pyramid chart
draw_pyramid()

# Create a dropdown to select a city to see more details
selected_city = st.selectbox("Select a city to explore its attractions", options=df['City'].tolist(), index=0)

# Create a dropdown to select the type of chart to view
chart_type = st.radio("Select the type of chart to visualize the city's attractions:", 
                      options=['Bar Chart', 'Bubble Chart', 'Radar Chart', 'Treemap'], index=0)

# Draw additional visualization for the selected city
if selected_city:
    draw_city_info(selected_city, chart_type)



