from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
import streamlit as st
from matplotlib.patches import Polygon
import sys
from matplotlib.ticker import FixedLocator
from mplsoccer import PyPizza, FontManager, Radar
from unidecode import unidecode

#Code written by Adith George, dataset built using data from FBRef, inspired by work by Naveen Elliott, Liam Henshaw, and other football data analysts on LinkedIn
    
#radar chart creation
def radar(player, position, stats, title):
    #define params & respective and ranges
    params = stats  #list of stat names
    low = [0] * len(params)  #min val (next max) per stat for percentile calc
    high = [100] * len(params)

    # Player and Position Values
    player_values = [player[stat] * 100 for stat in stats]  # Player percentiles
    #position_values = [position[stat] * 100 for stat in stats]  # Positional average percentiles

    #create a Radar object
    radar = Radar(
        params=params,  # Parameter names
        min_range=low,  # Minimum value for each parameter
        max_range=high,  # Maximum value for each parameter
        num_rings=4,  # Number of concentric circles
        ring_width=1,  # Width of each ring
        center_circle_radius=1  # Radius of the center circle
    )

    #create the figure and axis for the radar chart
    fig, ax = radar.setup_axis()

    #plot the player's data
    radar_output = radar.draw_radar(
        player_values,
        ax=ax,
        kwargs_radar={'facecolor': 'blue', 'alpha': 0.5, 'edgecolor': 'blue', 'linewidth': 2}, #radar style
    )
    #add outlines
    max_radius = radar.ring_width * radar.num_rings  # Calculate the maximum radius
    for i in range(1, radar.num_rings + 2):
        radius = i  # Normalize radius based on number of rings
        circle = plt.Circle(
            (0, 0), radius,  # Set the center and radius
            transform=ax.transData._b,  # Use the correct transformation for the radar
            color="gray", linestyle="--", linewidth=1, fill=False
        )
        ax.add_artist(circle)
    #draw parameter (outer) labels
    radar.draw_param_labels(ax=ax, fontsize=14, color="black")

    # Draw range (inner) labels
    radar.draw_range_labels(ax=ax, fontsize=8, color="gray")

    # Add title
    fig.text(
        0.5, .9, title,
        ha="center", fontsize=30, color="black", weight="bold"
    )

    return fig
    
    
#pizza chart:
def pizza(player,player_2,stats,title, name1=None,name2=None):
    num = len(stats)
    angles=np.linspace(0,2*np.pi,num,endpoint=False)
    angles_mids=angles+(angles[1]/2)
    player_vals = [round(player[stat] *100) for stat in stats]
    if player_2:
        player_2_vals = [round(player_2[stat] *100) for stat in stats]
        #get params_offset (True if the absolute difference is <10)
        params_offset = [abs(p1 - p2) < 10 for p1, p2 in zip(player_vals, player_2_vals)]

        #create pie chart
        baker = PyPizza(
            params=stats,                  #list of params (stats)
            straight_line_color="#000000", #color for straight lines
            straight_line_lw=1,            #linewidth for straight lines
            last_circle_lw=1,              #linewidth of last circle
            other_circle_lw=1,             #linewidth for other circles
            other_circle_ls="-."           #linestyle for other circles
        )

        #plot pizza for player1 and 2
        fig, ax = baker.make_pizza(
            player_vals,                      #player 1 values
            compare_values=player_2_vals,    #player 2 values
            figsize=(11, 11),                   #adjust figsize
            param_location=110,               #where the parameter labels will be added
            kwargs_slices=dict(
                facecolor="yellow", edgecolor="#000000",
                zorder=2, linewidth=1
            ),                                #Player 1 slice appearance
            kwargs_compare=dict(
                facecolor="blue", edgecolor="#000000",
                zorder=2, linewidth=1
            ),                                #Player 2 slice appearance
            kwargs_params=dict(
                color="#000000", fontsize=16,
                va="center"
            ),                                #Param label appearance
            kwargs_values=dict(
                color="#000000", fontsize=13, zorder=3,
                bbox=dict(
                    edgecolor="#000000", facecolor="yellow",
                    boxstyle="round,pad=0.2", lw=1
                )
            ),                                #Player 1 values appearance
            kwargs_compare_values=dict(
                color="#000000", fontsize=14, zorder=3,
                bbox=dict(
                    edgecolor="#000000", facecolor="cornflowerblue",
                    boxstyle="round,pad=0.2", lw=1
                )
            )                                 #Player 2 values appearance
        )

        #close value text adjustment from mplsoccer
        baker.adjust_texts(params_offset, offset=-0.2, adj_comp_values=True)
        #add legend
        ax.legend(
            handles=[
                plt.Line2D([0], [0], color="yellow", lw=4, label=f"{name1}"),
                plt.Line2D([0], [0], color="lightblue", lw=4, label=f"{name2}")
            ],
            loc="lower left",
            bbox_to_anchor=(-0.2, -0.1),
            fontsize=14,
            frameon=False
        )
    else:
        #raw_values = [raw_player_values[stat] for stat in stats]
        #create plot
        baker = PyPizza(
            params=stats,                  
            straight_line_color="#000000", 
            straight_line_lw=1,            
            last_circle_lw=1,              
            other_circle_lw=1,             
            other_circle_ls="-."           
        )

        #plot pizza for player
        fig, ax = baker.make_pizza(
            player_vals,                 #player values
            figsize=(8, 8),                
            param_location=110,            
            kwargs_slices=dict(
                facecolor="yellow", edgecolor="#000000",
                zorder=2, linewidth=1
            ),                             #slices appearance
            kwargs_params=dict(
                color="#000000", fontsize=12,
                va="center"
            ),                             #param label appearance
            kwargs_values=dict(
                color="#000000", fontsize=12, zorder=3,
                bbox=dict(
                    edgecolor="#000000", facecolor="yellow",
                    boxstyle="round,pad=0.2", lw=1
                )
            )                              
        )

    #add title
    fig.text(
        0.515, 0.95, title, size=24,
        ha="center", color="#000000"
    )

    #add credits
    CREDIT_1 = "Program Written By Adith George; Data Taken From FBRef (StatsBomb)"
    CREDIT_2 = "Inspired by mplsoccer visualizations and work by Naveen Elliott & Liam Henshaw"

    fig.text(
        0.99, 0.005, f"{CREDIT_1}\n{CREDIT_2}", size=9,
        color="#000000", ha="right"
    )

    return fig

def highlight(row, player):
    styles = []
    for col in row.index:
        if col in player:
            if row[col] > player[col]:
                styles.append("background-color: lightgreen")
            elif row[col] < player[col]:
                styles.append("background-color: lightcoral")
            else:
                styles.append("")  # No change if equal
        else:
            styles.append("")  # No styling for non-stat columns
    return styles

#similarity calc
def similarity(df,name,position,stats,threshold, nation, team):
    
    player=df[df["Player"]==name]
    if player.empty: #null case
        st.error(f"{name} not found")
        return None
    position_data = df[df["Position"] == position].copy()
    #position_data["Team"] = position_data["Team"].apply(lambda x: teams.get(x))
    if nation:
        position_data = position_data[position_data["Nation"]==nation].copy()
    if team:
        position_data = position_data[position_data["Team"]==team].copy()
    #ensure player is present in position df
    if not position_data.isin(player).all(axis=1).any():
        position_data = pd.concat([position_data,player], ignore_index=True)
    position_data = position_data.drop_duplicates(subset=["Player"]).fillna(0)
    #calculate percentiles
    for stat in stats:
        position_data[f"{stat}_Percentile"] = rankdata(position_data[stat], method="average") / len(position_data)
        #account for statistics where lower numbers = better performance like errors and GA
        inverse_stats = ["Errors Per 90", "Goals Allowed Per 90", "GA/SoT Per 90"]
        if stat in inverse_stats:
            position_data[f"{stat}_Percentile"] = 1 - position_data[f"{stat}_Percentile"]
    #extract the percentiles for the selected player
    player_percentiles = position_data[position_data["Player"] == name][[f"{stat}_Percentile" for stat in stats]]

    if player_percentiles.empty:
        st.error(f"No percentile data found for player '{name}' in position '{position}'.")
        return None

    player_percentiles = player_percentiles.iloc[0].to_numpy().reshape(1, -1)

    #perform similarity calculation
    similarity_scores = cosine_similarity(
        position_data[[f"{stat}_Percentile" for stat in stats]], player_percentiles
    ).flatten()
    position_data["Similarity"] = similarity_scores*100

    #sort
    position_data = position_data.sort_values("Similarity", ascending=False)

    #use raw values for radar stats (dislay purpose)
    radar_stats = stats
    output_columns = ["Player", "Team", "Age", "Nation", "Position", "Similarity"] + radar_stats

    return position_data[output_columns].head(threshold)


def load():
    uploaded_file = st.sidebar.file_uploader("Upload field player dataset (CSV)", type=["csv"])
    check_f = True
    if uploaded_file is not None:
        check_f = False
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Uploaded dataset: {uploaded_file.name}")
    else:
        df = pd.read_csv("mls_players.csv")
        st.sidebar.info("Using default MLS field player dataset.")
    return df, check_f
def load_gk():
    uploaded_file = st.sidebar.file_uploader("Upload goalkeeper dataset (CSV)", type=["csv"])
    check_g = True
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "Position" in df.columns:
            st.sidebar.success(f"Uploaded dataset: {uploaded_file.name}")
            check_g = False
            if "Name" in df.columns:                
                df.rename(columns={'Name': 'Player'}, inplace=True)
            elif "name" in df.columns:                
                df.rename(columns={'name': 'Player'}, inplace=True)
        else:
            st.warning("Failed to load, missing position column")
    else:
        df = pd.read_csv("mls_gk.csv")
        st.sidebar.info("Using default MLS goalkeeper dataset.")
    return df, check_g
    
#new function test for generating default stats per position:
def generate_stats(df,position):
    #Define keywords to check for
    position_keywords = {
        "FW": ["Goals", "xG", "Assists", "xAG", "Aerial", "Shot Creating Actions", "SoT", "Take-Ons", "Passes Completed"],
        "MF": ["Take-On", "Pass Completion", "Tackles", "Shot Creating Actions", "Dribble", "Goals per Shot", "Interceptions", "Progressive Pass"],
        "DF": ["Tackles", "Errors", "Passes Completed", "Aerial", "Dribble", "Progressive Pass", "Long Pass"],
        "GK": ["Save", "PSxG", "Crosses", "Pass Completion", "Goals Allowed", "GA/SoT", "Avg Pass"]
    }

    default_stats = []
    choose = position_keywords.get(position)
    for keyword in choose:
        #matching to keywords
        matched_stats = [
            col for col in df.columns 
            if keyword in col
        ]
        default_stats.append(matched_stats)
        

    return default_stats
#create app
def main():
    #st.set_page_config(layout="wide")
    #st.title("MLS Player Comparer")
    
    f, check_f = load()
    g,check_g=load_gk()
    pl = f.copy()
    gk=g.copy()
    if pl.empty or gk.empty:
        st.error("Dataset could not be loaded. Please check the file path.")
        return
    if "Name" in f.columns:
        f.rename(columns={'Name': 'Player'}, inplace=True)
    elif "name" in f.columns:
        f.rename(columns={'name': 'Player'}, inplace=True)      
    
    #setup sidebar and tabs
    st.sidebar.header("Player Search and Filters")
    tab1, tab2,tab3 = st.tabs(["Chart", "Similar Players", "Best Players"])
    with st.sidebar.expander("Required", expanded=True):
        position = st.selectbox("Select Position", options=["FW", "MF", "DF", "GK"])
        df = gk if position=="GK" else pl
        teams = df["Team"].unique()
        all_player_names = sorted(df["Player"].apply(lambda x: unidecode(x)).tolist())
        name = st.selectbox(
                    "Select a Player",
                    options=[""]+all_player_names, 
                    help="Type and it will auto-recommend"
                )       
        
        stats = df.columns.tolist()
        excluded_cols = ["Player", "Name", "Team", "Position", "Age", "Similarity", "Secondary Position", "Nation", "Conference"] + list(df.select_dtypes(exclude='number'))
        stats = [col for col in stats if col not in excluded_cols]
        compare = st.selectbox(
                    "Select a Player to Compare with",
                    options=[""]+all_player_names,
                    help="Select a player from the suggestions."
                )
        stats = st.multiselect(
            "Select Stats for Chart & Table:",
            options=stats,
            default = stats[:5]
        )
        if not stats:
            st.warning("No default stats available for the selected position. Please select stats manually.")
        chart_type = st.selectbox("Select Chart Type", options=["Radar", "Pizza"])
        threshold = st.slider("Number of Similar/Best Players", 1, 25, 1) 
    with st.sidebar.expander("Optional Filters", expanded=False):
        
        nation_filter = st.selectbox("Filter by Nation", options=["All"] + sorted(df["Nation"].dropna().unique().tolist()))
        team_filter = st.selectbox("Filter by Team", options=["All"] + sorted(list(teams)))
        
    with tab3:
        #st.subheader("Top Players Across Selected Stats")

        if not stats:
            st.warning("Please select stats to evaluate the best players.")
        else:
            position_data = df[df["Position"] == position].copy()
            #position_data["Team"] = position_data["Team"].apply(lambda x: teams.get(x))
            #country filter
            if nation_filter != "All":
                position_data = position_data[position_data["Nation"] == nation_filter]

            #team filter
            if team_filter != "All":
                position_data = position_data[position_data["Team"] == team_filter]

            position_data[stats] = position_data[stats].fillna(0)

            for stat in stats:
                position_data[f"{stat}_Percentile"] = rankdata(position_data[stat], method="average") / len(position_data)

                inverse_stats = ["Errors Per 90", "Goals Allowed Per 90", "GA/SoT Per 90"]
                if stat in inverse_stats:
                    position_data[f"{stat}_Percentile"] = 1 - position_data[f"{stat}_Percentile"]

            #uesa average percentile for comparisons
            percentile_columns = [f"{stat}_Percentile" for stat in stats]
            position_data["Score"] = round(position_data[percentile_columns].mean(axis=1)*100,1)
            min_score = position_data["Score"].min()
            max_score = position_data["Score"].max()
            position_data["Score"] = ((position_data["Score"] - min_score) / (max_score - min_score)) * 100
            position_data["Score"] = position_data["Score"].round(1)
            # Round the scaled scores for display
            position_data["Score"] = position_data["Score"].round(1)
            top_players = position_data.sort_values("Score", ascending=False).head(threshold)

        
            st.write(f"### Top {threshold} Players in Position: {position}")
            st.dataframe(
                top_players[["Player", "Team", "Age","Nation","Score"] + stats]
            )
    #choose df from position, then provide available stats for selection
    df["Player"] = df["Player"].apply(lambda x: unidecode(x) if isinstance(x, str) else x)
    if name:
        #name = name.title()
        player = df[df["Player"]==name]
        if not player.empty:
            if len(stats)<3:
                st.warning("Select 3 or more stats for visualization")
            position_data = df[df["Position"] == position].copy()
            #temporarily add the player to the position data if positions don't match
            if player["Position"].iloc[0] != position:
                if position == "GK":
                    st.warning(f"{name} is not a goalkeeper, except Sean Zawadzki :)")
                else:
                    temp_player = player.copy()
                    temp_player["Position"] = position
                    position_data = pd.concat([position_data, temp_player], ignore_index=True)
            #check if comparison player exists
            if compare:
                #st.header(f"{name.title()} vs. {compare.title()}")
                #compare = compare.title()
                
                player_2 = df[df["Player"]==compare]
                if player_2.empty:
                    st.warning(f"{compare} not found")
                #temporarily add the player to the position data if positions don't match
                if player_2["Position"].iloc[0] != position:
                    temp_player = player_2.copy()
                    temp_player["Position"] = position
                    position_data = pd.concat([position_data, temp_player], ignore_index=True)
            #else:
                  #st.header(f"{name}")
            for stat in stats:
                if stat not in position_data.columns:
                    st.warning(f"Stat '{stat}' is not found in the data and will be skipped.")
                    continue
                position_data[stat] = position_data[stat].fillna(0)
                position_data[f"{stat}_Percentile"] = rankdata(position_data[stat], method="average") / len(position_data)
                #inverse/lower score=better stats
                inverse_stats = ["Errors Per 90", "Goals Allowed Per 90", "GA/SoT Per 90"]
                if stat in inverse_stats:
                    position_data[f"{stat}_Percentile"] = 1 - position_data[f"{stat}_Percentile"]
            player_percentiles = position_data[position_data["Player"] == name][[f"{stat}_Percentile" for stat in stats]].iloc[0].to_dict()
            if compare:
                player_2_percentiles = position_data[position_data["Player"] == compare][[f"{stat}_Percentile" for stat in stats]].iloc[0].to_dict()
            position_percentiles = position_data[[f"{stat}_Percentile" for stat in stats]].mean().to_dict()

            #adjust keys for radar chart input
            player_stats = {stat: player_percentiles[f"{stat}_Percentile"] for stat in stats}
            if compare:
                player_2_stats = {stat: player_2_percentiles[f"{stat}_Percentile"] for stat in stats}
            position_stats = {stat: position_percentiles[f"{stat}_Percentile"] for stat in stats}
            #create radar chart
            if stats:
                if compare:
                    title = f"{name} vs {compare} - {position} Comparison"
                    fig = pizza(player_stats, player_2_stats, stats, title, name, compare)
                    #st.pyplot(fig)
                elif chart_type == "Pizza":
                    fig=pizza(player_stats,None,stats,f"{name} - {position} Analysis")
                    #st.pyplot(fig)
                elif chart_type == "Radar":
                    fig=radar(player_stats,position_stats,stats,f"{name} - {position} Analysis")
                    #st.pyplot(fig)
                else:
                    st.warning("Choose Chart Type")
            else:
                st.error("Make sure you have selected stats")
            #similar["Player"] = similar["Player"].apply(lambda x: unidecode(x.title()))            
            #develop tab contents
            with tab1:
                st.pyplot(fig)
            with tab2:
                st.subheader("Similar Players")
                #nation filter
                if name not in all_player_names:
                    st.warning("Player not found")
                else:
                    if nation_filter == "All":
                        nation = None
                    else:
                        nation = nation_filter
                    #team filter
                    if team_filter == "All":
                        team = None
                    else:
                        team = team_filter
                    #get similar players
                    similar = similarity(df,name,position,stats, threshold, nation, team)
                    #similar.reset_index(drop=True, inplace=True)
                    similar = similar.drop_duplicates(subset=['Player'])
                    #get player and pass for highlighter +/-
                    player = similar.iloc[0].to_dict()
                    #similar = similar.style.apply(lambda row: highlight(row,player),axis=1)
                    if not similar.empty:
                        st.dataframe(similar)
                    else:
                        st.warning("No similar players found matching the selected filters.")    

            
        st.sidebar.info(
        """
        **Note**: Currently only have comparison pizza graphs, not radar.
        Also, percentiles for certain stats *Errors Per 90*, *Goals Allowed Per 90*, 
        and *GA/SoT Per 90* are **inverted**. This means lower raw values for these stats 
        correspond to higher percentiles, as lower values indicate better performance.
        """
    )
if __name__ == "__main__":
    main()
                
        