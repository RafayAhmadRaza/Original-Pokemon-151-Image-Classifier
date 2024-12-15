import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from PIL import Image

import tensorflow as tf
import time
gif_url = "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/029b8bd9-cb5a-41e4-9c7e-ee516face9bb/dayo3ow-7ac86c31-8b2b-4810-89f2-e6134caf1f2d.gif?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcLzAyOWI4YmQ5LWNiNWEtNDFlNC05YzdlLWVlNTE2ZmFjZTliYlwvZGF5bzNvdy03YWM4NmMzMS04YjJiLTQ4MTAtODlmMi1lNjEzNGNhZjFmMmQuZ2lmIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.ooubhxjHp9PIMhVxvCFHziI6pxDAS8glXPWenUeomWs"

df = pd.read_csv("PokemonFacts/PokemonFacts.csv")
df.columns = df.columns.str.strip()
df['Names'] = df['Names'].str.strip()
modelDict = {"Best Model":"models/my_model11.keras","Second Best":"models/my_model3.keras","Worst":"models/my_model.keras"}
def main():
    st.title('Pokemon Generation 1 Classifier')
    st.write('Upload any image of the original 151 pokemons')
    modelType = st.selectbox("Models",["Best Model","Second Best","Worst"])
    modelName = modelDict[modelType]
    file= st.file_uploader('Please upload an image',type=['jpg','png','jpeg'])
    

    if file:
        image = Image.open(file)
        st.image(image,use_column_width=True)
        resized_image = image.resize((256,256))
        img_array=np.array(resized_image)/255

        img_array = img_array.reshape(1,256,256,4)

        if st.button("Predict"):    
            gif_placeholder = st.empty()
            text_placeholder = st.empty()
            model = tf.keras.models.load_model(f'{modelName}')
            
            text_placeholder.markdown("Predicting....",unsafe_allow_html=True)
            gif_placeholder.markdown(f"<img src='{gif_url}' width=150>", unsafe_allow_html=True)
            predictions = model.predict(img_array)  
            time.sleep(2)
            text_placeholder.empty()
            gif_placeholder.empty()
            PokemonArray = ['Golbat', 'Beedrill', 'Caterpie', 'Clefable', 'Raichu', 'Sandslash', 'Metapod', 'Drowzee', 'Oddish', 
                'Charizard', 'Tauros', 'Ponyta', 'Primeape', 'Spearow', 'Mankey', 'Poliwag', 'Krabby', 'Rattata', 
                'Tentacruel', 'Graveler', 'Koffing', 'Zapdos', 'Articuno', 'Psyduck', 'Bellsprout', 'Lapras', 
                'Butterfree', 'Weezing', 'Abra', 'Muk', 'Cloyster', 'Porygon', 'Flareon', 'Jigglypuff', 'Raticate', 
                'Venusaur', 'Dewgong', 'Horsea', 'Rhydon', 'Omanyte', 'Exeggcute', 'Kabuto', 'Ditto', 'Growlithe', 
                'Mew', 'Electrode', 'Vileplume', 'Seaking', 'Exeggutor', 'Electabuzz', 'Chansey', 'Magmar', 'Haunter', 
                'Ninetales', 'Clefairy', 'Nidoran-m', 'Gyarados', 'Tangela', 'Marowak', 'Snorlax', 'Nidoqueen', 
                'Hitmonchan', 'Ekans', 'Sandshrew', 'Jolteon', 'Kabutops', 'Lickitung', 'Pidgeotto', 'Shellder', 
                'Slowpoke', 'Pikachu', 'Poliwrath', 'Fearow', 'Magnemite', 'Hitmonlee', 'Machoke', 'Poliwhirl', 
                'Magneton', 'Diglett', 'Venonat', 'Kakuna', 'Eevee', 'Ivysaur', 'Doduo', 'Wigglytuff', 'Goldeen', 
                'Alakazam', 'Starmie', 'Grimer', 'Pinsir', 'Tentacool', 'Mewtwo', 'Dodrio', 'Kangaskhan', 'Arcanine', 
                'Dratini', 'Aerodactyl', 'Gastly', 'Geodude', 'Magikarp', 'Zubat', 'Paras', 'Machamp', 'Victreebel', 
                'Wartortle', 'Omastar', 'Meowth', 'Nidorina', 'Bulbasaur', 'Farfetchd', 'Nidoran-f', 'Rapidash', 
                'Seel', 'Blastoise', 'Venomoth', 'Hypno', 'Golduck', 'Nidoking', 'Vaporeon', 'Dragonite', 'Onix', 
                'Pidgeot', 'Machop', 'Moltres', 'Scyther', 'MrMime', 'Cubone', 'Gengar', 'Kingler', 'Dugtrio', 'Gloom', 
                'Parasect', 'Persian', 'Golem', 'Seadra', 'Squirtle', 'Nidorino', 'Charmander', 'Jynx', 'Dragonair', 
                'Arbok', 'Weedle', 'Pidgey', 'Kadabra', 'Rhyhorn', 'Weepinbell', 'Charmeleon', 'Staryu', 'Voltorb', 
                'Slowbro', 'Vulpix']

                
            PokemonArray.sort()
            st.markdown(f"### My Predication is: " + PokemonArray[np.argmax(predictions[0])])
            st.markdown(f"### Fun Fact About {PokemonArray[np.argmax(predictions[0])]}")
            predictedname = PokemonArray[np.argmax(predictions[0])]
            fact = df.loc[df['Names'] == predictedname]['Fact'].values[0]

            st.markdown("#### " + fact,unsafe_allow_html=True)
            
            st.markdown("### Prediction Spread")
           
            fig,ax = plt.subplots(figsize=(10, 27.5)) 
            y_pos = np.arange(len(PokemonArray))
            ax.barh(y_pos,predictions[0],align='center')
            ax.set_yticks(y_pos)

            ax.set_yticklabels(PokemonArray)

            ax.invert_yaxis()
            ax.set_xlabel("Probabilitly")
            ax.set_ylabel("Pokemons")
            plt.ylabel(ax.get_ylabel(),fontsize=25)
            plt.xlabel(ax.get_xlabel(),fontsize=20)

            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10.5)

            st.pyplot(fig)


    else:
        st.text('You have not uploaded an image yet')
    


if __name__ == '__main__':
    main()