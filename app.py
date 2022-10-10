import numpy as np
from pickle import load
import pandas as pd
import sklearn
import streamlit as st
import time
import os
from numpy.linalg import norm

# formatting, create 3 columns
col1, col2  = st.columns([0.75,1])

# Title

col2.title('Discover San Diego!')

# Subtitle
subtitle = '<p style="color:#FF5A5F; font-size: 23px;">Select an Airbnb listing and get recommendations for similar listings in San Diego</p>'
col2.markdown(subtitle, unsafe_allow_html = True)
col2.caption("Built by Eric Au | Github: @eric8395 | LinkedIn: www.linkedin.com/in/eric-au8395/")

# add image
col1.image("https://images.unsplash.com/photo-1593970107436-6b5c6f8f1138?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80", use_column_width= 'always', caption = "Photo Credit Jermaine Ee via Unsplash")

# Disclaimer
with st.expander("About This App"):
    st.write("What is this?")
    st.caption("Airbnb is an online marketplace that is focussed on connecting people who rent out their homes with people who are looking for accommodations around the world. If you're like me and love to travel, Airbnb provides a cheaper alternative to hotels while also offering an eccentric experience that adds value to vacations.")
    st.caption("I personally love traveling to the west coast and finding unique Airbnb listings. Sometimes I'd find myself wanting to go back but have trouble looking for a similar experience. There is currently no system in place where Airbnb will provide (or recommend) similar homes I have previously stayed in.")
    st.caption('**With this premise, I developed a web application that provides recommendations for San Diego Airbnb listings and similar listings based on user input.**')
    st.caption("For more information on how I built this recommendation system, feel free to check out my Github here: https://github.com/eric8395/airbnb_recommendations")

    st.write("Data Source")
    st.caption("San Diego is a city that I frequently visit on the west coast. Therefore, I was compelled to work on a dataset of Airbnb listings within the area.\
        The dataset for this project consists of over 13,000 rows of data for San Diego Airbnb Listings as of August 2019 and publicly sourced from data.world via Inside Airbnb.")
    
    st.write("Disclaimer")
    st.caption("All content and information on this application is for informational and educational purposes only and is not directly affiliated with Airbnb in any way.")


## UPLOAD THE DATAFRAMES ##
# ----------------------------------------------- #

# load in sd_trans dataframe to be transformed
sd_trans = pd.read_csv('sd_trans', index_col = 0)

# load in url_listings dataframe to be joined
sd_listings_url = pd.read_csv('url_listings.csv', index_col = 0)

## UPLOAD THE SELECTION DFS ## 
# ----------------------------------------------- #

# load in sd_pp, FOR SELECTION OF URL WITHIN THE PREPROCESSED DF
sd_pp = pd.read_csv('sd_pp.csv', index_col = 0)

# load in sd_clustered
sd_clustered = pd.read_csv('sd_clustered.csv', index_col = 0)

# merge url listings with sd_trans
sd_merged = sd_listings_url.join(sd_trans)

## GET URL LISTING FROM PP DATASET ##
# ----------------------------------------------- #
# SELECT THE LISTING FROM UNPROCESSED DATASET 
# get sd_clustered and merge with urls on index
sd_clustered = sd_clustered.join(sd_listings_url)

# select a listing from sd_merged
selected_listing = st.selectbox("Choose a listing below:", sd_merged.listing_url)

# based on selected listing, get the index from sd_pp
index_value = sd_merged.listing_url[sd_merged.listing_url == str(selected_listing)].index[0]
selected_listing_df = pd.DataFrame(sd_pp.iloc[index_value]).T

## TRANSFORMS THE DATASET INTO A PREPROCESSED SET ## 
# ----------------------------------------------- #
# unpickle and load in column transformer
ct = load(open("column_transformer.pkl", 'rb'))

 ## GET RECOMMENDATION BASED ON SELECTED LISTING ##
# ----------------------------------------------- #

# get a recommendation based on url selection
def get_recommendations(df, listing):
    """
    Takes in preprocessed dataframe and selected listing as inputs and gives top 5
    recommendations based on cosine similarity. 
    """
    # reset the index
    df = df.reset_index(drop = 'index')
    
    # convert single listing to an array
    listing_array = listing.values

    # convert all listings to an array
    df_array = df.values
    
    # get arrays into a single dimension
    A = np.squeeze(np.asarray(df_array))
    B = np.squeeze(np.asarray(listing_array))
    
    # compute cosine similarity 
    cosine = np.dot(A,B)/(norm(A, axis = 1)*norm(B))
    
    # add similarity into recommendations df and reset the index
    rec = sd_clustered.copy().reset_index(drop = 'index')
    rec['similarity'] = pd.DataFrame(cosine).values

    # simplify the dataframe and keep only necessary columns
    rec = rec[['listing_url', 'similarity', 'cluster_label', 'neighbourhood_cleansed',
                'property_type', 'room_type', 'accommodates', 'bathrooms', 'beds',
                'nightly_price', 'review_scores_rating']]
    
    # rename columns
    rec = rec.rename(columns = {'listing_url': 'URL', 
                                'similarity': 'Similarity Score',
                                'cluster_label': 'Listing Category',
                                'neighbourhood_cleansed': 'Neighborhood',
                                'property_type': 'Property Type',
                                'room_type': 'Room Type',
                                'accommodates': 'Accommodates',
                                'bathrooms': ' Bathrooms',
                                'beds': 'Beds',
                                'nightly_price': 'Nightly Price',
                                'review_scores_rating': 'Review Rating'})

    # rename the cluster_label AKA Listing Category column values
    rec = rec.replace({'Listing Category': {0:'Popular High End',
                                            1:'Highly Rated & Moderately Priced',
                                            2:'Diverse & Moderately Priced',
                                            3:'Favorable & Budget Friendly',
                                            4:'Unfavorable & Poorly Rated'}})
    

    # selected listing
    selection = st.dataframe(rec.sort_values(by = ['Similarity Score'], ascending = False)[0:1])
    
    if selection: 
        st.write("Recommended similar stays:")

    # sort by top 5 descending
    recommended_listings = st.dataframe(rec.sort_values(by = ['Similarity Score'], ascending = False)[1:6])

    return selection, recommended_listings

# get recommendation
get_recommendations(sd_pp, selected_listing_df)

## GET A DATAFRAME BASED ON USER SELECTED PARAMETERS FOR A SIMPLIFIED RECOMMENDATION ## 
# ----------------------------------------------- #

# load in the simplified dataset
sd_simplified = pd.read_csv('sd_simplified.csv', index_col = 0)


# formatting, create 3 columns
col3, col4  = st.columns([1,0.95])

# header
col3.markdown(" ### Personalize Your Stay!")
# add image
col4.image("https://images.unsplash.com/photo-1617142584114-730e9bda61b2?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1471&q=80", width = 410, caption = "Photo Credit Andres Garcia via Unsplash")
# subtitle markdown
# Subtitle
subtitle2 = '<p style="color:#FF5A5F; font-size: 20px;">Using the dropdown menus and sliders below, get recommendations for your perfect getaway</p>'
col3.markdown(subtitle2, unsafe_allow_html = True)

## GET USER INPUTS ## 
# ----------------------------------------------- #
neighborhood = st.selectbox('Neighborhood:',(sd_simplified['neighbourhood_cleansed'].unique()))
property = st.selectbox('Property Type:',(sd_simplified['property_type'].unique()))
room = st.selectbox('Room Type:',(sd_simplified['room_type'].unique()))
accommodation = st.selectbox('Accommodation:',(sd_simplified['accommodates'].unique()))
bathrooms = st.number_input("Bathrooms:",
                            min_value = 1,
                            max_value = 10)
beds = st.number_input("Beds:",
                        min_value = 1,
                        max_value = 10)
price = st.slider('Minimum Nightly Price ($):', int(sd_simplified['nightly_price'].quantile(0.05).item()),  # min
                                            int(sd_simplified['nightly_price'].quantile(0.85).item()),  # max
                                            int(sd_simplified['nightly_price'].median().item()),
                                            step = 1) # start point

rating = st.slider('Minimum Rating:',  int(sd_simplified['review_scores_rating'].min().item()),  # min
                                            int(sd_simplified['review_scores_rating'].quantile(0.75).item()),  # max
                                            90, # start at 90
                                            step = 1) # start point


## GET RECOMMENDATION BASED ON USER INPUTS ##
# ----------------------------------------------- #

def get_simplified_recommendations(df, user_inputs):
    """
    Takes in preprocessed dataframe and user inputs df and gives top 5
    recommendations based on cosine similarity. 
    """
    # reset the index
    df = df.reset_index(drop = 'index')
    
    # transform the user_inputs dataframe into preprocessed dataset
    user_inputs_df_pp = pd.DataFrame(simple_ct.transform(user_inputs))
    
    # convert single listing to an array
    listing_array = user_inputs_df_pp.values

    # convert all listings to an array
    df_array = df.values
    
    # get arrays into a single dimension
    A = np.squeeze(np.asarray(df_array))
    B = np.squeeze(np.asarray(listing_array))
    
    # compute cosine similarity 
    cosine = np.dot(A,B)/(norm(A, axis = 1)*norm(B))
    
    # add similarity into recommendations df and reset the index
    rec = sd_simplified.copy().reset_index(drop = 'index')
    rec['similarity'] = pd.DataFrame(cosine).values
    
    # add in listings_urls
    # merge on index
    rec = rec.join(sd_listings_url)
    
    # reorder column names
    rec = rec[['listing_url', 'similarity', 'neighbourhood_cleansed', 'property_type', 
                'room_type', 'accommodates', 'bathrooms', 'beds', 'nightly_price', 'review_scores_rating']]
    
    rec = rec.rename(columns = {'listing_url': 'URL', 
                                'similarity': 'Similarity Score',
                                'neighbourhood_cleansed': 'Neighborhood',
                                'property_type': 'Property Type',
                                'room_type': 'Room Type',
                                'accommodates': 'Accommodates',
                                'bathrooms': ' Bathrooms',
                                'beds': 'Beds',
                                'nightly_price': 'Nightly Price',
                                'review_scores_rating': 'Review Rating'})

    # sort by top 5 descending
    return st.dataframe(rec.sort_values(by = ['Similarity Score'], ascending = False).head(5))

# get recommendations if button is pressed
if st.button("Take Me Away!"):
    
    # store inputs into df
    column_names = ['Neighborhood', 'Property Type', 'Room Type', 'Accommodation', 'Bathrooms', 
                'Number of Beds', 'Minimum Nightly Price ($)', 'Minimum Rating']
    
    user_inputs = pd.DataFrame([[neighborhood, property, room, accommodation, bathrooms, beds, price, rating]], 
                            columns = sd_simplified.columns)
    
    # unpickle and load in the SIMPLE column transformer
    simple_ct = load(open("simple_column_transformer.pkl", 'rb'))
    # transform the simplified dataset
    sd_simplified_pp = pd.DataFrame(simple_ct.fit_transform(sd_simplified))

    st.markdown("#### Recommendations based on your selections:")
    get_simplified_recommendations(sd_simplified_pp, user_inputs)


