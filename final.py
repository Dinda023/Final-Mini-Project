import streamlit as st
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

df = pd.read_csv('ramen-ratings.csv')

# Sidebar
st.sidebar.title('Halaman')
selected_option = st.sidebar.selectbox('Select an option:', ['Dashboard', 'Distribution', 'Comparison', 'Composition', 'Relationship', 'Clustering'])

# Main content based on selected option
if selected_option == 'Dashboard':
    st.title("Dashboard")
    st.subheader("""
    Analisis Ramen Terpopuler Berdasarkan Penilaian Konsumen
    """)
    st.text('Tabel')
    st.write(df)
    st.markdown("""
                Analisis ini bertujuan untuk menganalisis faktor-faktor yang memengaruhi penilaian konsumen terhadap ramen dan meningkatkan kualitas produk serta pangsa pasar ramen.

                Data yang dipakai adalah data Ramen Ratings oleh ALEKSEY BILOGUR dari Kaggle. Dataset Ramen Ratings adalah kumpulan data yang memuat informasi tentang berbagai merek ramen dari seluruh dunia bersama dengan penilaian atau peringkat yang diberikan oleh para pengulas. Dataset ini dapat digunakan untuk menganalisis variasi dalam berbagai merek dan jenis ramen, serta untuk memahami preferensi pengguna.
                """)
elif selected_option == 'Distribution':
    st.title('Rating Tertinggi')
    df['Stars'] = df['Stars'].str.replace(r'[^0-9.]', '')
    df['Stars'] = pd.to_numeric(df['Stars'], errors='coerce')
    df.dropna(subset=['Stars'], inplace=True)
    review_counts = df['Brand'].value_counts()
    top_10_brands = review_counts.head(10)
    average_ratings_data = []
    for brand in top_10_brands.index:
        brand_reviews = df[df['Brand'] == brand]
        average_rating = brand_reviews['Stars'].mean()
        average_ratings_data.append({'Brand': brand, 'Average Rating': average_rating})
    average_ratings = pd.DataFrame(average_ratings_data)
    top_10_brands = top_10_brands.sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    top_10_brands.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title('Top 10 Brands by Review Counts')
    ax.set_xlabel('Brand')
    ax.set_ylabel('Review Counts')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.markdown("""
Grafik menunjukkan 10 merek mie instan teratas berdasarkan jumlah ulasan. Merek dengan jumlah ulasan terbanyak adalah Indomie, Mama, Maruchan, Myojo, Nissin, Ottogi, Paldo, Samyang Foods, dan Vina Acecook.
""")
    st.pyplot(fig)
    st.write("Data Rata-rata Rating untuk Top 10 Brand:")
    st.write(average_ratings)
    st.markdown("""
Rata-rata rating dari 10 brand mie instan ini adalah 3.7873. Hal ini menunjukkan bahwa secara keseluruhan, konsumen puas dengan kualitas mie instan yang mereka beli.
1. Ada 3 brand mie instan yang memiliki rating di atas 4, yaitu Paldo, Indomie, dan Samyang Foods. Brand-brand ini dapat dikatakan sebagai brand mie instan yang paling disukai oleh konsumen.
2. Ada 4 brand mie instan yang memiliki rating antara 3.5 dan 4, yaitu Nongshim, Maruchan, Mama, dan Myojo. Brand-brand ini juga merupakan brand mie instan yang populer dan disukai oleh konsumen.
3. Ada 3 brand mie instan yang memiliki rating di bawah 3.5, yaitu Nissin, Ottogi, dan Vina Acecook. Brand-brand ini perlu meningkatkan kualitas produknya agar dapat meningkatkan rating dari konsumen.
""")
elif selected_option == 'Comparison':
    st.title('Comparison')
    top_countries = df['Country'].value_counts().head(10)

    # Gabungkan data negara yang tidak termasuk dalam 10 teratas ke dalam kategori 'Other'
    other_countries_count = df['Country'].value_counts().sum() - top_countries.sum()
    top_countries['Other'] = other_countries_count

    # Ambil 3 data teratas untuk kolom 'Style'
    top_styles = df['Style'].value_counts().head(3)

    # Gabungkan data gaya yang tidak termasuk dalam 3 teratas ke dalam kategori 'Other'
    other_styles_count = df['Style'].value_counts().sum() - top_styles.sum()
    top_styles['Other'] = other_styles_count

    # Sidebar
    st.sidebar.title('Pilihan Kolom untuk Dibandingkan')
    selected_column = st.sidebar.selectbox("Pilih Kolom untuk Dibandingkan", ['Country', 'Style'])

    # Data preprocessing
    if selected_column == 'Country':
        comparison_data = top_countries
        title = 'Komparasi Berdasarkan Negara'
        xlabel = 'Negara'
        ylabel = 'Jumlah'
        st.markdown("""
Berikut adalah beberapa poin kunci dari grafik:

1. Amerika Serikat memiliki jumlah review ramen terbanyak, dengan lebih dari 20% dari total review. Hal ini menunjukkan bahwa ramen adalah makanan yang populer di Amerika Serikat.
2. Indonesia dan Jepang memiliki jumlah review ramen yang hampir sama, masing-masing sekitar 18% dan 13% dari total review. Hal ini menunjukkan bahwa ramen juga merupakan makanan yang populer di Indonesia dan Jepang.
3. Korea Selatan dan Singapura memiliki jumlah review ramen yang signifikan, masing-masing sekitar 8% dan 5% dari total review. Hal ini menunjukkan bahwa ramen mulai populer di Korea Selatan dan Singapura.
4. Negara-negara lain dalam daftar ini memiliki jumlah review ramen yang lebih kecil, dengan persentase berkisar antara 1% hingga 4%. Hal ini menunjukkan bahwa ramen masih belum terlalu populer di negara-negara tersebut.
""")
    elif selected_column == 'Style':
        comparison_data = top_styles
        title = 'Komparasi Berdasarkan Gaya'
        xlabel = 'Gaya'
        ylabel = 'Jumlah'
        st.markdown("""
1. Ramen kemasan Pack adalah kemasan ramen yang paling populer, dengan setengah lebih dari semua ulasan menggunakan kemasan ini.
2. Ramen Bowl adalah gaya ramen terpopuler kedua, dengan 18.7% ulasan menyebutkan kemasan ini.
3. Ramen Miso adalah gaya ramen terpopuler ketiga, dengan 17.5% ulasan menggunakan kemasan ini.
                    """)

    # Plot pie chart
    fig, ax = plt.subplots()
    ax.pie(comparison_data, labels=comparison_data.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    st.pyplot(fig)
    # Add your code for comparison here
elif selected_option == 'Composition':
    st.title('Composition')
    df['Stars'] = df['Stars'].str.replace(r'[^0-9.]', '')
    df['Stars'] = pd.to_numeric(df['Stars'], errors='coerce')
    top_stars = df['Stars'].value_counts().head(10).index.tolist()
    df_top_stars = df[df['Stars'].isin(top_stars)]
    df_numeric = df_top_stars.select_dtypes(include=['int', 'float64'])
    stars_composition = df_numeric.groupby('Stars').mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(stars_composition.T, annot=True, fmt='g', cmap='YlGnBu', ax=ax)
    ax.set_title('Komposisi untuk Top 10 Stars Terbanyak')
    ax.set_xlabel('Nilai Stars')
    ax.set_ylabel('Fitur')
    st.markdown("""Grafik tersebut menunjukkan bahwa rating paling umum untuk restoran ramen adalah 4 bintang, diikuti oleh 3 bintang, 5 bintang, 2 bintang, dan 1 bintang. Ini menandakan bahwa sebagian besar restoran ramen menerima review positif.

Selain itu, grafik ini juga menunjukkan bahwa jumlah review untuk tiap rating semakin sedikit seiring dengan meningkatnya rating. Ini artinya orang lebih cenderung menulis review untuk restoran ramen yang meninggalkan kesan, baik positif maupun negatif.

- Banyaknya rating 4 bintang menunjukkan bahwa ramen adalah hidangan populer yang pada umumnya disukai orang.
- Jumlah review yang relatif sedikit untuk restoran ramen dengan rating 1 dan 2 bintang menunjukkan bahwa tidak banyak restoran ramen yang sangat buruk.
- Jumlah review yang relatif tinggi untuk restoran ramen dengan rating 4 dan 5 bintang menunjukkan bahwa ada banyak restoran ramen yang luar biasa.""")
    st.pyplot(fig)
    # Add your code for composition here
elif selected_option == 'Relationship':
    st.title('Relationship')
    # Membersihkan kolom 'Stars' dari nilai 'Unrated'
    df_cleaned = df[df['Stars'] != 'Unrated']

    # Konversi kolom 'Stars' menjadi tipe data numerik
    df_cleaned['Stars'] = pd.to_numeric(df_cleaned['Stars'], errors='coerce')

    # Pilih kolom 'Review' dan 'Stars'
    review_stars_df = df_cleaned[['Review #', 'Stars']]

    # Hitung matriks korelasi
    correlation_matrix = review_stars_df.corr()

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Heatmap Korelasi antara Review dan Stars')
    plt.tight_layout()
    st.pyplot(plt.gcf()) 
    st.markdown("""Heatmap berikan menunjukkan hubungan antar kolom dalam database penilaian ramen. Heatmap ini menggunakan kode warna, dengan warna hangat (merah dan jingga) menunjukkan korelasi yang kuat dan warna dingin (biru dan hijau) menunjukkan korelasi yang lemah.
""")

elif selected_option == 'Clustering':
    st.subheader('Clustering Analysis based on Selected Features')
    st.write("For clustering analysis, we'll focus on the selected features.")

    # Select numeric and non-numeric features for clustering
    selected_numeric_features = ['Stars']
    selected_non_numeric_features = ['Brand', 'Variety', 'Style', 'Country']
    clustering_data_numeric = df[selected_numeric_features]
    clustering_data_non_numeric = df[selected_non_numeric_features]

    # Handle missing values by converting 'Stars' to numeric
    clustering_data_numeric['Stars'] = pd.to_numeric(clustering_data_numeric['Stars'], errors='coerce')

    # Handle missing values by replacing them with mean for numeric features
    imputer_numeric = SimpleImputer(strategy='mean')
    clustering_data_numeric_imputed = pd.DataFrame(imputer_numeric.fit_transform(clustering_data_numeric), columns=clustering_data_numeric.columns)

    # Handle missing values by replacing them with mode for non-numeric features
    imputer_non_numeric = SimpleImputer(strategy='most_frequent')
    clustering_data_non_numeric_imputed = pd.DataFrame(imputer_non_numeric.fit_transform(clustering_data_non_numeric), columns=clustering_data_non_numeric.columns)

    # Combine numeric and non-numeric features
    clustering_data_combined = pd.concat([clustering_data_numeric_imputed, clustering_data_non_numeric_imputed], axis=1)

    # Perform one-hot encoding for categorical variables
    clustering_data_encoded = pd.get_dummies(clustering_data_combined)

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data_encoded)

    # Selecting number of clusters with slider
    num_clusters = st.slider("Select number of clusters (2-8):", min_value=2, max_value=8, value=4, step=1)

    # Load the pre-trained models
    with open('KMeans.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open('Hierarchical.pkl', 'rb') as f:
        hierarchical = pickle.load(f)

    # Fit KMeans model
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(scaled_data)

    # Get cluster labels from KMeans model
    kmeans_cluster_labels = kmeans.labels_
    hierarchical_cluster_labels = hierarchical.fit_predict(scaled_data)

    # Visualizing the clusters
    plt.figure(figsize=(8, 6))

    # Plot KMeans clustering
    plt.subplot(1, 2, 1)
    plt.scatter(clustering_data_numeric_imputed['Stars'], kmeans_cluster_labels, cmap='viridis', s=50)
    plt.title('KMeans Clustering')
    plt.xlabel('Stars')
    plt.ylabel('Cluster')
    plt.grid(True)

    # Plot Hierarchical clustering
    plt.subplot(1, 2, 2)
    plt.scatter(clustering_data_encoded['Stars'], hierarchical_cluster_labels, cmap='viridis', s=50)
    plt.title('Hierarchical Clustering')
    plt.xlabel('Stars')
    plt.ylabel('Cluster')
    plt.grid(True)

    st.pyplot(plt)

    # Interpretation of clusters
    st.write(f"Number of Clusters (KMeans): {kmeans.n_clusters}")
    st.write(f"Number of Clusters (Hierarchical): {hierarchical.n_clusters}")
    st.markdown("""Clustering digunakan untuk menganalisis dan memahami berbagai kumpulan data. Dalam database penilaian ramen, clustering dapat digunakan untuk mengenali berbagai jenis ramen dan memberikan informasi mengenai produk yang disukai pasar.""")