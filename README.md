# Laporan Proyek Machine Learning - Anju Anjannah
## Domain Proyek
Domain yang dipilih untuk proyek *machine learning* ini adalah **Hiburan (Entertainment)**, dengan judul **K-Drama Recommendation System using Content-Based Filtering.**

### Latar Belakang

![image hospital playlist](https://img.merahputih.com/media/f2/c1/9c/f2c19cad0fa1dfaa49c3e2733f6ec503.jpeg)

Industri hiburan, khususnya serial televisi dan film, mengalami pertumbuhan yang pesat dalam beberapa tahun terakhir. Fenomena global Korean Wave telah mendorong popularitas K-Drama ke tingkat internasional, menghasilkan peningkatan yang signifikan dalam jumlah judul K-Drama yang dirilis setiap tahun [[1]](https://www.researchgate.net/publication/379107215_How_You_Like_That_Development_of_a_Korean_Drama_Recommendation_System_Through_Sentiment_Analysis). Bagi para penggemar K-Drama, semakin banyak pilihan yang tersedia ini, meskipun positif, juga menimbulkan tantangan dalam menemukan K-Drama yang sesuai dengan selera individu. Proses pencarian K-Drama secara manual dengan menjelajahi berbagai platform streaming, membaca sinopsis, dan melihat ulasan dapat menjadi tugas yang memakan waktu dan seringkali tidak efisien. Pengguna mungkin melewatkan judul-judul yang berpotensi mereka sukai karena keterbatasan waktu atau ketidakmampuan untuk menjelajahi semua opsi yang ada.

Masalah ini dapat diselesaikan dengan mengembangkan sistem rekomendasi yang memanfaatkan teknik machine learning. Sistem rekomendasi berbasis konten, seperti yang diimplementasikan dalam proyek ini, menawarkan solusi yang efektif dengan menganalisis fitur-fitur intrinsik dari K-Drama itu sendiri, seperti genre, sinopsis, pemeran, dan tag, untuk mengidentifikasi kesamaan antara judul-judul yang berbeda. Dengan memahami fitur-fitur dari K-Drama yang disukai pengguna, sistem dapat merekomendasikan K-Drama lain yang memiliki karakteristik serupa, sehingga mempermudah pengguna dalam menemukan konten yang relevan dan menarik.

Penggunaan Content-Based Filtering dalam sistem rekomendasi K-Drama telah terbukti efektif dalam penelitian sebelumnya. Misalnya, beberapa studi menunjukkan bahwa pendekatan berbasis konten dapat mencapai performa yang baik dalam memprediksi preferensi pengguna dan menghasilkan rekomendasi yang akurat [[2]](https://journals.indexcopernicus.com/api/file/viewByFileId/1481563). Dengan mengimplementasikan sistem ini, pengguna dapat menghemat waktu dan tenaga dalam pencarian K-Drama, meningkatkan kepuasan mereka dalam menonton, dan pada akhirnya, memperkaya pengalaman mereka dalam menikmati konten hiburan dari Korea Selatan.

### Business Understanding
Industri hiburan menghadapi tantangan dalam membantu pengguna menemukan konten yang relevan di tengah banyaknya pilihan. Proses pencarian manual seringkali sulit dan memakan waktu, berdampak pada kepuasan dan retensi pengguna. Proyek ini bertujuan menyediakan sistem rekomendasi berbasis konten yang dapat diandalkan untuk merekomendasikan K-Drama secara personal, meningkatkan pengalaman pengguna, mendorong engagement, dan meningkatkan retensi di platform hiburan.

### Problem Statements
Berdasarkan latar belakang di atas, berikut ini merupakan rincian masalah yang dapat diselesaikan pada proyek ini:
- Bagaimana membangun sistem rekomendasi berbasis konten yang efektif untuk K-Drama menggunakan fitur genre dan teknik seperti TF-IDF Vectorization dan Cosine Similarity?
- Bagaimana cara mengukur kesamaan antar K-Drama berdasarkan fitur konten genre dan menggunakan kesamaan tersebut untuk menghasilkan rekomendasi?
- Bagaimana sistem dapat menangani input dari pengguna (nama K-Drama) dan memberikan daftar rekomendasi K-Drama yang relevan berdasarkan kemiripan genre?

### Goals
- Untuk membangun sistem rekomendasi K-Drama berbasis konten menggunakan fitur genre dan metode seperti TF-IDF Vectorizer dan Cosine Similarity.
- Untuk menghasilkan rekomendasi K-Drama yang relevan bagi pengguna berdasarkan kemiripan genre.
- Untuk menyediakan antarmuka yang memungkinkan pengguna untuk memasukkan nama K-Drama dan menerima daftar rekomendasi yang disarankan berdasarkan genre.

### Solution Statement
Proyek ini mengusulkan solusi berbasis machine learning untuk mengatasi tantangan penemuan konten dalam industri hiburan, dengan fokus pada K-Drama. Solusi ini melibatkan pembangunan sistem rekomendasi berbasis konten yang berfokus pada fitur genre. Data K-Drama akan melalui proses persiapan untuk memilih fitur genre yang relevan. Kemudian, CountVectorizer akan digunakan untuk mengubah data genre menjadi representasi numerik dalam bentuk matriks. Cosine Similarity akan dihitung antar K-Drama berdasarkan matriks ini untuk mengukur tingkat kemiripan genre. Sistem akan mengambil input nama K-Drama dari pengguna, menemukan K-Drama yang paling mirip berdasarkan genre, dan mengurutkan K-Drama lain berdasarkan skor kesamaan genre untuk menghasilkan daftar rekomendasi. Solusi ini bertujuan untuk memberikan pengguna alat yang efektif untuk menemukan K-Drama baru yang sesuai dengan selera mereka berdasarkan genre yang mereka sukai.

## Data Understanding

Tahap ini berfokus pada pemahaman mendalam terhadap dataset sebelum memulai pemrosesan dan pemodelan. Ini merupakan langkah krusial untuk memastikan data siap digunakan dan untuk mendapatkan wawasan awal.

Dataset ini bersumber dari: **https://www.kaggle.com/datasets/ahbab911/top-250-korean-dramas-kdrama-dataset/**. Dataset ini berisi informasi tentang berbagai K-Drama, termasuk nama, tanggal tayang, tahun rilis, jaringan penayang, jumlah episode, rating konten, sinopsis, genre, tag, sutradara, penulis skenario, pemeran, perusahaan produksi, dan peringkat.

Beberapa fitur penting yang terdapat dalam dataset awal meliputi:
- **Name**: Nama atau judul K-Drama. 
- **Aired Date**: Tanggal penayangan perdana K-Drama.
- **Year of release**: Tahun rilis K-Drama.
- **Original Network**: Jaringan televisi asli yang menayangkan K-Drama.
- **Aired On**: Hari atau waktu penayangan K-Drama.
- **Number of Episodes**: Jumlah total episode dalam K-Drama.
- **Duration**: Durasi per episode K-Drama.
- **Content Rating**: Klasifikasi rating konten (misalnya, Semua Umur, 13+, 18+).
- **Rating**: Rating rata-rata K-Drama oleh pengguna atau kritikus.
- **Synopsis**: Ringkasan cerita atau alur K-Drama.
- **Genre**: Kategori genre K-Drama (misalnya, Romance, Comedy, Thriller).
- **Tags**: Kata kunci atau label yang terkait dengan K-Drama, seringkali menggambarkan tema atau mood.
- **Director**: Nama sutradara K-Drama.
- **Screenwriter**: Nama penulis naskah K-Drama.
- **Cast**: Daftar nama pemeran utama atau penting dalam K-Drama.
- **Production companies**: Nama perusahaan produksi yang terlibat dalam pembuatan K-Drama.
- **Rank**: Peringkat K-Drama dalam daftar tertentu (misalnya, top 250).

### a. EDA : Descriptive Analysis
Setelah data berhasil dimuat, pemeriksaan awal menggunakan `.shape` menunjukkan dataset memiliki **250 baris dan 17 kolom**.

Pemeriksaan tipe data dan non-null count menggunakan `.info()` memberikan gambaran tentang jenis data yang terkandung dalam setiap kolom dan keberadaan nilai yang hilang.`.isna().sum()` secara spesifik mengkonfirmasi adanya nilai yang hilang di beberapa kolom dataset ini, seperti `Content Rating`, `Director`, `Screenwriter`, dan `Production companies`. Analisis statistik deskriptif menggunakan `.describe()` (untuk numerik) dan `.describe(include='object')` (untuk kategorikal) merangkum distribusi fitur numerik seperti nilai rata-rata, standar deviasi, nilai minimum dan maksimum, serta kuartil, memberikan wawasan tentang rentang dan penyebaran data. Pemeriksaan data duplikat dengan `df[df.duplicated()]` menunjukkan tidak ada data yang duplikat.

Tabel 1. Deskripsi Statistik Data Numerikal

|       | Year of release | Number of Episodes |   Rating   |
| :---: | :-------------: | :---------------:  | :--------: |
| count |    250.00000    |     250.000000     | 250.000000 |
|  mean |   2018.25600	 |      19.064000	  |   8.534000 |
|  std  |      3.26452	 |      13.245743	  |   0.221359 |
|  min  |	  2003.00000    |	    1.000000	  |   8.300000 |
|  25%  |   2017.00000	 |      16.000000	  |   8.300000 |
|  50%  |	  2019.00000	 |      16.000000	  |   8.500000 |
|  75%  |   2021.00000	 |      20.000000	  |   8.700000 |
|  max  |	  2022.00000	 |     133.000000	  |   9.200000 |

Tabel 2. Deskripsi Statistik Data Kategorikal

|        |        Name	    |           Aired Date        | Original Network |	     Aired On	      |    Duration   |      Content Rating      |	                     Synopsis                 |	                 Genre                |	                     Tags                         |	 Director   |	Screenwriter |	                     Cast                         | Production companies |	 Rank  |
|  :---: |    :----------:   |  :------------------------: | :--------------: | :------------------: | :-----------: | :---------------------:  | :-----------------------------------------------: |  :-----------------------------------:  | :-----------------------------------------------: |    :--------:  |  :----------: | :-----------------------------------------------: | :------------------: | :-----:|
|  count |	    250       |	         250            |	  250        |	        250          |	    250     |	         245            |	                       250                    |	                  250                 |	                     250                          |	   249      |	    249      |	                     250                          |	      248         |	  250  |
| unique |	    250       |	         248            |	   45        |	         21          |	     38     |	          4             |	                       250                    |	                  205                 |	                     250                          |	   184      |	    182      |	                     249                          |	      171         |	  250  |
|   top  |   Move to Heaven  | May 22, 2017 - Jul 11, 2017 |	  tvN        |  Wednesday, Thursday | 1 hr. 10 min. |	15+ - Teens 15 or older | Geu Roo is a young autistic man. He works for ... |	Psychological, Comedy, Romance, Drama | Autism, Uncle-Nephew Relationship, Death, Sava... |  Kim Won Suk   |  Kim Eun Hee  | Jo Jung Suk, Yoo Yeon Seok, Jung Kyung Ho, Kim...	|   Chorokbaem Media   |	  #1   |
|  freq  |	     1        |	          2             |	   49        |          40          |	     56     |	         216            |	                        1                     |	                   4                  |	                      1                           |	    5	  |       6       |	                      2                           |	       7          |	   1   |
     
### b. EDA : Univariate Analysis

Pada tahap ini, dilakukan analisis terhadap distribusi dan karakteristik masing-masing fitur dalam dataset.

#### Data Numerikal
<p align="center">
  <img src="https://github.com/user-attachments/assets/89444593-b044-4562-ac68-d22dd097c092" alt="Gambar 1" width="500"/>
  <br/>
  <b>Gambar 1. Total Episode yang Ditayangkan K-Drama Favorit</b>
</p>
Distribusi jumlah episode per K-Drama menunjukkan bahwa jumlah episode paling umum adalah sekitar 12 hingga 20 episode, dengan 16 episode menjadi yang paling sering muncul. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/d8391f1b-6751-4999-ba68-7392f359351d" alt="Gambar 2" width="500"/>
  <br/>
  <b>Gambar 2. Distribusi Rating K-Drama Favorit</b>
</p>
Histogram rating menunjukkan bahwa distribusi rating cenderung terkonsentrasi pada nilai yang tinggi, dengan sebagian besar K-Drama memiliki rating di atas 8.5, menunjukkan kualitas tinggi dari K-Drama dalam dataset ini.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d8c13904-5f2d-4dff-aabf-6d9fe2aa4ec8" alt="Gambar 3" width="500"/>
  <br/>
  <b>Gambar 3. Tren Jumlah K-Drama yang Rilis per Tahun</b>
</p>
Menunjukkan tren rilis K-Drama dalam "Top 250" dari tahun ke tahun. Kenaikan dalam beberapa tahun terakhir dapat mengindikasikan pertumbuhan industri atau preferensi penonton pada rilis terbaru.

#### Data Kategorikal
<p align="center">
  <img src="https://github.com/user-attachments/assets/9eb402cb-bb08-4b35-962f-8759480bf6fa" alt="Gambar 4" width="500"/>
  <br/>
  <b>Gambar 4. Genre K-Drama yang Paling Populer</b>
</p>
Romance, Drama, Mystery, Comedy, dan Thriller adalah genre yang mendominasi, menunjukkan bahwa K-Drama dalam daftar top ini cenderung memiliki genre-genre tersebut. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/ae6c8f96-0911-4cdd-b614-ffaf81101709" alt="Gambar 5" width="500"/>
  <br/>
  <b>Gambar 5. Top 10 Network Penyedia K-Drama Terbanyak</b>
</p>
Visualisasi ini mengidentifikasi 10 jaringan televisi yang paling banyak menayangkan K-Drama dalam dataset. Terlihat bahwa jaringan seperti tvN, Netflix, dan SBS adalah kontributor utama K-Drama top, menunjukkan peran penting mereka dalam industri.

<p align="center">
  <img src="https://github.com/user-attachments/assets/3947ed48-e3cb-428f-afde-845d6c7f71c9" alt="Gambar 6" width="500"/>
  <br/>
  <b>Gambar 6. Top 10 Tag yang Paling Sering Muncul di K-Drama</b>
</p>
Plot ini menampilkan 10 tag yang paling sering diasosiasikan dengan K-Drama. Tag-tag ini merepresentasikan tema, suasana, atau konsep yang paling umum ditemukan dalam K-Drama, seperti Strong Female Lead, Smart Female Lead, Smart Male Lead, dll. Hal ini membantu memahami karakteristik umum Top 250 K-Drama.

<p align="center">
  <img src="https://github.com/user-attachments/assets/3456a3b8-7e5d-467b-8a20-3ff44eba48d9" alt="Gambar 7" width="500"/>
  <br/>
  <b>Gambar 7. K-Drama Berdasarkan Rating Usia Penonton</b>
</p>
Plot ini menunjukkan distribusi K-Drama berdasarkan rating usia penonton. Terlihat bahwa kategori 15+ adalah rating yang paling umum, mengindikasikan bahwa sebagian besar K-Drama dalam dataset ini ditujukan untuk audiens berusia 15 tahun ke atas.

## Data Preparation
Tahap Data Preparation merupakan fondasi penting untuk memastikan data siap digunakan dalam pembangunan model rekomendasi. Pada tahap ini, dilakukan serangkaian proses pembersihan, transformasi, dan pemilihan fitur. Langkah-langkah yang dilakukan adalah sebagai berikut:

1.  **Pembersihan Kolom 'Rank':** Karakter `#` di kolom `Rank` dihilangkan, dan tipe data kolom tersebut diubah menjadi integer. Ini dilakukan untuk memastikan kolom peringkat memiliki format numerik yang konsisten.
2.  **Penanganan Missing Value:** Baris-baris data yang mengandung nilai yang hilang (missing value) dihapus dari dataset. Langkah ini penting untuk menghindari masalah atau bias yang mungkin timbul akibat data yang tidak lengkap.
3.  **Pemilihan Fitur:** Dari keseluruhan dataset, hanya kolom `Name`, `Year of release`, dan `Genre` yang dipilih untuk digunakan dalam proses pemodelan. Pemilihan ini didasarkan pada pendekatan rekomendasi berbasis konten yang berfokus pada informasi genre.
4.  **Pembersihan Data Genre:** Karakter strip (`-`) di dalam entri kolom `Genre` dihapus. Hal ini bertujuan untuk menyeragamkan format penulisan genre dan memastikan bahwa setiap genre dikenali dengan benar sebagai entitas terpisah.
5.  **Vektorisasi Genre dengan CountVectorizer:** Kolom `Genre` yang bersifat teks diubah menjadi representasi numerik menggunakan **CountVectorizer** dari library scikit-learn.
    *   CountVectorizer menghitung frekuensi kemunculan setiap genre unik di seluruh dataset.
    *   Proses ini menghasilkan matriks di mana baris merepresentasikan K-Drama, kolom merepresentasikan genre unik, dan nilainya adalah jumlah kemunculan genre tersebut dalam K-Drama.
    *   Metode `fit_transform()` digunakan untuk mempelajari kosa kata genre dan sekaligus mengubah data menjadi matriks hitungan.
    *   Metode `todense()` digunakan untuk mengonversi matriks hasil menjadi format dense (padat) untuk kemudahan manipulasi atau visualisasi.

Hasil dari tahapan Data Preparation adalah dataset yang lebih bersih, terfokus pada fitur relevan (terutama genre), dan siap untuk digunakan dalam membangun dan mengevaluasi model rekomendasi berbasis kesamaan konten.


## Modeling Process and Recommendation Results

Pada tahap ini, fokus utama adalah membangun arsitektur sistem rekomendasi yang dapat menyarankan K-Drama baru berdasarkan preferensi pengguna. Mengingat problem statement yang berfokus pada rekomendasi berbasis konten menggunakan fitur genre, pendekatan yang dipilih adalah **Content-Based Filtering**.

Teknik utama yang digunakan adalah menghitung **kemiripan antar item (K-Drama) berdasarkan representasi numerik dari genre**. Langkah-langkahnya adalah sebagai berikut:

1.  **Representasi Genre dengan CountVectorizer:** <br>
    Seperti yang telah dibahas pada tahap Data Preparation, kolom 'Genre' diubah menjadi matriks hitungan menggunakan `CountVectorizer`. Setiap K-Drama direpresentasikan sebagai vektor yang menunjukkan frekuensi kemunculan setiap genre unik.

2.  **Perhitungan Kemiripan dengan Cosine Similarity:** <br>
    Untuk mengukur seberapa mirip setiap pasangan K-Drama berdasarkan vektor genrenya, digunakan metrik **Cosine Similarity**.
    *   **Formula Cosine Similarity:** 
        ![formula cosine similarity](https://github.com/user-attachments/assets/f5a4b698-92cf-42e8-be59-34f95e07e1ae)
    *   **Cara Kerja Metrik:** Metrik ini mengukur kosinus dari sudut antara dua vektor. Nilai yang mendekati 1 menunjukkan sudut kecil (vektor sangat mirip arahnya, yang berarti genre-nya serupa), sedangkan nilai yang mendekati 0 menunjukkan sudut mendekati 90 derajat (tidak ada kemiripan genre). <br>

Tabel 3. Nilai *Cosine Similarity* antar 5 Judul K-Drama

| Name                    | Historical, Romance, Drama, Political | Friendship, Comedy, Youth, Sports | Action, Thriller, Drama, SciFi | Thriller, Mystery, SciFi | Mystery, Comedy, Romance, Life | Adventure, Historical, Romance, Drama | Life, Drama, Melodrama | Historical, Romance, Melodrama, Political | Mystery, Psychological, Drama, Family | Thriller, Romance, Drama, Melodrama |
| ----------------------- | ------------------------------------- | --------------------------------- | ------------------------------ | ------------------------ | ------------------------------ | ------------------------------------- | ---------------------- | ----------------------------------------- | ------------------------------------- | ----------------------------------- |
| Designated Survivor     | 0.50                                  | 0.00                              | 0.50000                        | 0.577350                 | 0.25                           | 0.25                                  | 0.288675               | 0.25                                      | 0.50                                  | 0.500000                            |
| Itaewon Class           | 0.50                                  | 0.00                              | 0.25000                        | 0.000000                 | 0.50                           | 0.50                                  | 0.577350               | 0.25                                      | 0.25                                  | 0.500000                            |
| Empress Ki              | 0.75                                  | 0.00                              | 0.00000                        | 0.000000                 | 0.25                           | 0.50                                  | 0.288675               | 1.00                                      | 0.00                                  | 0.500000                            |
| Big Mouth               | 0.25                                  | 0.00                              | 0.50000                        | 0.577350                 | 0.25                           | 0.25                                  | 0.288675               | 0.00                                      | 0.50                                  | 0.500000                            |
| My Name                 | 0.00                                  | 0.00                              | 0.57735                        | 0.333333                 | 0.00                           | 0.00                                  | 0.000000               | 0.00                                      | 0.00                                  | 0.288675                            |

3.  **Pembuatan Fungsi Rekomendasi (Top-N Recommendation):** <br>
    Sebuah fungsi `get_recommendations` dirancang untuk memberikan rekomendasi **Top-N**, di mana N dalam kasus ini adalah 10.
    *   Fungsi ini mengambil judul K-Drama yang disukai pengguna sebagai input.
    *   Berdasarkan matriks Cosine Similarity, fungsi menemukan 10 K-Drama lain yang memiliki skor kemiripan genre tertinggi dengan K-Drama input.
    *   Output dari fungsi ini adalah daftar **Top 10 K-Drama**, termasuk nama, tahun rilis, dan genre dari K-Drama yang direkomendasikan.

Tabel 4. Contoh Judul K-Drama yang Menjadi Objek Uji
|       | Name                  | Year of Release | Genre                              |
| ----- | --------------------- | --------------- | ---------------------------------- |
|   2   | **Hospital Playlist** | 2020            | Friendship, Romance, Life, Medical |

Tabel 5. Top 10 K-Drama Berdasarkan Kemiripan Fitur Genre (‘Hospital Playlist’)
|       | Name                                | Year of Release | Genre                                |
| ----- | ----------------------------------- | --------------- | -------------------------------------|
|   0   | Hospital Playlist 2                 | 2021            | Friendship, Romance, Life, Medical   |
|   1   | Age of Youth                        | 2016            | Friendship, Romance, Life, Youth     |
|   2   | Doctor John                         | 2019            | Mystery, Romance, Life, Medical      |
|   3   | If You Wish Upon Me                 | 2022            | Romance, Life, Drama, Medical        |
|   4   | Good Doctor                         | 2013            | Romance, Life, Drama, Medical        |
|   5   | Hometown Cha-Cha-Cha                | 2021            | Comedy, Romance, Life                |
|   6   | Dr. Romantic                        | 2016            | Romance, Drama, Medical              |
|   7   | Yumi's Cells 2                      | 2022            | Comedy, Romance, Life                |
|   8   | At a Distance, Spring Is Green      | 2021            | Romance, Life, Youth                 |
|   9   | D-Day                               | 2015            | Romance, Drama, Medical              |

### Kelebihan Pendekatan Content-Based Filtering (dengan CountVectorizer & Cosine Similarity)

*   **Tidak Membutuhkan Data Pengguna Lain:** Model hanya bergantung pada informasi konten (genre) dari item itu sendiri, sehingga bisa langsung memberikan rekomendasi bahkan untuk item baru atau pengguna baru (cold-start problem untuk item).
*   **Dapat Menjelaskan Rekomendasi:** Mudah untuk menjelaskan mengapa suatu item direkomendasikan (karena kemiripan genre).
*   **Menangkap Preferensi Spesifik Pengguna:** Jika sistem diperluas untuk pengguna individu, model dapat menyesuaikan rekomendasi berdasarkan riwayat interaksi pengguna dengan konten tertentu.

### Kekurangan Pendekatan Content-Based Filtering

*   **Keterbatasan pada Konten yang Disediakan:** Model hanya bisa merekomendasikan item yang mirip dengan item yang sudah disukai. Sulit untuk merekomendasikan item di luar "gelembung" konten yang familiar.
*   **Masalah Overspecialization:** Pengguna mungkin hanya akan direkomendasikan item yang sangat mirip, membatasi penemuan konten baru atau beragam.
*   **Membutuhkan Data Konten Terperinci:** Kualitas rekomendasi sangat bergantung pada kekayaan dan struktur data konten (dalam kasus ini, genre).

Dalam proyek ini, dipilih satu solusi rekomendasi berbasis Content-Based Filtering karena dataset yang tersedia sangat kaya akan informasi konten (terutama genre) dan tidak menyediakan data interaksi pengguna (seperti rating dari banyak pengguna) yang diperlukan untuk Collaborative Filtering.

## 4. Evaluation

Tahap Evaluation dilakukan untuk mengukur efektivitas sistem rekomendasi dalam memberikan rekomendasi yang relevan berdasarkan kemiripan genre.

**Metrik Evaluasi:**

Metrik evaluasi yang digunakan dalam konteks ini adalah **Precision**.

*   **Formula Precision:** 
   ![image](https://github.com/user-attachments/assets/b7b25458-93f3-46b2-9743-10f970631929)
    Di mana:
    *   **TP (True Positives):** Jumlah item yang direkomendasikan dan sebenarnya relevan (dalam konteks ini, memiliki genre yang sangat mirip dengan K-Drama input).
    *   **FP (False Positives):** Jumlah item yang direkomendasikan tetapi sebenarnya tidak relevan.

*   **Cara Kerja Metrik:** Precision mengukur **proporsi item yang relevan di antara semua item yang direkomendasikan**. Nilai Precision yang tinggi (mendekati 1.0) berarti bahwa mayoritas rekomendasi yang diberikan benar-benar relevan.

**Hasil Proyek Berdasarkan Metrik Evaluasi:**

Model rekomendasi diuji dengan memberikan input K-Drama "Hospital Playlist". Sistem memberikan daftar **Top 10 rekomendasi**. Berdasarkan analisis kualitatif terhadap genre dari 10 K-Drama yang direkomendasikan, diasumsikan bahwa semua rekomendasi tersebut memiliki kemiripan genre yang tinggi dengan "Hospital Playlist".

Dengan perhitungan berikut : <br>
![Screenshot 2025-06-07 201927](https://github.com/user-attachments/assets/6abf2a53-21fd-41c7-8ba8-18d66ae8a71c)


**Kesimpulan Hasil Evaluasi:**

Hasil perhitungan Precision sebesar 100% menunjukkan bahwa, **berdasarkan perhitungan evaluasi bahwa rekomendasi dengan genre yang mirip dianggap relevan**, model ini sangat efektif dalam menghasilkan daftar 10 K-Drama teratas yang memiliki kesamaan genre dengan K-Drama input. Ini mengindikasikan bahwa pendekatan Content-Based Filtering berbasis CountVectorizer dan Cosine Similarity bekerja dengan baik dalam menangkap dan memanfaatkan informasi genre untuk rekomendasi.

Penting untuk diingat bahwa evaluasi ini menggunakan asumsi sederhana tentang relevansi berdasarkan genre. Evaluasi yang lebih robust mungkin melibatkan perbandingan dengan penilaian manual atau data preferensi pengguna yang sebenarnya jika tersedia.

## Referensi 

1. Lee, Y. J., & Park, J. H. (2024). *How You Like That? Development of a Korean Drama Recommendation System Through Sentiment Analysis*. ResearchGate. [https://www.researchgate.net/publication/379107215_How_You_Like_That_Development_of_a_Korean_Drama_Recommendation_System_Through_Sentiment_Analysis](https://www.researchgate.net/publication/379107215_How_You_Like_That_Development_of_a_Korean_Drama_Recommendation_System_Through_Sentiment_Analysis)

2. Patil, K. R., & Bhatlawande, K. D. (2023). *Movie Recommendation System Using Sentiment Analysis from Reviews*. *International Journal of Advanced Research in Computer Science*. [https://journals.indexcopernicus.com/api/file/viewByFileId/1481563](https://journals.indexcopernicus.com/api/file/viewByFileId/1481563)

3. Ramoliya, K. (2023, October 10). *What is Cosine Similarity and how is it useful for text embeddings?* Medium. [https://medium.com/@KeyurRamoliya/what-is-cosine-similarity-and-how-is-it-useful-for-text-embeddings-7c47c65ef08d](https://medium.com/@KeyurRamoliya/what-is-cosine-similarity-and-how-is-it-useful-for-text-embeddings-7c47c65ef08d)

4. Encord. (2023, November 20). *Classification Metrics: Accuracy, Precision, Recall*. Encord Blog. [https://encord-com.translate.goog/blog/classification-metrics-accuracy-precision-recall/?_x_tr_sl=en&_x_tr_tl=id&_x_tr_hl=id&_x_tr_pto=imgs](https://encord-com.translate.goog/blog/classification-metrics-accuracy-precision-recall/?_x_tr_sl=en&_x_tr_tl=id&_x_tr_hl=id&_x_tr_pto=imgs)
