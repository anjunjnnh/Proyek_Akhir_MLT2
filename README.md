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
| mean	|   2018.25600	  |      19.064000	   |   8.534000 |
|  std	|      3.26452	  |      13.245743	   |   0.221359 |
|  min  |	  2003.00000    |	      1.000000	   |   8.300000 |
|  25%	|   2017.00000	  |      16.000000	   |   8.300000 |
|  50%  |	  2019.00000	  |      16.000000	   |   8.500000 |
|  75%	|   2021.00000	  |      20.000000	   |   8.700000 |
|  max  |	  2022.00000	  |     133.000000	   |   9.200000 |

Tabel 2. Deskripsi Statistik Data Kategorikal

|        |      Name	    |           Aired Date        | Original Network |	     Aired On	      |    Duration  	|      Content Rating     |	                     Synopsis                     |	                 Genre                |	                     Tags                         |	  Director  |	Screenwriter |	                     Cast                         | Production companies |	 Rank  |
|  :---: |   :----------: |  :------------------------: | :--------------: | :------------------: | :-----------: | :---------------------: | :-----------------------------------------------: | :-----------------------------------: | :-----------------------------------------------: |  :--------: | :----------: | :------------------------------------------------: | :------------------: | :-----: |
|  count |	     250      |	             250            |	      250        |	       250          |	     250      |	          245           |	                       250                        |	                  250                 |	                     250                          |	    249     |	     249     |	                     250                          |	         248         |	 250   |
| unique |	     250      |	             248            |	       45        |	        21          |	      38      |	            4           |	                       250                        |	                  205                 |	                     250                          |	    184     |	     182     |	                     249                          |	         171         |	 250   |
|   top	 | Move to Heaven |	May 22, 2017 - Jul 11, 2017 |	      tvN        |  Wednesday, Thursday	| 1 hr. 10 min. |	15+ - Teens 15 or older	| Geu Roo is a young autistic man. He works for ... |	Psychological, Comedy, Romance, Drama	| Autism, Uncle-Nephew Relationship, Death, Sava... |	Kim Won Suk	|  Kim Eun Hee | Jo Jung Suk, Yoo Yeon Seok, Jung Kyung Ho, Kim...	|   Chorokbaem Media   |	 #1    |
|  freq  |	      1       |	              2             |	       49        |          40          |	      56      |	          216           |	                        1                         |	                   4                  |	                      1                           |	     5	    |       6      |	                      2                           |	          7          |	  1    |
     
## b. EDA : Univariate Analysis

Pada tahap ini, dilakukan analisis terhadap distribusi dan karakteristik masing-masing fitur dalam dataset.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8bf507e6-c818-4a31-975f-774d3581d60c" alt="Gambar 1" width="500"/>
  <br/>
  <b>Gambar 1. Banyak Total Episode yang Digunakan K-Drama Favorit</b>
</p>
Distribusi jumlah episode per K-Drama menunjukkan bahwa jumlah episode paling umum adalah sekitar 12 hingga 20 episode, dengan 16 episode menjadi yang paling sering muncul.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d8391f1b-6751-4999-ba68-7392f359351d" alt="Gambar 1" width="500"/>
  <br/>
  <b>Gambar 2. Distribusi Rating K-Drama Favorit</b>
</p>
Histogram rating menunjukkan bahwa distribusi rating cenderung terkonsentrasi pada nilai yang tinggi, dengan sebagian besar K-Drama memiliki rating di atas 8.5, menunjukkan kualitas tinggi dari K-Drama dalam dataset ini.


## Data Preparation
Tahap ini mempersiapkan data K-Drama untuk pembangunan model Content-Based Filtering. Proses ini meliputi langkah-langkah sebagai berikut:

1.  **Membersihkan Kolom 'Rank'**: Menghilangkan karakter '#' pada kolom 'Rank' dan mengubah tipe data kolom tersebut menjadi integer.
2.  **Menangani Missing Value**: Menghapus baris yang mengandung nilai-nilai yang hilang pada dataset.
3.  **Memilih Fitur yang Relevan**: Memilih kolom-kolom yang relevan dengan konten K-Drama yang akan digunakan untuk rekomendasi. Kolom-kolom yang dipilih meliputi 'Name', 'Original Network', 'Synopsis', 'Genre', 'Tags', 'Director', dan 'Cast'.
4.  **Mengganti Missing Value pada Fitur Terpilih**: Mengganti nilai-nilai yang hilang pada fitur-fitur terpilih dengan string kosong (''). Ini dilakukan untuk memastikan bahwa semua fitur yang akan digabungkan memiliki nilai string dan tidak menyebabkan error saat pemrosesan teks.
5.  **Menggabungkan Fitur Terpilih**: Menggabungkan konten dari semua fitur terpilih menjadi satu string tunggal untuk setiap K-Drama. String gabungan ini akan menjadi representasi teks dari setiap K-Drama yang akan digunakan untuk perhitungan kemiripan.

