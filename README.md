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
- Bagaimana membangun sistem rekomendasi berbasis konten yang efektif untuk K-Drama menggunakan teknik seperti TF-IDF Vectorization dan Cosine Similarity?
- Bagaimana cara mengukur kesamaan antar K-Drama berdasarkan fitur-fitur konten (sinopsis, genre, pemeran, dll.) dan menggunakan kesamaan tersebut untuk menghasilkan rekomendasi?
- Bagaimana sistem dapat menangani input dari pengguna (nama K-Drama) dan memberikan daftar rekomendasi K-Drama yang relevan dan personal?

### Goals
- Untuk membangun sistem rekomendasi K-Drama berbasis konten menggunakan metode seperti TF-IDF Vectorizer dan Cosine Similarity.
- Untuk menghasilkan rekomendasi K-Drama yang relevan bagi pengguna berdasarkan kemiripan konten.
- Untuk menyediakan antarmuka yang memungkinkan pengguna untuk memasukkan nama K-Drama dan menerima daftar rekomendasi yang disarankan.

### Solution Statement
Proyek ini mengusulkan solusi berbasis machine learning untuk mengatasi tantangan penemuan konten dalam industri hiburan. Solusi ini melibatkan pembangunan sistem rekomendasi berbasis konten untuk K-Drama. Sistem ini akan menggunakan teknik Natural Language Processing (NLP) dan aljabar linear. Data K-Drama akan melalui proses persiapan untuk menggabungkan fitur-fitur konten yang relevan (nama, sinopsis, genre, tag, sutradara, dan pemeran). Kemudian, TF-IDF Vectorizer akan digunakan untuk mengubah teks gabungan menjadi representasi numerik. Cosine Similarity akan dihitung antar K-Drama berdasarkan vektor fitur ini untuk mengukur tingkat kemiripan. Sistem akan mengambil input nama K-Drama dari pengguna, menemukan K-Drama yang paling mirip, dan mengurutkan K-Drama lain berdasarkan skor kesamaan untuk menghasilkan daftar rekomendasi. Solusi ini bertujuan untuk memberikan pengguna alat yang efektif untuk menemukan K-Drama baru yang sesuai dengan selera mereka, meningkatkan pengalaman menonton, dan mendukung platform hiburan dalam meningkatkan engagement dan retensi pengguna.

## Data Understanding

Tahap ini berfokus pada pemahaman mendalam terhadap dataset sebelum memulai pemrosesan dan pemodelan. Ini merupakan langkah krusial untuk memastikan data siap digunakan dan untuk mendapatkan wawasan awal.

Dataset ini bersumber dari: **https://www.kaggle.com/datasets/ahbab911/top-250-korean-dramas-kdrama-dataset/**. Dataset ini berisi informasi tentang berbagai K-Drama, termasuk nama, sinopsis, genre, dan pemeran.

Setelah data berhasil dimuat, pemeriksaan awal menunjukkan dataset memiliki **250 baris dan 17 kolom**.

Pemeriksaan tipe data dan non-null count menggunakan `.info()` memberikan gambaran tentang jenis data yang terkandung dalam setiap kolom dan keberadaan nilai yang hilang.`.isnull().sum()` secara spesifik mengkonfirmasi tidak ada nilai yang hilang di dataset ini. Analisis statistik deskriptif menggunakan `.describe().T` merangkum distribusi fitur numerik seperti nilai rata-rata, standar deviasi, nilai minimum dan maksimum, serta kuartil, memberikan wawasan tentang rentang dan penyebaran data.

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

## Data Preparation
Tahap ini mempersiapkan data K-Drama untuk pembangunan model Content-Based Filtering. Proses ini meliputi langkah-langkah sebagai berikut:

1.  **Membersihkan Kolom 'Rank'**: Menghilangkan karakter '#' pada kolom 'Rank' dan mengubah tipe data kolom tersebut menjadi integer.
2.  **Menangani Missing Value**: Menghapus baris yang mengandung nilai-nilai yang hilang pada dataset.
3.  **Memilih Fitur yang Relevan**: Memilih kolom-kolom yang relevan dengan konten K-Drama yang akan digunakan untuk rekomendasi. Kolom-kolom yang dipilih meliputi 'Name', 'Original Network', 'Synopsis', 'Genre', 'Tags', 'Director', dan 'Cast'.
4.  **Mengganti Missing Value pada Fitur Terpilih**: Mengganti nilai-nilai yang hilang pada fitur-fitur terpilih dengan string kosong (''). Ini dilakukan untuk memastikan bahwa semua fitur yang akan digabungkan memiliki nilai string dan tidak menyebabkan error saat pemrosesan teks.
5.  **Menggabungkan Fitur Terpilih**: Menggabungkan konten dari semua fitur terpilih menjadi satu string tunggal untuk setiap K-Drama. String gabungan ini akan menjadi representasi teks dari setiap K-Drama yang akan digunakan untuk perhitungan kemiripan.

