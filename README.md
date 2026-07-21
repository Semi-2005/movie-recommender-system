# 🎬 CineMatch: Hybrid Ensembled Movie Recommendation Engine 🚀

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green)
![React](https://img.shields.io/badge/React-18-blue)
![Vite](https://img.shields.io/badge/Vite-5-purple)
![Architecture](https://img.shields.io/badge/Architecture-Decoupled_Microservices-orange)

## 📌 Mimari Vizyon (Architectural Vision)
CineMatch, ölçeklenebilir ve yüksek doğruluklu (high-accuracy) bir film öneri sistemi sunmak amacıyla tasarlanmış modern bir web uygulamasıdır. Projenin kalbinde, yalnızca içerik (content) veya yalnızca işbirlikçi filtreleme (collaborative filtering) algoritmalarına bağlı kalmak yerine, bu iki yaklaşımı dinamik olarak harmanlayan **Adaptive Hybrid Ensemble** (Uyarlanabilir Hibrit Topluluk) modeli yatmaktadır. 

Bir Senior Software Architect perspektifiyle; sistemimiz Client-Server (İstemci-Sunucu) ayrımına katı bir şekilde uyar. Frontend tarafı, yüksek performanslı DOM manipülasyonu için React & Vite ile inşa edilmişken, Backend tarafı yüksek I/O ve asenkron operasyon kapasitesi sebebiyle FastAPI ile tasarlanmıştır. ML modellerimiz, runtime performansını artırmak adına memory-resident (bellekte yerleşik) veya pre-computed (önceden hesaplanmış) matrisler halinde sunulur.

## ✨ Çekirdek Sistem Özellikleri (Core System Features)
- **🧠 Adaptive Hybrid Recommendation (Uyarlanabilir Hibrit Öneri):** Kullanıcı etkileşiminin az olduğu durumlarda (Cold Start) içerik bazlı (content-based), yeterli verinin olduğu durumlarda işbirlikçi (collaborative) ağırlığı artıran, dinamik $\alpha$ (alpha) katsayılı füzyon algoritması.
- **⚡ Yüksek Performanslı API:** FastAPI ve Uvicorn altyapısı sayesinde asenkron (async/await) endpoint yönetimi ve düşük gecikme (low-latency) süresi.
- **🔍 Fuzzy Matching (Bulanık Arama):** Kullanıcıların yazım hatalarına karşı toleranslı, mesafelendirme algoritmalarıyla güçlendirilmiş film arama motoru.
- **🚀 Modern UI/UX:** React 18, Vite ve Lucide-React ikon seti ile donatılmış, responsive ve bileşen (component) bazlı arayüz.
- **📊 Modüler Veri Bilimi Boru Hattı (Data Science Pipeline):** Deneysel EDA (Keşifçi Veri Analizi) ve model doğrulama süreçleri bağımsız `notebooks/` dizininde izole edilmiştir.

## 🛠 Teknoloji Yığını (Tech Stack)
| Katman (Layer) | Teknolojiler (Technologies) | Motivasyon (Motivation) |
| --- | --- | --- |
| **Frontend** | React 18, Vite, React Router DOM | Hızlı HMR (Hot Module Replacement), bileşen yeniden kullanılabilirliği (component reusability) ve client-side routing. |
| **Backend** | FastAPI, Uvicorn, Pydantic | Veri validasyonu (Data Validation), OpenAPI entegrasyonu, Asenkron I/O destekli. |
| **Data / ML** | Pandas, Numpy, Scikit-Learn, Scipy | Vektörel matris işlemleri (Vectorized Matrix Ops), Kosinüs Benzerliği (Cosine Similarity) algoritmaları. |

## 🏗 Sistem Mimarisi & Klasör Yapısı (System Architecture & Folder Structure)
Projemiz Separation of Concerns (İlgi Alanlarının Ayrımı) prensibine göre modülerleştirilmiştir:

```text
movie-recommender-system/
├── backend/                  # Python/FastAPI Sunucu ve ML Inference Katmanı
│   ├── app/
│   │   ├── api/              # Route Controller'ları (routes.py)
│   │   ├── services/         # İş mantığı (recommender.py, movie_identity.py)
│   │   └── main.py           # Application Entrypoint
│   └── requirements.txt      # Backend bağımlılıkları
├── frontend/                 # React SPA (Single Page Application)
│   ├── src/
│   │   ├── components/       # Tekrar kullanılabilir UI parçaları (SearchBar, TabBar vb.)
│   │   ├── pages/            # Sayfa görünümleri (Views)
│   │   └── services/         # Backend API wrapper'ları
│   ├── package.json          # Node bağımlılıkları
│   └── vite.config.js        # Vite bundler konfigürasyonu
├── data/                     # Raw ve Processed Datasetler (movie.csv, rating.csv)
├── notebooks/                # Model Ar-Ge (R&D) ve EDA (.ipynb dosyaları)
└── tests/                    # Birim (Unit) ve Entegrasyon (Integration) Testleri
```

## 🔌 API Referansı (API Reference)
Backend, `/docs` endpoint'i üzerinden tam interaktif bir Swagger UI sunar. Temel endpoint'ler:
- `GET /recommend?movie={title}&top_n={n}` : Saf içerik bazlı öneriler.
- `GET /recommend/collaborative?movie={title}` : Item-based collaborative filtering önerileri.
- `GET /recommend/hybrid?movie={title}` : Gelişmiş ağırlıklı ensemble (hibrit) tahminleri.
- `GET /search?q={query}` : Film kataloğunda fuzzy arama.

## 🚀 Kurulum ve Ortam Hazırlığı (Setup & Initialization)

Projeyi lokal geliştirme (development) ortamında ayağa kaldırmak için aşağıdaki adımları izleyin:

### 1. Veri Seti Hazırlığı (Data Preparation)
Öneri motorunun çalışması için `data/` klasöründe `movie.csv` ve `rating.csv` bulunmalıdır. Matrislerin önceden hesaplanmış (pre-computed) versiyonları `data/processed/` dizini içerisine oluşturulmalıdır.

### 2. Backend Kurulumu
Backend, bağımlılık izolasyonunu sağlamak için sanal ortamda (virtualenv) çalıştırılmalıdır.
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
API artık `http://localhost:8000` adresinde yayına başlayacaktır.

### 3. Frontend Kurulumu
Node.js ortamında (>=18.x) aşağıdaki komutları çalıştırın:
```bash
cd frontend
npm install
npm run dev
```
Uygulama arayüzü `http://localhost:5173` adresinde render edilecektir.

## 🔮 Gelecek Yol Haritası (Roadmap & Future Scaling)
Sistemin genişletilmesi adına bir sonraki mimari iterasyonda (Next Iteration) odaklanılacak performans iyileştirmeleri:
1. **Caching Layer (Önbellekleme):** Sık sorgulanan (hot) filmler için öneri sonuçlarını Redis önbelleğine almak. Bu, I/O yükünü düşürüp, request maliyetini O(1) hızına çekecektir.
2. **Horizontal Scaling (Yatay Ölçeklendirme):** Gelen trafiğin artması durumunda FastAPI worker'larının Docker Container'lar içerisine alınıp Kubernetes veya Docker Swarm üzerinde cluster yapıya geçirilmesi.
3. **Implicit Feedback Pipeline:** Tıklama (Click-through) verilerinin gerçek zamanlı Message Broker'lar (Kafka/RabbitMQ) ile modele anlık aktarımı.