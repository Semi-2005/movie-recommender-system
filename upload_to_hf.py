"""
upload_to_hf.py — CineMatch ML Artifact Uploader
==================================================

Bu script, item_similarity_matrix.npy dosyasını Hugging Face Hub'a yükler.

Kullanım:
    python upload_to_hf.py --token hf_xxxxxxxxxx --username KULLANICI_ADINIZ

Yükleme tamamlandıktan sonra URL şu formatta olacak:
    https://huggingface.co/datasets/KULLANICI_ADINIZ/cinematch-artifacts/resolve/main/item_similarity_matrix.npy
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def upload(token: str, username: str):
    REPO_ID = f"{username}/cinematch-artifacts"
    NPY_PATH = Path("data/processed/collaborative_artifacts/item_similarity_matrix.npy")

    if not NPY_PATH.exists():
        print(f"❌ Hata: {NPY_PATH} bulunamadı!")
        print("   Önce şu komutu çalıştırın:")
        print("   python -m backend.app.models.train_collaborative")
        return

    size_mb = NPY_PATH.stat().st_size / (1024 ** 2)
    print(f"📦 Dosya: {NPY_PATH}")
    print(f"📏 Boyut: {size_mb:.0f} MB")
    print(f"📤 Hedef: {REPO_ID}")
    print()

    api = HfApi(token=token)

    # Dataset repo oluştur (zaten varsa hata vermez)
    print("🔧 Hugging Face dataset repo oluşturuluyor (yoksa)...")
    create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        private=False,   # Public → startup.sh token gerektirmeden indirebilir
        token=token,
        exist_ok=True,
    )
    print(f"✅ Repo hazır: https://huggingface.co/datasets/{REPO_ID}")
    print()

    # Dosyayı yükle
    print(f"📤 {NPY_PATH.name} yükleniyor... (Bu birkaç dakika sürebilir)")
    api.upload_file(
        path_or_fileobj=str(NPY_PATH),
        path_in_repo="item_similarity_matrix.npy",
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    download_url = (
        f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/item_similarity_matrix.npy"
    )

    print()
    print("=" * 60)
    print("✅ Yükleme tamamlandı!")
    print()
    print("📋 Render'da şu env variable'ı ayarlayın:")
    print()
    print(f"  SIM_MATRIX_URL = {download_url}")
    print()
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload similarity matrix to Hugging Face Hub")
    parser.add_argument("--token", required=True, help="Hugging Face Write token (hf_xxx...)")
    parser.add_argument("--username", required=True, help="Hugging Face kullanıcı adınız")
    args = parser.parse_args()

    upload(args.token, args.username)
