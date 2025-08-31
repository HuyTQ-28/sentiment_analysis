import modal
import sys

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("src")
)

app = modal.App(name="sentiment-analysis", image=image)

VOLUME_PATH = "/data"

volume = modal.Volume.from_name("sentiment_data", create_if_missing=True)

@app.function(
    gpu="A100",
    volumes={str(VOLUME_PATH): volume},
    timeout=7200,
)

def train_model():
    from src.train import train as training_main
    from src.config import get_config

    original_get_config = get_config
    def get_modal_config():
        print(f"Loading config with base path: {VOLUME_PATH}")
        return original_get_config(base_path=VOLUME_PATH)

    setattr(sys.modules["src.config"], "get_config", get_modal_config)
    import src.train
    src.train.get_config = get_modal_config
    
    try:
        training_main()
    except Exception:
        import traceback
        print("Đã xảy ra lỗi trong quá trình huấn luyện:")
        traceback.print_exc()
        raise

    print("Huấn luyện hoàn tất!")

if __name__ == "__main__":
    with app.run():
        train_model.remote()