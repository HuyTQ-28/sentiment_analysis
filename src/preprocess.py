import os
import re
import unicodedata
import pandas as pd
from pathlib import Path
from html import unescape
import emoji
from tqdm import tqdm
from sklearn.model_selection import train_test_split
tqdm.pandas()

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "text.csv"
PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_FILE_PATH = PROCESSED_DATA_DIR / "processed_data.csv"

def _replace_urls_emails_phones(text):
    """Thay thế URLs, emails, và số điện thoại bằng khoảng trắng."""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", text)
    text = re.sub(r"\+?\d[\d\-\s]{7,}\d", " ", text)
    return text

def clean_text(text: str) -> str:
    """
    Pipeline hoàn chỉnh để làm sạch và tiền xử lý văn bản cho mô hình BERT-uncased.
    """
    if not isinstance(text, str):
        text = str(text)
        
    text = unescape(text) 
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = _replace_urls_emails_phones(text)
    text = emoji.demojize(text, language="en")
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def preprocess_data(df):
    """Hàm tiền xử lý dữ liệu, bao gồm làm sạch, xóa NaN và xóa trùng lặp."""
    initial_rows = len(df)
    print(f"Số dòng ban đầu: {initial_rows}")
    
    df['text'] = df['text'].progress_apply(clean_text)
    
    df.dropna(subset=['text', 'label'], inplace=True)
    rows_after_na = len(df)
    print(f"Đã xóa {initial_rows - rows_after_na} dòng chứa giá trị NaN.")

    df.drop_duplicates(subset=['text'], keep='first', inplace=True)
    rows_after_duplicates = len(df)
    print(f"Đã xóa {rows_after_na - rows_after_duplicates} dòng trùng lặp.")
    
    return df

def sample_data(df):
    label_column_name = 'label'
    
    # Lấy số lượng dòng nhỏ nhất trong các nhãn
    label_counts = df[label_column_name].value_counts()
    sample_size_per_label = min(label_counts)

    sample_df = df.groupby(label_column_name, group_keys=False).apply(
        lambda x: x.sample(sample_size_per_label, random_state=42)
    )

    print(f"Số dòng sau khi lấy mẫu: {len(sample_df)}")
    return sample_df

def save_dataset(df, base_dir, split_name):
    """Hàm lưu dữ liệu đã xử lý vào file csv và lưu một file mẫu."""
    base_dir.mkdir(parents=True, exist_ok=True)
    csv_path = base_dir / f'{split_name}.csv'

    # Lưu dữ liệu đã xử lý vào file csv
    df.to_csv(csv_path, index=False)
    print(f'\nĐã lưu dữ liệu đã xử lý vào {csv_path}')

    # Lưu mẫu dữ liệu để kiểm tra
    sample_path = base_dir / f'{split_name}_sample.txt'
    with open(sample_path, 'w', encoding='utf-8') as f:
        f.write("--- MẪU DỮ LIỆU SAU KHI XỬ LÝ ---\n\n")
        sample_df = df.head(5)
        for i, row in sample_df.iterrows():
            f.write(f"--- Mẫu {i+1} ---\n")
            f.write(f"Text:\n{row['text']}\n\n")
            f.write(f"Label:\n{row['label']}\n\n")
def main():
    df = pd.read_csv(RAW_DATA_DIR)
    df.drop(columns=['Unnamed: 0'], inplace=True)
    
    print("\nBắt đầu tiền xử lý dữ liệu...")
    df_processed = preprocess_data(df)
    print("\nTiền xử lý hoàn tất.")
    print(f"Tổng số dòng sau khi xử lý: {len(df_processed)}")

    sample_df = sample_data(df_processed)

    # Chia dữ liệu thành 3 tập train, val, test
    train_df, temp_df = train_test_split(sample_df, test_size=0.2, random_state=42, stratify=sample_df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    save_dataset(train_df, PROCESSED_DATA_DIR, "train")
    save_dataset(val_df, PROCESSED_DATA_DIR, "val")
    save_dataset(test_df, PROCESSED_DATA_DIR, "test")

    print("\nĐã lưu dữ liệu đã xử lý vào các file csv.")

if __name__ == "__main__":
    main()