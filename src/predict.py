import os
import torch
import pandas as pd

from sklearn.metrics import classification_report

def predict_on_test_set(model, tokenizer, config):
    """Đánh giá mô hình trên toàn bộ tập dữ liệu test."""
    print("\n--- Bắt đầu đánh giá trên tập test ---")
    
    test_file_path = config["test_file"]
    if not os.path.exists(test_file_path):
        print(f"Lỗi: Không tìm thấy file test tại '{test_file_path}'")
        return
        
    test_df = pd.read_csv(test_file_path)
    texts = test_df.text.tolist()
    true_labels = test_df.label.tolist()
    
    id2label = config["id2label"]
    device = config["device"]
    
    model.to(device)
    model.eval()
    
    predictions = []
    print(f"Đang chạy dự đoán trên {len(texts)} mẫu của tập test...")
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", max_length=config["max_seq_length"], padding="max_length", truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=1).item()
            predictions.append(pred_id)
            
    target_names = [id2label[i] for i in sorted(id2label.keys())]
    print("\n--- BÁO CÁO PHÂN LOẠI CHI TIẾT ---")
    print(classification_report(true_labels, predictions, target_names=target_names, digits=4))
    print("--------------------------------------")

def predict_single_sentence(model, tokenizer, config, text):
    """Dự đoán cảm xúc cho một câu văn đơn lẻ."""
    id2label = config["id2label"]
    device = config["device"]
    model.to(device)
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", max_length=config["max_seq_length"], padding="max_length", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=1).item()

    return id2label[pred_id]