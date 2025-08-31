import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame chứa hai cột 'text' và 'label'.
            tokenizer: Tokenizer từ thư viện Hugging Face.
            max_len (int): Độ dài tối đa của chuỗi sau khi tokenization.
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text.to_list()
        self.labels = dataframe.label.to_list()
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        
        text = str(self.text[index])
        label = self.labels[index]

        # Thực hiện tokenization trên văn bản đã được xử lý
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
